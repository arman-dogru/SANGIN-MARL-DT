"""Propagate archived OMM data and build a Mont-Tremblant backhaul schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs
import skyfield
from panama_uav.leo import C_KM_S, handshake, link_geometry
from panama_uav.leo_radio import RadioAssumptions, atmospheric_curves, link_budget, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default="data/leo/starlink_omm.json")
    parser.add_argument("--output", default="data/leo/tremblant_full.npz")
    parser.add_argument("--start", default="2026-08-31T00:00:00+00:00")
    parser.add_argument("--duration", type=int, default=21600)
    parser.add_argument("--latitude", type=float, default=46.2)
    parser.add_argument("--longitude", type=float, default=-74.6)
    parser.add_argument("--elevation-mask", type=float, default=25)
    parser.add_argument("--gateway-latitude", type=float, default=45.3)
    parser.add_argument("--ground-delay", type=float, default=0.01)
    parser.add_argument("--schedule-seconds", type=float, default=15.0)
    parser.add_argument("--interruption-seconds", type=float, default=0.05)
    parser.add_argument("--rain-exceedance-percent", type=float, default=1.0)
    args = parser.parse_args()
    source = PROJECT / args.elements
    raw = source.read_bytes()
    elements = json.loads(raw)
    start = datetime.fromisoformat(args.start).astimezone(UTC)
    times = np.arange(args.duration + 1, dtype=float)
    jd = start.timestamp() / 86400 + 2440587.5
    dates = jd + times / 86400
    jd_whole = np.floor(dates)
    jd_fraction = dates - jd_whole
    ts = load.timescale(builtin=True)
    sky_times = ts.tt_jd(ts.from_datetime(start).tt+times/86400)
    def site(latitude, longitude, altitude):
        point = wgs84.latlon(latitude, longitude, elevation_m=altitude*1000)
        lat, lon = np.radians([latitude, longitude])
        return point.itrs_xyz.km, np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    observer, up = site(args.latitude, args.longitude, 0.44)
    gateway, gateway_up = site(args.gateway_latitude, args.longitude, 0.3)
    # This is an operational-altitude proxy, not a claim that every satellite is serving users.
    records, satellites = [], []
    rejected = {"epoch": 0, "orbit": 0, "initialization": 0}
    for record in elements:
        epoch = datetime.fromisoformat(record["EPOCH"]).replace(tzinfo=UTC)
        if abs((start - epoch).total_seconds()) > 3 * 86400:
            rejected["epoch"] += 1
            continue
        try:
            sat = Satrec()
            omm.initialize(sat, record)
        except (ValueError, KeyError):
            rejected["initialization"] += 1
            continue
        if not 350 <= sat.altp * sat.radiusearthkm <= 700:
            rejected["orbit"] += 1
            continue
        satellites.append(sat)
        records.append(record)
    if not satellites:
        raise ValueError("No usable archived orbital elements near the requested epoch")
    best_elevation = np.full(len(times), -90.0)
    best_satellite = np.full(len(times), -1, dtype=int)
    visible_counts = np.zeros(len(times), dtype=int)
    passes = {}
    error_samples = 0
    for first in range(0, len(satellites), 32):
        error, r, v = SatrecArray(satellites[first : first + 32]).sgp4(
            jd_whole, jd_fraction
        )
        error_samples += int(np.count_nonzero(error))
        # Maintained Skyfield frame objects include Earth rotation in velocity.
        earth_fixed = [EarthSatellite.from_satrec(sat, ts).at(sky_times).frame_xyz_and_velocity(itrs)
                       for sat in satellites[first:first+32]]
        r = np.stack([position.km.T for position,velocity in earth_fixed])
        v = np.stack([velocity.km_per_s.T for position,velocity in earth_fixed])
        elevation, distance, rate = link_geometry(r, v, observer, up)
        gateway_elevation, gateway_distance, gateway_rate = link_geometry(
            r, v, gateway, gateway_up
        )
        visible = (
            (error == 0)
            & (elevation >= args.elevation_mask)
            & (gateway_elevation >= 10)
        )
        visible_counts += visible.sum(axis=0)
        score = np.where(visible, elevation, -90)
        winner = score.argmax(axis=0)
        maximum = score[winner, np.arange(len(times))]
        improve = maximum > best_elevation
        best_elevation[improve] = maximum[improve]
        best_satellite[improve] = first + winner[improve]
        for local in range(len(r)):
            satellite_id = first + local
            edges = np.diff(np.r_[False, visible[local], False].astype(int))
            passes[satellite_id] = []
            for left, right in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            ):
                passes[satellite_id].append(
                    {
                        "left": int(left),
                        "right": int(right),
                        "elevation": elevation[local, left:right].astype(np.float32),
                        "space_propagation": (
                            (
                                distance[local, left:right]
                                + gateway_distance[local, left:right]
                            )
                            / C_KM_S
                        ).astype(np.float32),
                        "range_rate": rate[local, left:right].astype(np.float32),
                        "range": distance[local, left:right].astype(np.float32),
                        "gateway_range": gateway_distance[local,left:right].astype(np.float32),
                        "gateway_elevation": gateway_elevation[local,left:right].astype(np.float32),
                        "gateway_rate": gateway_rate[local,left:right].astype(np.float32),
                        "ecef": r[local,left:right].astype(np.float32),
                    }
                )
        if first % 512 == 0:
            print(f"Propagated {first}/{len(satellites)} satellites", flush=True)
    selected = np.full(len(times), -1, dtype=int)
    propagation = np.full(len(times), np.nan)
    elevations = np.full(len(times), np.nan)
    ranges = np.full(len(times), np.nan)
    rates = np.full(len(times), np.nan)
    gateway_ranges = np.full(len(times), np.nan)
    gateway_elevations = np.full(len(times), np.nan)
    gateway_rates = np.full(len(times), np.nan)
    satellite_ecef = np.full((len(times),3), np.nan, dtype=np.float32)
    blocked = []
    events = []
    cursor, previous = 0, -1
    while cursor < len(times):
        satellite = best_satellite[cursor]
        if satellite < 0:
            left = cursor
            while cursor < len(times) and best_satellite[cursor] < 0:
                cursor += 1
            blocked.append((float(left), min(float(cursor), times[-1]), 0))
            previous = -1
            continue
        contact = next(p for p in passes[satellite] if p["left"] <= cursor < p["right"])
        right = min(contact["right"], int((np.floor(cursor/args.schedule_seconds)+1)*args.schedule_seconds), len(times))
        offset = cursor - contact["left"]
        portion = slice(offset, offset+right-cursor)
        norad = int(records[satellite]["NORAD_CAT_ID"])
        selected[cursor:right] = norad
        propagation[cursor:right] = (
            contact["space_propagation"][portion] + args.ground_delay
        )
        elevations[cursor:right] = contact["elevation"][portion]
        ranges[cursor:right] = contact["range"][portion]
        rates[cursor:right] = contact["range_rate"][portion]
        gateway_ranges[cursor:right] = contact["gateway_range"][portion]
        gateway_elevations[cursor:right] = contact["gateway_elevation"][portion]
        gateway_rates[cursor:right] = contact["gateway_rate"][portion]
        satellite_ecef[cursor:right] = contact["ecef"][portion]
        finish, control = handshake(
            float(cursor), float(contact["space_propagation"][offset])
        )
        if finish < cursor+args.interruption_seconds:
            control.append({"kind": "settling", "start": finish, "end": cursor+args.interruption_seconds})
            finish = cursor+args.interruption_seconds
        if finish <= times[-1]:
            blocked.append((float(cursor), finish, 1))
            events.append(
                {
                    "start": float(cursor),
                    "end": finish,
                    "from_satellite": previous,
                    "to_satellite": norad,
                    "initial_attachment": previous < 0,
                    "attachment_changed": previous != norad,
                    "event_type": "satellite_attachment" if previous != norad else "resource_reconfiguration",
                    "protocol": "assumed four-message handshake",
                    "control_messages": control,
                }
            )
        previous = norad
        cursor = right
    events_handover = [e for e in events if not e["initial_attachment"] and e["attachment_changed"]]
    parameters = RadioAssumptions(atmospheric_exceedance_percent=args.rain_exceedance_percent)
    site_curves = atmospheric_curves(args.latitude,args.longitude,.44,parameters)
    gateway_curves = atmospheric_curves(args.gateway_latitude,args.longitude,.3,parameters)
    budgets = {}
    for name, distance, elevation, direction, curves in (
        ("uav_up", ranges, elevations, "up", site_curves),
        ("uav_down", ranges, elevations, "down", site_curves),
        ("gateway_up", gateway_ranges, gateway_elevations, "up", gateway_curves),
        ("gateway_down", gateway_ranges, gateway_elevations, "down", gateway_curves),
    ):
        for quantity, values in link_budget(distance,elevation,direction,curves,parameters).items():
            budgets[name+"_"+quantity] = values
    up_capacity = np.minimum(budgets["uav_up_rate_bound_bps"],budgets["gateway_down_rate_bound_bps"])
    down_capacity = np.minimum(budgets["gateway_up_rate_bound_bps"],budgets["uav_down_rate_bound_bps"])
    source_info = json.loads((source.parent / "orbit_source.json").read_text())
    metadata = {
        "orbital_source": source_info,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_elements": len(elements),
        "used_elements": len(satellites),
        "rejected": rejected,
        "sgp4_error_samples": error_samples,
        "simulation_start_utc": start.isoformat(),
        "element_epoch_min": min(r["EPOCH"] for r in records),
        "element_epoch_max": max(r["EPOCH"] for r in records),
        "archived_epoch_replay": True,
        "duration_seconds": args.duration,
        "sample_interval_seconds": 1,
        "site": {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "altitude_km": 0.44,
        },
        "gateway": {
            "latitude": args.gateway_latitude,
            "longitude": args.longitude,
            "altitude_km": 0.3,
            "measured_provider_gateway": False,
        },
        "elevation_mask_degrees": args.elevation_mask,
        "gateway_elevation_mask_degrees": 10,
        "satellite_selection": "highest visible elevation at 15-second scheduling boundaries or visibility expiry",
        "scheduling_interval_seconds": args.schedule_seconds,
        "reconfiguration_interruption_seconds": args.interruption_seconds,
        "scheduled_event_count": len(events),
        "scheduling_basis": "SATPIPE observed 15 s / 50 ms; not provider schedule reconstruction",
        "provider_handover_schedule": False,
        "time_transform": "Skyfield EarthSatellite and ITRS frame objects with bundled UT1 timescale; zero polar motion",
        "libraries": {"skyfield": skyfield.__version__, "sgp4": "2.27"},
        "radio": provenance(parameters),
        "route": "UAV -> one serving LEO -> assumed visible gateway -> cloud; no ISL",
        "long_run_replay": "explicit repetition of this archived geometry window; no element extrapolation",
        "ground_delay_seconds": args.ground_delay,
        "handover_count": len(events_handover),
        "mean_contact_seconds": args.duration / max(len(events_handover), 1),
        "mean_handshake_seconds": float(
            np.mean([e["end"] - e["start"] for e in events])
        ),
        "no_visibility_seconds": int(np.count_nonzero(selected < 0)),
        "doppler": "-range_rate/carrier_wavelength; raw geometric shift, ideal compensation assumed",
        "mean_visible_satellites": float(visible_counts.mean()),
        "minimum_visible_satellites": int(visible_counts.min()),
    }
    output = PROJECT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=times,
        propagation_seconds=propagation,
        blocked_windows=np.asarray(blocked, dtype=float).reshape(-1, 3),
        handover_events=json.dumps(events),
        metadata=json.dumps(metadata),
        satellite_ids=selected,
        elevation_degrees=elevations,
        range_km=ranges,
        range_rate_km_s=rates,
        uplink_doppler_hz=-rates * 14.25e9 / C_KM_S,
        downlink_doppler_hz=-rates * 11.7e9 / C_KM_S,
        visible_satellites=visible_counts,
        satellite_up_bps=up_capacity,
        satellite_down_bps=down_capacity,
        gateway_range_km=gateway_ranges,
        gateway_elevation_degrees=gateway_elevations,
        gateway_range_rate_km_s=gateway_rates,
        satellite_ecef_km=satellite_ecef,
        **budgets,
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
