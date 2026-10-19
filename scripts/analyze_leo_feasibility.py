"""Independent orbit/handshake feasibility calculation; never changes training.

Outputs are modeled network geometry, not measured Starlink or policy results.
Historical 550-km FCC shell parameters; deliberately specified Walker phasing.
Spherical rotating Earth and circular two-body orbits, not current TLE/SGP4.
Individual control exchanges below are an assumed protocol, not proprietary
Starlink signaling. No losses/retries, RF link budget, terrain, or TCP emulation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EARTH_KM = 6371.0
MU = 398600.4418  # km^3/s^2
EARTH_RATE = 7.2921159e-5  # rad/s
LIGHT_KM_S = 299792.458
UL_HZ, DL_HZ = 14.25e9, 11.7e9
MODEL_BYTES, EXPERIENCE_BYTES = 6591156, 6108


def describe(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return {
        "mean": float(a.mean()),
        "min": float(a.min()),
        "p50": float(np.quantile(a, 0.5)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
        "samples": int(a.size),
    }


class Shell:
    def __init__(self, seed, altitude=550.0, planes=72, per_plane=22):
        self.radius = EARTH_KM + altitude
        self.rate = math.sqrt(MU / self.radius**3)
        rng = np.random.default_rng(seed)
        plane = np.repeat(np.arange(planes), per_plane)
        slot = np.tile(np.arange(per_plane), planes)
        self.node = 2 * math.pi * plane / planes + rng.uniform(0, 2 * math.pi)
        self.phase = (
            2 * math.pi * slot / per_plane
            + 2 * math.pi * plane / (planes * per_plane)
            + rng.uniform(0, 2 * math.pi)
        )
        self.inc = math.radians(53.0)

    def state(self, time, indices=None):
        node = self.node if indices is None else self.node[indices]
        phase = self.phase if indices is None else self.phase[indices]
        u = phase + self.rate * time
        cn, sn, cu, su = np.cos(node), np.sin(node), np.cos(u), np.sin(u)
        ci, si = math.cos(self.inc), math.sin(self.inc)
        xyz = self.radius * np.column_stack(
            (cn * cu - sn * su * ci, sn * cu + cn * su * ci, su * si)
        )
        vel = (
            self.radius
            * self.rate
            * np.column_stack(
                (-cn * su - sn * cu * ci, -sn * su + cn * cu * ci, cu * si)
            )
        )
        c, s = math.cos(EARTH_RATE * time), math.sin(EARTH_RATE * time)
        rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        xyz, vel = xyz @ rot.T, vel @ rot.T
        vel -= np.cross(np.array([0, 0, EARTH_RATE]), xyz)
        return xyz, vel


def site(latitude, longitude, height_km):
    lat, lon = np.radians([latitude, longitude])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return (EARTH_KM + height_km) * up, up


def geometry(xyz, vel, position, up):
    delta = xyz - position
    ranges = np.linalg.norm(delta, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(delta @ up / ranges, -1, 1)))
    radial = np.sum(delta * vel, axis=1) / ranges
    return elevation, ranges, radial


def longest_contact(shell, time, candidates, observer, gateway, mask):
    low, high = np.zeros(len(candidates)), np.full(len(candidates), 900.0)
    # Each candidate is already visible. No re-entry occurs within this horizon.
    # Individual candidate times require a direct circular state calculation.
    for _ in range(15):
        mid = (low + high) / 2
        t = time + mid
        u = shell.phase[candidates] + shell.rate * t
        node = shell.node[candidates] - EARTH_RATE * t
        cn, sn, cu, su = np.cos(node), np.sin(node), np.cos(u), np.sin(u)
        ci, si = math.cos(shell.inc), math.sin(shell.inc)
        xyz = shell.radius * np.column_stack(
            (cn * cu - sn * su * ci, sn * cu + cn * su * ci, su * si)
        )

        def elevation_at(location, xyz=xyz):
            pos, up = location
            d = xyz - pos
            return np.degrees(
                np.arcsin(np.clip(d @ up / np.linalg.norm(d, axis=1), -1, 1))
            )

        visible = (elevation_at(observer) >= mask) & (elevation_at(gateway) >= 10)
        low = np.where(visible, mid, low)
        high = np.where(visible, high, mid)
    return int(candidates[np.argmax(low)])


def handshake(shell, at, satellite, observer, gateway):
    """Sequential assumed request/grant/complete/ack; gateway control anchor.

    Control packets use 10/50 Mbps, plus both geometric space legs per message.
    Sync=5 ms, scheduling wait=5 ms, each remote response processing=5 ms.
    RF Doppler acquisition/compensation is assumed successful; no retries.
    """
    stages = []
    time = at

    def stage(name, duration, **extra):
        nonlocal time
        stages.append({"stage": name, "start": time, "end": time + duration, **extra})
        time += duration

    stage("target_downlink_sync", 0.005)
    stage("control_scheduling_wait", 0.005)
    for name, direction, count in [
        ("access_request", "uplink", 64),
        ("access_grant", "downlink", 128),
        ("handover_complete", "uplink", 64),
        ("handover_ack", "downlink", 128),
    ]:
        if direction == "downlink":
            stage("gateway_processing", 0.005)
        xyz, vel = shell.state(time, np.array([satellite]))
        _, ru, _ = geometry(xyz, vel, *observer)
        _, rg, _ = geometry(xyz, vel, *gateway)
        propagation = float((ru[0] + rg[0]) / LIGHT_KM_S)
        capacity = 10e6 if direction == "uplink" else 50e6
        transmission = 8 * count / capacity
        stage(
            name,
            propagation + transmission,
            direction=direction,
            bytes=count,
            propagation_seconds=propagation,
            serialization_seconds=transmission,
        )
    return time - at, stages


def run_seed(seed, seconds, observer, gateway, masks, records):
    shell = Shell(seed)
    xyz, vel = shell.state(100)
    assert np.allclose(np.linalg.norm(xyz, axis=1), shell.radius)
    x1, _ = shell.state(100.001)
    assert np.allclose((x1 - xyz) / 0.001, vel, atol=1e-4)
    data = {}
    for mask in masks:
        data[mask] = {
            "counts": [],
            "pass_lengths": [],
            "pass_start": np.full(1584, -1.0),
            "previous_visible": np.zeros(1584, bool),
            "gap_start": None,
            "gaps": [],
            "policies": {},
        }
        for policy in ("retain_longest_visible", "strongest_review_15s"):
            data[mask]["policies"][policy] = {
                "satellite": -1,
                "switches": 0,
                "ranges": [],
                "space_roundtrip": [],
                "ul_doppler": [],
                "dl_doppler": [],
                "ul_doppler_rate": [],
                "last_ul_doppler": None,
                "handshakes": [],
            }
    for tick in range(seconds):
        xyz, vel = shell.state(tick)
        el, ranges, radial = geometry(xyz, vel, *observer)
        gel, gateway_ranges, _ = geometry(xyz, vel, *gateway)
        for mask, d in data.items():
            visible = (el >= mask) & (gel >= 10)
            d["counts"].append(int(visible.sum()))
            entering = visible & ~d["previous_visible"]
            leaving = ~visible & d["previous_visible"]
            valid = leaving & (d["pass_start"] > 0)
            d["pass_lengths"].extend((tick - d["pass_start"][valid]).tolist())
            d["pass_start"][entering] = tick
            d["previous_visible"] = visible
            if not visible.any():
                if d["gap_start"] is None:
                    d["gap_start"] = tick
            elif d["gap_start"] is not None:
                d["gaps"].append(tick - d["gap_start"])
                d["gap_start"] = None
            for policy, p in d["policies"].items():
                previous = p["satellite"]
                lost = previous < 0 or not visible[previous]
                review = policy == "strongest_review_15s" and tick % 15 == 0
                if lost or review:
                    candidates = np.flatnonzero(visible)
                    if not candidates.size:
                        chosen = -1
                    elif policy == "retain_longest_visible":
                        chosen = longest_contact(
                            shell, tick, candidates, observer, gateway, mask
                        )
                    else:
                        chosen = int(candidates[np.argmax(el[candidates])])
                    if chosen != previous:
                        if previous >= 0 and chosen >= 0:
                            duration, stages = handshake(
                                shell, tick, chosen, observer, gateway
                            )
                            p["switches"] += 1
                            p["handshakes"].append(duration)
                            records.write(
                                json.dumps(
                                    {
                                        "seed": seed,
                                        "time": tick,
                                        "policy": policy,
                                        "mask_degrees": mask,
                                        "previous_satellite": previous,
                                        "target_satellite": chosen,
                                        "elevation_degrees": float(el[chosen]),
                                        "gateway_elevation_degrees": float(gel[chosen]),
                                        "duration_seconds": duration,
                                        "stages": stages,
                                    }
                                )
                                + "\n"
                            )
                        p["last_ul_doppler"] = None
                    p["satellite"] = chosen
                chosen = p["satellite"]
                if chosen < 0:
                    continue
                ul = -UL_HZ * radial[chosen] / LIGHT_KM_S
                dl = -DL_HZ * radial[chosen] / LIGHT_KM_S
                p["ranges"].append(float(ranges[chosen]))
                p["space_roundtrip"].append(
                    float(2 * (ranges[chosen] + gateway_ranges[chosen]) / LIGHT_KM_S)
                )
                p["ul_doppler"].append(float(ul))
                p["dl_doppler"].append(float(dl))
                if p["last_ul_doppler"] is not None:
                    p["ul_doppler_rate"].append(abs(float(ul) - p["last_ul_doppler"]))
                p["last_ul_doppler"] = float(ul)
    for d in data.values():
        if d["gap_start"] is not None:
            d["gaps"].append(seconds - d["gap_start"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/orbit-feasibility-tremblant-01")
    )
    parser.add_argument("--hours-per-seed", type=float, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observer = site(46.2, -74.6, 0.4)
    gateway = site(
        45.3, -74.6, 0.3
    )  # Illustrative relay, not an actual provider gateway.
    masks = [25.0, 40.0]
    aggregate = defaultdict(lambda: defaultdict(list))
    policy_data = defaultdict(lambda: defaultdict(list))
    seconds = int(args.hours_per_seed * 3600)
    with (args.output / "handover_events.jsonl").open("w", encoding="utf-8") as records:
        for seed in args.seeds:
            result = run_seed(seed, seconds, observer, gateway, masks, records)
            for mask, d in result.items():
                for name in ("counts", "pass_lengths", "gaps"):
                    aggregate[mask][name].extend(d[name])
                for name, p in d["policies"].items():
                    target = policy_data[(mask, name)]
                    for key in (
                        "ranges",
                        "space_roundtrip",
                        "ul_doppler",
                        "dl_doppler",
                        "ul_doppler_rate",
                        "handshakes",
                    ):
                        target[key].extend(p[key])
                    target["switches"].append(p["switches"])
            print(
                json.dumps(
                    {"completed_seed": seed, "simulated_hours": args.hours_per_seed}
                ),
                flush=True,
            )
    total = seconds * len(args.seeds)
    rows = []
    for (mask, policy), d in policy_data.items():
        a = aggregate[mask]
        switches = sum(d["switches"])
        rows.append(
            {
                "elevation_mask_degrees": mask,
                "selection_policy": policy,
                "mean_visible_satellites": float(np.mean(a["counts"])),
                "min_visible_satellites": int(min(a["counts"])),
                "visibility_gap_seconds": sum(a["gaps"]),
                "visibility_gap_fraction": float(np.mean(np.asarray(a["counts"]) == 0)),
                "complete_visibility_pass_seconds": describe(a["pass_lengths"]),
                "satellite_changes": switches,
                "handover_rate_per_hour": switches / total * 3600,
                "mean_interval_seconds": total / switches if switches else None,
                "expected_changes_per_250s": switches / total * 250,
                "warm_handshake_seconds": describe(d["handshakes"]),
                "space_roundtrip_seconds": describe(d["space_roundtrip"]),
                "raw_uplink_doppler_hz": describe(d["ul_doppler"]),
                "raw_downlink_doppler_hz": describe(d["dl_doppler"]),
                "uplink_absolute_doppler_rate_hz_per_second": describe(
                    d["ul_doppler_rate"]
                ),
                "interruption_time_fraction": sum(d["handshakes"]) / total,
            }
        )
    radius = EARTH_KM + 550
    rate = math.sqrt(MU / radius**3)
    minimum_el = math.radians(25)
    psi = math.acos(EARTH_KM / radius * math.cos(minimum_el)) - minimum_el
    edge_range = math.sqrt(
        radius**2 - EARTH_KM**2 * math.cos(minimum_el) ** 2
    ) - EARTH_KM * math.sin(minimum_el)
    traffic = []
    for agents in (2, 4, 6, 8):
        transmission = MODEL_BYTES * 8 / 50e6
        traffic.append(
            {
                "agents": agents,
                "experience_upload_mbps": EXPERIENCE_BYTES * 8 * agents / 1e6,
                "per_agent_unicast_model_downlink_mbps": MODEL_BYTES
                * 8
                * agents
                / 16
                / 1e6,
                "one_copy_to_UAV_model_downlink_mbps": MODEL_BYTES * 8 / 16 / 1e6,
                "50Mbps_store_forward_deployment_min_seconds": 2 * transmission,
                "50Mbps_store_forward_deployment_mean_seconds": (agents + 3)
                / 2
                * transmission,
                "50Mbps_store_forward_deployment_max_seconds": (agents + 1)
                * transmission,
            }
        )
    output = {
        "result_type": "modeled geometry and protocol feasibility, not measured network or policy results",
        "site": {"latitude": 46.2, "longitude": -74.6, "altitude_km": 0.4},
        "gateway": {
            "latitude": 45.3,
            "longitude": -74.6,
            "altitude_km": 0.3,
            "description": "illustrative control gateway, not actual Starlink infrastructure",
        },
        "constellation": {
            "altitude_km": 550,
            "inclination_degrees": 53,
            "planes": 72,
            "satellites_per_plane": 22,
            "walker_phasing_F": 1,
            "provenance": "historical shell dimensions from FCC 21-48; phasing/epochs are simulation assumptions",
            "not_current_operational_ephemerides": True,
        },
        "sampling": {
            "seeds": args.seeds,
            "hours_per_seed": args.hours_per_seed,
            "time_step_seconds": 1,
            "initial_attach_excluded": True,
            "visibility_passes_boundary_censored": True,
        },
        "assumptions": {
            "spherical_earth_km": EARTH_KM,
            "circular_two_body_orbits": True,
            "earth_rotation_rad_per_second": EARTH_RATE,
            "gateway_elevation_mask_degrees": 10,
            "uplink_carrier_hz": UL_HZ,
            "downlink_carrier_hz": DL_HZ,
            "doppler_compensation": "ideal in baseline; raw shift does not imply outage",
            "sync_seconds": 0.005,
            "scheduling_wait_seconds": 0.005,
            "gateway_processing_seconds_per_response": 0.005,
            "control_packet_sizes_bytes": [64, 128, 64, 128],
            "control_protocol": "assumed break-before-make request/grant/complete/ack; not Starlink protocol",
            "control_capacity_mbps": {"uplink": 10, "downlink": 50},
            "handshake_data_queueing": "excluded from free-link protocol estimate",
            "terrain": "none; 40-degree mask is a sensitivity assumption, not measured local terrain",
            "J2_drag_fading_TCP_retries_ISL": "not simulated",
        },
        "analytic": {
            "orbital_period_minutes": 2 * math.pi / rate / 60,
            "orbital_speed_km_per_second": radius * rate,
            "footprint_half_angle_degrees": math.degrees(psi),
            "overhead_pass_no_earth_rotation_seconds": 2 * psi / rate,
            "25_degree_edge_slant_range_km": edge_range,
            "minimum_one_space_leg_seconds": 550 / LIGHT_KM_S,
            "maximum_one_space_leg_seconds": edge_range / LIGHT_KM_S,
        },
        "orbit_results": rows,
        "traffic_calculations": traffic,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
