"""Reproducible paper-channel averaging and independent physical checks."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .leo import OrbitTimeline, C_KM_S
from .paper_radio import access_budget
from .records import sha256, write_json
from .uav import AirToGroundModel, UAVController


def association_average(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(data["metadata"]))
    interval = metadata["scheduling_interval_seconds"]
    bins = np.floor(data["times"]/interval).astype(int)
    changed = (np.diff(bins) != 0) | (np.diff(data["satellite_ids"]) != 0)
    starts = np.r_[0, np.flatnonzero(changed)+1]
    ends = np.r_[starts[1:], len(bins)]
    for key in ("range_km", "gateway_range_km", "elevation_degrees", "gateway_elevation_degrees",
                "range_rate_km_s", "gateway_range_rate_km_s", "uplink_doppler_hz", "downlink_doppler_hz"):
        if key in data:
            for left, right in zip(starts, ends):
                data[key][left:right] = np.mean(data[key][left:right])
    radio = metadata["radio"]["parameters"]
    for prefix, distance, direction in (("uav_up", "range_km", "up"), ("uav_down", "range_km", "down"),
                                        ("gateway_up", "gateway_range_km", "up"),
                                        ("gateway_down", "gateway_range_km", "down")):
        attenuation = data[prefix+"_atmosphere_db"]
        for left, right in zip(starts, ends):
            attenuation[left:right] = np.mean(attenuation[left:right])
        up = direction == "up"
        frequency = radio["uplink_ghz" if up else "downlink_ghz"]*1e9
        bandwidth = radio["uplink_bandwidth_hz" if up else "downlink_bandwidth_hz"]
        eirp = radio["terminal_eirp_dbw" if up else "satellite_eirp_dbw"]
        gt = radio["satellite_receive_gt_db_k" if up else "terminal_receive_gt_db_k"]
        loss = 20*np.log10(4*np.pi*frequency*data[distance]*1000/(C_KM_S*1000))
        snr = eirp+gt-loss-attenuation-radio["implementation_loss_db"]-10*np.log10(1.380649e-23*bandwidth)
        data[prefix+"_fspl_db"] = loss
        data[prefix+"_cn_db"] = snr
        data[prefix+"_rate_bound_bps"] = radio["shannon_efficiency"]*bandwidth*np.log2(1+10**(snr/10))
    data["satellite_up_bps"] = np.minimum(data["uav_up_rate_bound_bps"], data["gateway_down_rate_bound_bps"])
    data["satellite_down_bps"] = np.minimum(data["gateway_up_rate_bound_bps"], data["uav_down_rate_bound_bps"])
    data["propagation_seconds"] = (data["range_km"]+data["gateway_range_km"])/C_KM_S+metadata["ground_delay_seconds"]
    metadata["association_average"] = dict(parent_sha256=sha256(source), interval_seconds=interval,
        groups=len(starts), method="mean range and dB attenuation per association/scheduling interval; recompute FSPL and Shannon rates",
        interruptions="original interruption windows retained", fading="no instantaneous small-scale fading",
        positions="raw ECEF retained as provenance; service and reported ranges use association means")
    data["metadata"] = np.array(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    return metadata["association_average"]


def audit_physics(config, original_timeline, averaged_timeline, output):
    model, p = AirToGroundModel(config.uav), config.resource_allocation
    checks = []
    for height in (20, 50, 100):
        for distance in (0, 10, 100, 1000):
            angle = math.degrees(math.atan2(height, distance))
            probability = 1/(1+model.a*math.exp(-model.b*(angle-model.a)))
            loss = 20*math.log10(4*math.pi*config.uav.carrier_frequency_hz/3e8)
            loss += 20*math.log10(math.hypot(height, distance))
            loss += probability*model.eta_los+(1-probability)*model.eta_nlos
            actual = float(model.path_loss_db(height, distance))
            assert abs(actual-loss) < 1e-5
            budget = access_budget(actual, p, config.communication.access_uplink_mbps*1e6)
            noise = p.noise_psd_w_hz*p.access_bandwidth_hz*10**(p.access_noise_figure_db/10)
            expected = p.access_bandwidth_hz*math.log2(1+p.robot_power_w*10**(-loss/10)/noise)
            assert abs(budget["rate_bound_bps"]-expected) < 20
            checks.append(dict(height_m=height, horizontal_m=distance, path_loss_db=actual,
                               rate_bound_bps=budget["rate_bound_bps"], applied_bps=budget["applied_bps"]))
    original, average = OrbitTimeline.load(original_timeline), OrbitTimeline.load(averaged_timeline)
    assert average.metadata["association_average"]["parent_sha256"] == sha256(original_timeline)
    radio = average.metadata["radio"]["parameters"]
    for timeline in (original, average):
        ranges = timeline.series["range_km"]
        expected = 20*np.log10(4*np.pi*radio["uplink_ghz"]*1e9*ranges*1000/(C_KM_S*1000))
        np.testing.assert_allclose(timeline.series["uav_up_fspl_db"], expected, atol=1e-10)
        expected = (ranges+timeline.series["gateway_range_km"])/C_KM_S+timeline.metadata["ground_delay_seconds"]
        np.testing.assert_allclose(timeline.propagation, expected, atol=1e-12)
        np.testing.assert_allclose(timeline.series["satellite_up_bps"], np.minimum(
            timeline.series["uav_up_rate_bound_bps"], timeline.series["gateway_down_rate_bound_bps"]), atol=1e-6)
        snr = radio["terminal_eirp_dbw"]+radio["satellite_receive_gt_db_k"]-timeline.series["uav_up_fspl_db"]
        snr -= timeline.series["uav_up_atmosphere_db"]+radio["implementation_loss_db"]
        snr -= 10*np.log10(1.380649e-23*radio["uplink_bandwidth_hz"])
        np.testing.assert_allclose(timeline.series["uav_up_cn_db"], snr, atol=1e-10)
        rate = radio["shannon_efficiency"]*radio["uplink_bandwidth_hz"]*np.log2(1+10**(snr/10))
        np.testing.assert_allclose(timeline.series["uav_up_rate_bound_bps"], rate, rtol=1e-12)
    np.testing.assert_array_equal(original.blocked, average.blocked)
    controller = UAVController((32, 32), config.map.meters_per_cell, config.uav)
    initial = controller.position.copy()
    diagnostic = controller.update(np.array([[1., 1.], [20., 20.]]))
    assert np.linalg.norm(controller.position-initial) <= config.uav.speed_m_per_step+1e-12
    displacement_bound = math.hypot(32*math.sqrt(2), model.optimal_height_m)
    report = dict(status="passed", access_checks=checks, uav=asdict(config.uav),
        uav_optimal_reference_height_m=model.optimal_height_m, uav_coverage_radius_m=model.maximum_radius_m,
        uav_movement_m_per_step=diagnostic.distance_moved,
        effective_noise_psd_dbm_hz=10*math.log10(p.noise_psd_w_hz)+30+p.access_noise_figure_db,
        conducted_drone_power_w=p.drone_power_w, eirp_dbw=p.terminal_eirp_dbw,
        implied_antenna_gain_dbi=p.terminal_eirp_dbw-10*math.log10(p.drone_power_w),
        space_speed_km_s=C_KM_S, access_speed_approx_m_s=3e8,
        original_timeline_sha256=sha256(original_timeline), averaged_timeline_sha256=sha256(averaged_timeline),
        orbital_metadata=average.metadata, association_average=average.metadata["association_average"],
        ranges_km=dict(minimum=float(average.series["range_km"].min()), maximum=float(average.series["range_km"].max())),
        round_trip_propagation_seconds=dict(minimum=float(2*average.propagation.min()), maximum=float(2*average.propagation.max())),
        far_field_uav_displacement_bound_m=displacement_bound,
        far_field_range_relative_error_bound=displacement_bound/(1000*average.series["range_km"].min()),
        far_field_round_trip_error_bound_seconds=2*displacement_bound/(C_KM_S*1000),
        interpretation="Association-mean geometry and ITU-R long-term attenuation quantile; no sampled fading; rate held fixed in each solve",
        delay_accounting="one full-route outbound plus one full-route return; actual response serialization retained")
    write_json(output, report)
    return report
