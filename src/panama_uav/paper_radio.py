"""Explicit RF assumptions and full-route propagation for paper-mode telemetry."""
import math

import numpy as np

from .leo import C_KM_S


def access_budget(path_loss_db, parameters, ceiling_bps):
    bandwidth = parameters.access_bandwidth_hz
    noise = parameters.noise_psd_w_hz*bandwidth*10**(parameters.access_noise_figure_db/10)
    snr = parameters.robot_power_w*10**(-path_loss_db/10)/noise
    bound = bandwidth*math.log1p(snr)/math.log(2)
    return dict(path_loss_db=float(path_loss_db), bandwidth_hz=bandwidth,
        noise_psd_w_hz=parameters.noise_psd_w_hz, noise_figure_db=parameters.access_noise_figure_db,
        transmit_power_w=parameters.robot_power_w, snr_db=10*math.log10(snr),
        rate_bound_bps=bound, configured_cap_bps=ceiling_bps, applied_bps=min(bound, ceiling_bps))


def backhaul_budget(transport, parameters):
    config = transport.config
    result = dict(available=True, applied_bps=config.satellite_uplink_mbps*1e6,
        configured_cap_bps=config.satellite_uplink_mbps*1e6,
        downlink_applied_bps=config.satellite_downlink_mbps*1e6,
        one_way_seconds=config.one_way_propagation_seconds,
        air_to_space_one_way_seconds=None, satellite_gateway_one_way_seconds=None,
        ground_one_way_seconds=None, distance_km=None, elevation_degrees=None,
        fspl_db=None, atmosphere_db=None, snr_db=None, rate_bound_bps=None,
        bandwidth_hz=20e6, transmit_power_w=parameters.drone_power_w,
        terminal_eirp_dbw=parameters.terminal_eirp_dbw,
        implied_antenna_gain_dbi=parameters.terminal_eirp_dbw-10*math.log10(parameters.drone_power_w),
        provenance="configured full-route one-way delay and allocated rate caps")
    orbit = transport.orbit
    if orbit is None:
        fspl = 20*math.log10(4*math.pi*parameters.assumed_satellite_distance_km*1000
                            *parameters.a2s_frequency_hz/299792458)
        snr_db = (parameters.terminal_eirp_dbw+parameters.satellite_receive_gt_db_k-fspl
            -parameters.assumed_atmosphere_db-parameters.implementation_loss_db
            -10*math.log10(1.380649e-23)-10*math.log10(parameters.a2s_bandwidth_hz))
        bound = parameters.shannon_efficiency*parameters.a2s_bandwidth_hz*math.log2(1+10**(snr_db/10))
        result.update(distance_km=parameters.assumed_satellite_distance_km, fspl_db=fspl,
            atmosphere_db=parameters.assumed_atmosphere_db, snr_db=snr_db,
            bandwidth_hz=parameters.a2s_bandwidth_hz, rate_bound_bps=bound,
            applied_bps=min(result["applied_bps"], bound),
            air_to_space_one_way_seconds=parameters.assumed_satellite_distance_km/C_KM_S,
            satellite_gateway_one_way_seconds=parameters.assumed_gateway_distance_km/C_KM_S,
            provenance="assumed fixed geometry and attenuation; full-route delay configured separately")
        space = result["air_to_space_one_way_seconds"]+result["satellite_gateway_one_way_seconds"]
        if result["one_way_seconds"] < space:
            raise ValueError("Full-route propagation is shorter than its assumed space legs")
        result["ground_one_way_seconds"] = result["one_way_seconds"]-space
        result["round_trip_seconds"] = 2*result["one_way_seconds"]
        return result
    local = orbit.local_time(transport.now+config.backhaul_time_offset_seconds, config.orbital_replay)
    i = max(0, int(np.searchsorted(orbit.times, local, side="right"))-1)
    def value(key):
        series = orbit.series.get(key)
        if series is None:
            return None
        v = float(series[i])
        return v if math.isfinite(v) else None
    one_way = float(orbit.propagation[i])
    up, down = value("satellite_up_bps"), value("satellite_down_bps")
    blocked = any(left <= local < right and (cause != 1 or config.handover_enabled)
                  for left, right, cause in orbit.blocked)
    available = math.isfinite(one_way) and one_way >= 0 and up is not None and up > 0 and not blocked
    result.update(available=available, applied_bps=min(result["applied_bps"], up) if available else 0.0,
        downlink_applied_bps=min(result["downlink_applied_bps"], down) if down is not None else 0.0,
        one_way_seconds=one_way if math.isfinite(one_way) else 0.0,
        distance_km=value("range_km"), elevation_degrees=value("elevation_degrees"),
        fspl_db=value("uav_up_fspl_db"), atmosphere_db=value("uav_up_atmosphere_db"),
        snr_db=value("uav_up_cn_db"), rate_bound_bps=value("uav_up_rate_bound_bps"),
        provenance="archived ITU-R/geometry budget; service uses existing orbital timeline")
    if result["distance_km"] is not None:
        result["air_to_space_one_way_seconds"] = result["distance_km"]/C_KM_S
    gateway = value("gateway_range_km")
    if gateway is not None:
        result["satellite_gateway_one_way_seconds"] = gateway/C_KM_S
    if result["air_to_space_one_way_seconds"] is not None and gateway is not None:
        result["ground_one_way_seconds"] = max(0.0, result["one_way_seconds"]
            -result["air_to_space_one_way_seconds"]-gateway/C_KM_S)
    # Archived rates cannot be reused under a changed EIRP assumption.
    radio = orbit.metadata.get("radio", orbit.metadata.get("radio_model", {}))
    archived = radio.get("parameters", {}).get("terminal_eirp_dbw", 35.0)
    if abs(archived-parameters.terminal_eirp_dbw) > 1e-6:
        raise ValueError("Rebuild the orbital rate timeline when changing terminal EIRP")
    result["round_trip_seconds"] = 2*result["one_way_seconds"]
    return result
