"""ITU-Rpy atmospheric predictions and an explicit engineering rate bound.

No proprietary Starlink modulation, beam scheduler or measured UAV RF budget
is claimed. Statistical attenuation quantiles are not a generated weather trace.
"""

from dataclasses import asdict, dataclass

import numpy as np
from astropy.constants import c, k_B
import itur


@dataclass(frozen=True)
class RadioAssumptions:
    uplink_ghz: float = 14.25
    downlink_ghz: float = 11.7
    uplink_bandwidth_hz: float = 20e6
    downlink_bandwidth_hz: float = 100e6
    terminal_eirp_dbw: float = 35.0
    satellite_receive_gt_db_k: float = -5.0
    satellite_eirp_dbw: float = 30.0
    terminal_receive_gt_db_k: float = 5.0
    antenna_diameter_m: float = 0.3
    atmospheric_exceedance_percent: float = 1.0
    implementation_loss_db: float = 3.0
    shannon_efficiency: float = 0.5


def atmospheric_curves(latitude, longitude, altitude_km, parameters):
    angles = np.linspace(10.0, 90.0, 161)
    result = {"angles": angles}
    for name, frequency in (("up", parameters.uplink_ghz), ("down", parameters.downlink_ghz)):
        contributions = itur.atmospheric_attenuation_slant_path(
            latitude, longitude, frequency, angles, parameters.atmospheric_exceedance_percent,
            parameters.antenna_diameter_m, hs=altitude_km, return_contributions=True,
        )
        result[name] = [np.asarray(value.value, dtype=float) for value in contributions]
    return result


def link_budget(range_km, elevation, direction, curves, parameters):
    is_up = direction == "up"
    f = parameters.uplink_ghz if is_up else parameters.downlink_ghz
    bandwidth = parameters.uplink_bandwidth_hz if is_up else parameters.downlink_bandwidth_hz
    eirp = parameters.terminal_eirp_dbw if is_up else parameters.satellite_eirp_dbw
    gt = parameters.satellite_receive_gt_db_k if is_up else parameters.terminal_receive_gt_db_k
    free_space = 20*np.log10(4*np.pi*np.asarray(range_km)*1e3*f*1e9/c.value)
    attenuation = np.interp(elevation, curves["angles"], curves[direction][-1])
    cn_db = eirp + gt - free_space - attenuation - parameters.implementation_loss_db \
            - 10*np.log10(k_B.value) - 10*np.log10(bandwidth)
    rate_bound = parameters.shannon_efficiency*bandwidth*np.log2(1+10**(cn_db/10))
    return {"fspl_db": free_space, "atmosphere_db": attenuation,
            "cn_db": cn_db, "rate_bound_bps": rate_bound}


def provenance(parameters):
    return {"library": "ITU-Rpy", "version": itur.__version__,
            "parameters": asdict(parameters),
            "attenuation": "ITU-R P.618 combined rain/cloud/scintillation/gas statistics and bundled climatology",
            "weather": "fixed long-term attenuation exceedance quantile; not instantaneous weather",
            "rate": "efficiency-scaled Shannon upper bound, capped by allocated Mbps; not provider MCS",
            "rf_parameters": "configurable simulation assumptions, not measured Starlink or UAV equipment"}
