import json

import numpy as np
import pytest

from panama_uav.paper_audit import association_average
from panama_uav.leo_radio import RadioAssumptions
from dataclasses import asdict


def test_association_average_recomputes_rate_and_retains_interruptions(tmp_path):
    radio = asdict(RadioAssumptions())
    data = dict(times=np.arange(4.), satellite_ids=np.ones(4), range_km=np.array([500.,600.,700.,800.]),
        gateway_range_km=np.array([600.,700.,800.,900.]), elevation_degrees=np.full(4,60.),
        gateway_elevation_degrees=np.full(4,50.), propagation_seconds=np.zeros(4),
        blocked_windows=np.array([[0.,.05,1.]]), handover_events=np.array("[]"),
        metadata=np.array(json.dumps(dict(scheduling_interval_seconds=2, ground_delay_seconds=.01,
                                         radio=dict(parameters=radio)))))
    for name in ("uav_up","uav_down","gateway_up","gateway_down"):
        data[name+"_atmosphere_db"] = np.array([1.,3.,2.,4.])
    source, target = tmp_path/"raw.npz",tmp_path/"average.npz"
    np.savez_compressed(source,**data)
    association_average(source,target)
    with np.load(target) as a:
        np.testing.assert_allclose(a["range_km"],[550,550,750,750])
        np.testing.assert_allclose(a["uav_up_atmosphere_db"],[2,2,3,3])
        np.testing.assert_array_equal(a["blocked_windows"],data["blocked_windows"])
        expected=20*np.log10(4*np.pi*14.25e9*550e3/299792458)
        assert a["uav_up_fspl_db"][0] == pytest.approx(expected)
        assert a["propagation_seconds"][0] == pytest.approx((550+650)/299792.458+.01)
