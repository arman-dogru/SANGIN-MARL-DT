import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from panama_uav.config import TransportConfig, load_config
from panama_uav.leo import OrbitTimeline
from panama_uav.transport import Transport


def orbit(tmp_path):
    path = tmp_path/"full.npz"
    event = {"start":0.,"end":.05,"to_satellite":11,
             "control_messages":[{"kind":"request","start":0.,"end":.02}]}
    import json
    np.savez(path,times=np.arange(11.),propagation_seconds=np.full(11,.002),
             satellite_up_bps=np.full(11,8000.),satellite_down_bps=np.full(11,16000.),
             blocked_windows=np.array([[0,.05,1],[5,5.05,1]]),
             handover_events=json.dumps([event,dict(event,start=5.,end=5.05)]),
             metadata=json.dumps({"test":True}))
    return str(path)


def test_capacity_frames_interruptions_and_archived_repeat(tmp_path):
    timeline = OrbitTimeline.load(orbit(tmp_path))
    finish, tx, pause, guard = timeline.service(9.9,2400,"satellite_up",8000,.02,.05,True)
    assert finish > 10.25 and tx == pytest.approx(.3)
    assert pause == pytest.approx(.05)
    assert guard > 0
    with pytest.raises(ValueError):
        timeline.service(9.9,2400,"satellite_up",8000,.02,.05,False)
    assert timeline.state(10.2,True)["replay_cycle"] == 1


def test_satellite_frame_service_advances_at_large_absolute_clocks(tmp_path):
    # The production failure was at 32775.06 s, just after a handover. This
    # subprocess makes recurrence a bounded test failure rather than a hung CI.
    import os
    import subprocess
    import sys
    code = '''
import numpy as np
from panama_uav.leo import OrbitTimeline
t = OrbitTimeline(np.array([0., 1e9]), np.zeros(2), np.empty((0, 3)), [], {})
starts = [32775.0550816, 32775.06, 65536.06, 80001000.06]
rng = np.random.default_rng(42)
for center in (2**15, 2**16, 2**26):
    for n in rng.integers(0, 1000, 200):
        edge = center + n*.02
        starts.extend([np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)])
for start in starts:
    end, tx, pause, guard = t.service(start, 50816, 'satellite_up', 1e7, .02, .05)
    assert end > start
    assert abs(tx - .0050816) < 1e-12
    assert pause == 0 and guard >= 0
    assert abs((end-start)-(tx+guard)) < 5e-8
end, *_ = t.service(32775.0550816, 50816, 'satellite_up', 1e7, .02, .05)
assert abs(end - 32775.0611632) < 1e-9
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=15, env=dict(os.environ, PYTHONPATH="src"))
    assert result.returncode == 0, result.stderr


def test_fractional_bit_remainder_finishes_at_frame_boundary():
    # The inference stall had 3.045e-7 bits left after the frame's data region.
    # Its 3.045e-14 s transmission must consume those bits, even below a clock ULP.
    import subprocess
    import sys
    script = '''
import numpy as np
from panama_uav.leo import OrbitTimeline
t = OrbitTimeline(np.arange(1015.), np.zeros(1015), np.empty((0, 3)), [], {})
end, tx, pause, guard = t.service(1012.0178352, 11648, 'satellite_up', 1e7, .02, .05, True)
assert abs(end - 1012.02) < 1e-10
assert abs(tx*1e7 - 11648) < 1e-8
assert pause == 0 and abs(guard - .001) < 1e-10
# A final transmission shorter than one clock ULP also terminates directly.
end, tx, _, _ = t.service(1012.02, 3.045424818992615e-7, 'satellite_up', 1e7, .02, .05, True)
assert abs(tx*1e7 - 3.045424818992615e-7) < 1e-20
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=10)


def test_replayed_interruption_boundaries_advance_in_absolute_time():
    import os
    import subprocess
    import sys
    code = '''
import numpy as np
from panama_uav.leo import OrbitTimeline
t = OrbitTimeline(np.array([0., 21600.]), np.zeros(2),
                  np.array([[0., .05, 1.], [1485., 1485.05, 1.]]), [], {})
for cycle in (0, 1, 3, 6, 100, 1000, 3703):
    for local in (0., 1485.):
        left = cycle*21600.+local
        right = cycle*21600.+local+.05
        for start in (left, np.nextafter(right, -np.inf), right, np.nextafter(right, np.inf)):
            end, tx, pause, guard = t.service(start, 50832., 'satellite_up', 1e7, .02, .05, True)
            assert end > start and tx > 0 and pause >= 0 and guard >= 0
            assert abs(tx - .0050832) < 1e-12
            assert abs((end-start)-(tx+pause+guard)) < 5e-8
            if start >= right:
                assert pause == 0
            if start == left:
                assert abs(pause - (right-left)) < 1e-8
# Exact second production failure: local subtraction is below the window end.
start = 131085.05
assert t.local_time(start, True) < 1485.05
end, tx, pause, guard = t.service(start, 50832., 'satellite_up', 1e7, .02, .05, True)
assert pause == 0 and abs(end - 131085.0550832) < 1e-9
# Disabling handover leaves radio service available at the same boundaries.
end, tx, pause, guard = t.service(131085., 50832., 'satellite_up', 1e7, .02, .05, True, False)
assert pause == 0
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=15, env=dict(os.environ, PYTHONPATH="src"))
    assert result.returncode == 0, result.stderr


def test_round_robin_chunks_atomic_delivery_conservation_and_resume(tmp_path):
    config = TransportConfig(profile="physical",leo_timeline=orbit(tmp_path),
        orbital_replay=True,contention_enabled=False,packet_payload_bytes=500,
        packet_overhead_bytes=0,access_uplink_mbps=.008,satellite_uplink_mbps=.008)
    t = Transport(config)
    for agent in (0,1):
        t.send(agent,"model",bytes(2000),route=("satellite_up",))
    assert t.advance(.8) == []
    resumed = pickle.loads(pickle.dumps(t))
    assert t.advance(8) == resumed.advance(8)
    assert t.log == resumed.log
    assert t.counters["bytes_generated"] == t.counters["bytes_delivered"]
    assert t.counters["messages_delivered"] == 2
    transmissions = [r for r in t.log if r["event"] == "transmit"]
    assert all(r["chunks"] > 1 for r in transmissions)
    assert abs(transmissions[0]["finish"]-transmissions[1]["finish"]) < .7


def test_simpy_calendars_share_no_state_and_local_uav_survives(tmp_path):
    config = TransportConfig(profile="physical",leo_timeline=orbit(tmp_path),
        orbital_replay=True,contention_enabled=False,packet_payload_bytes=500)
    t = Transport(config)
    t.send(0,"local",{},route=("access_up","access_down"))
    t.send(1,"cloud",{},route=("satellite_up",))
    assert [m.kind for m in t.advance(.01)] == ["local"]
    resumed = copy.deepcopy(t)
    assert t.advance(1) == resumed.advance(1)
    assert any(e["event"] == "handshake_packet" for e in t.log)


def test_long_budget_config_does_not_change_task_or_qualification():
    config = load_config("configs/leo_learning.toml")
    assert config.training.budget_seconds == 43200
    assert [s.num_agents for s in config.curriculum] == [2,4,6,8]
    assert all(s.max_episode_steps == 250 and s.graduation_success_rate == .95 for s in config.curriculum)
    assert config.training.validation_episodes == 200
    assert not config.communication.stale_information


def test_shared_model_crosses_backhaul_once_and_receivers_install_atomically():
    import torch
    from panama_uav.training import TrainingSession
    torch.set_num_threads(1)
    config=load_config("configs/base.toml").profile("smoke")
    config=replace(config,communication=replace(config.communication,profile="physical",
        handover_enabled=False,contention_enabled=False,model_distribution="uav_fanout",
        packet_payload_bytes=262144))
    session=TrainingSession(config,"combined",42,"fanout-test",torch.device("cpu"))
    for _ in range(17): session.step()
    assert session.deployed_versions=={0:0,1:0}
    for _ in range(4): session.step()
    assert session.deployed_versions=={0:1,1:1}
    satellite=[r for r in session.transport.log if r["event"]=="transmit" and r["link"]=="satellite_down"]
    local=[r for r in session.transport.log if r["event"]=="transmit" and r["link"]=="access_down"]
    assert len(satellite)==1 and satellite[0]["kind"]=="model_relay"
    assert len(local)==2 and {r["agent"] for r in local}=={0,1}


def test_full_orbit_fanout_learner_checkpoint_continues_identically(tmp_path):
    import io,json,torch
    from panama_uav.training import TrainingSession
    path=orbit(tmp_path)
    with np.load(path) as data: values={key:data[key] for key in data.files}
    values["satellite_up_bps"]=np.full(11,10e6)
    values["satellite_down_bps"]=np.full(11,50e6)
    np.savez(path,**values)
    config=load_config("configs/base.toml").profile("smoke")
    config=replace(config,communication=replace(config.communication,profile="physical",
        contention_enabled=False,model_distribution="uav_fanout",leo_timeline=path,
        orbital_replay=True,packet_payload_bytes=262144))
    torch.set_num_threads(1)
    session=TrainingSession(config,"combined",42,"full-resume",torch.device("cpu"))
    for _ in range(17):session.step()
    stream=io.BytesIO();torch.save(session.checkpoint(),stream)
    expected=[session.step()[0] for _ in range(8)]
    stream.seek(0)
    resumed=TrainingSession.restore(torch.load(stream,weights_only=False),config,"combined",42,"full-resume",torch.device("cpu"))
    actual=[resumed.step()[0] for _ in range(8)]
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)


def test_library_atmosphere_and_rf_units():
    pytest.importorskip("itur",reason="Install the optional [leo] dependencies for the RF library test")
    from panama_uav.leo_radio import RadioAssumptions,atmospheric_curves,link_budget
    parameters=RadioAssumptions()
    curves=atmospheric_curves(46.2,-74.6,.44,parameters)
    first=link_budget(500.,45.,"up",curves,parameters)
    second=link_budget(1000.,45.,"up",curves,parameters)
    assert first["fspl_db"]==pytest.approx(169.504,abs=.02)
    assert second["fspl_db"]-first["fspl_db"]==pytest.approx(6.020599913)
    assert 0 < first["atmosphere_db"] < 20
    assert 0 < second["rate_bound_bps"] < first["rate_bound_bps"]
