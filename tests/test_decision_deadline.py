from dataclasses import replace
import json

import numpy as np
import pytest

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.handover_study import derive_timeline
from panama_uav.leo import OrbitTimeline
from panama_uav.transport import Transport


def environment(delay):
    config = load_config('configs/base.toml')
    config = replace(config, communication=replace(config.communication, profile='controlled',
        stale_information=('task',), edge_processing_seconds=delay,
        contention_enabled=False, handover_enabled=False))
    env = ExperimentEnvironment(config, ExperimentMode.UAV, 'maps/random-32-32-10.map', 4, 250, 123)
    env.reset()
    return env


@pytest.mark.parametrize('delay,miss', [(.099, False), (.1, False), (.101, True)])
def test_arrival_deadline_boundary_and_unchanged_robot_clock(delay, miss):
    env = environment(delay)
    env.prepare_decision(.1)
    start = [a.position for a in env.agents]
    result = env.step({0: 0, 1: 1, 2: 3, 3: 4})
    positions = [a.position for a in env.agents]
    obs = env.prepare_decision(.1)
    assert env.step_count == 1 and env.communication.transport.now == 1.1
    assert positions == [a.position for a in env.agents] and positions != start
    for agent in range(4):
        entry = env.communication.decision['inputs'][agent]
        assert entry['task']['response_missed_deadline'] == miss
        assert entry['task']['packet_source_time'] == (0 if miss else 1)
        assert entry['task']['age_seconds'] == pytest.approx(1.1 if miss else .1)
        assert entry['intentions']['age_seconds'] == entry['priorities']['age_seconds'] == 0
        np.testing.assert_array_equal(result.observations[agent]['spatial'], obs[agent]['spatial'])
    env.step({0: 0, 1: 1, 2: 3, 3: 4})
    env.prepare_decision(.1)
    assert env.step_count == 2 and env.communication.transport.now == 2.1
    assert env.communication.caches[0]['task'].timestamp == (1 if miss else 2)


def test_zero_processing_deadline_preserves_current_state_features_and_rewards():
    left, right = environment(0), environment(0)
    right.prepare_decision(.1)
    for _ in range(10):
        actions = {0: 0, 1: 1, 2: 3, 3: 4}
        a, b = left.step(actions), right.step(actions)
        observations = right.prepare_decision(.1)
        assert a.rewards == b.rewards and a.info['agent_positions'] == b.info['agent_positions']
        for agent in range(4):
            for key in ('spatial', 'vector'):
                np.testing.assert_array_equal(a.observations[agent][key], observations[agent][key])
    with pytest.raises(ValueError, match='control cycle'):
        right.prepare_decision(1.0)


def source_timeline(path):
    events = [dict(start=5., end=5.05, from_satellite=1, to_satellite=2,
                   attachment_changed=True, initial_attachment=False, control_messages=[]),
              dict(start=10., end=10.05, from_satellite=2, to_satellite=2,
                   attachment_changed=False, initial_attachment=False, control_messages=[]),
              dict(start=15., end=15.05, from_satellite=2, to_satellite=3,
                   attachment_changed=True, initial_attachment=False, control_messages=[])]
    np.savez(path, times=np.arange(61.), propagation_seconds=np.full(61,.002),
             satellite_up_bps=np.full(61,1e7), satellite_down_bps=np.full(61,5e7),
             blocked_windows=np.array([[5,5.05,1],[10,10.05,1],[15,15.05,1]]),
             handover_events=json.dumps(events), metadata=json.dumps({}))


def test_handover_off_keeps_regular_reconfiguration_and_geometry(tmp_path):
    source, output = tmp_path/'source.npz', tmp_path/'native.npz'
    source_timeline(source)
    derive_timeline(source, output)
    timeline = OrbitTimeline.load(output)
    with np.load(source) as raw:
        np.testing.assert_array_equal(timeline.series['satellite_up_bps'], raw['satellite_up_bps'])
        np.testing.assert_array_equal(timeline.propagation, raw['propagation_seconds'])
    assert timeline.service(5., 1000, 'satellite_up', 1e7, .02,.05,False,False)[2] == 0
    assert timeline.service(5., 1000, 'satellite_up', 1e7, .02,.05,False,True)[2] == pytest.approx(.05)
    assert timeline.service(10., 1000, 'satellite_up', 1e7, .02,.05,False,False)[2] == pytest.approx(.05)
    base = load_config('configs/base.toml').communication
    transport = Transport(replace(base, profile='physical', leo_timeline=str(output),
        handover_enabled=False, contention_enabled=False))
    transport.advance(11)
    changes = [e for e in transport.log if e['event'] == 'orbit_handover']
    assert len(changes) == 2
    assert changes[0]['source_window']['interruption_applied'] is False
    assert changes[1]['source_window']['interruption_applied'] is True
    assert transport.handover_windows(0, 11) == [[10.,10.05]]


def test_long_handover_stress_unions_outages_and_does_not_block_local_access(tmp_path):
    source, output = tmp_path/'source.npz', tmp_path/'stress.npz'
    source_timeline(source)
    metadata = derive_timeline(source, output, 20)
    timeline = OrbitTimeline.load(output)
    np.testing.assert_array_equal(timeline.blocked[:,:2], [[5.,35.]])
    assert metadata['blocked_seconds_per_period'] == 30
    assert metadata['stress_is_measured'] is False
    base = load_config('configs/base.toml').communication
    config = replace(base, profile='physical', leo_timeline=str(output),
        backhaul_time_offset_seconds=6, contention_enabled=False)
    transport = Transport(config)
    transport.send(0,'local',{},route=('access_up','access_down'))
    transport.send(1,'remote',{},route=('satellite_up',))
    assert [m.kind for m in transport.advance(.1)] == ['local']
    assert [m.kind for m in transport.advance(30)] == ['remote']
    with pytest.raises(ValueError, match='Stress timelines'):
        Transport(replace(config, handover_enabled=False))
