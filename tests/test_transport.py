import copy
from itertools import pairwise

import numpy as np
import pytest

from panama_uav.config import TransportConfig
from panama_uav.transport import DelayTrace, Transport, import_trace


def test_shared_capacity_round_robin_fifo_and_conservation():
    c = TransportConfig(
        profile="physical",
        access_uplink_mbps=0.008,
        contention_enabled=False,
        handover_enabled=False,
        one_way_propagation_seconds=0,
    )
    t = Transport(c)
    for agent in (0, 0, 0, 1, 1, 2):
        t.send(agent, "data", bytes(1000), route=("access_up",))
    t.advance(100)
    transmissions = [r for r in t.log if r["event"] == "transmit"]
    assert [r["agent"] for r in transmissions] == [0, 1, 2, 0, 1, 0]
    for first, second in pairwise(transmissions):
        assert second["time"] >= first["finish"] - 1e-12
    assert all(
        abs(r["transmission_seconds"] - r["bytes"] / 1000) < 1e-10
        for r in transmissions
    )
    assert t.counters["messages_generated"] == t.counters["messages_delivered"] == 6
    assert t.counters["bytes_generated"] == t.counters["bytes_delivered"]
    assert t.counters["hop_bytes_transmitted"] == t.counters["bytes_generated"]


def test_handover_only_blocks_satellite_and_buffers():
    t = Transport(TransportConfig(profile="physical", contention_enabled=False))
    t.advance(15)
    t.send(0, "edge", {}, route=("access_up", "access_down"))
    t.send(1, "cloud", {}, route=("satellite_up", "satellite_down"))
    messages = t.advance(15.01)
    assert [m.kind for m in messages] == ["edge"]
    assert t.advance(15.2)[0].kind == "cloud"
    transmissions = [
        r for r in t.log if r["event"] == "transmit" and r["link"] == "satellite_up"
    ]
    assert transmissions[0]["handover_seconds"] == pytest.approx(0.05)


def test_pending_message_resume_and_late_old():
    t = Transport(TransportConfig(profile="controlled", delay_steps=10))
    t.windows = {0: [(0, 2)]}
    t.send(0, "old", {"x": 1}, version=1)
    t.advance(3)
    t.send(0, "new", {"x": 2}, version=2)
    assert t.advance(3)[0].version == 2
    checkpoint = copy.deepcopy(t)
    assert t.advance(20) == checkpoint.advance(20)
    assert t.log == checkpoint.log


def test_immutable_encoded_tuple():
    t = Transport(TransportConfig())
    t.windows = {0: [(0, 2)]}
    observation = np.array([1, 2])
    t.send(0, "experience", observation)
    observation[0] = 999
    np.testing.assert_array_equal(t.advance(10)[0].payload, [1, 2])


def test_schedule_independent_of_actions_and_early_completion():
    a = Transport(TransportConfig())
    b = Transport(TransportConfig())
    a.schedule_episode(123, 8)
    b.schedule_episode(123, 8)
    assert a.windows == b.windows
    assert all(1 <= len(w) <= 2 for w in a.windows.values())
    a.advance(0.1)
    assert a.counters["exposed_messages"] == 0
    assert len([r for r in a.log if r["event"] == "scheduled_reservation"]) >= 8


def test_trace_direction_units_segment_and_no_double_count(tmp_path):
    source = tmp_path / "ul.txt"
    source.write_text(
        "1 1000000000 30000000\n3 2000000000 40000000\n4 3000000000 50000000\n"
    )
    path = tmp_path / "ul.npz"
    metadata = import_trace(source, path, "uplink", "ns", start=1, end=3)
    assert metadata["missing_sequences_are_loss"] is False
    assert (
        len(metadata["source_sha256"]) == 64
        and metadata["source_first_timestamp"] == "1000000000"
    )
    trace = DelayTrace.load(path)
    assert trace.delay(0.5) == pytest.approx(0.03)
    with pytest.raises(ValueError):
        trace.delay(3)
    t = Transport(
        TransportConfig(
            profile="physical",
            uplink_trace=str(path),
            contention_enabled=False,
            one_way_propagation_seconds=10,
        )
    )
    t.send(0, "data", bytes(100000), route=("satellite_up",))
    assert len(t.advance(0.031)) == 1
    assert not any(e["event"] == "transmit" for e in t.log)
    with pytest.raises(ValueError):
        Transport(TransportConfig(downlink_trace=str(path)))


def test_invalid_capacities():
    with pytest.raises(ValueError):
        TransportConfig(satellite_uplink_mbps=0)


def test_reservation_blocks_newer_updates_in_same_flow():
    t = Transport(TransportConfig(profile="controlled", delay_steps=20))
    t.windows = {0: [(1, 3)]}
    t.advance(1)
    t.send(0, "information", {"version": 1})
    for tick in range(2, 21):
        assert not t.advance(tick)
        t.send(0, "information", {"version": tick})
    received = t.advance(22)
    assert [p.payload["version"] for p in received] == list(range(1, 21))
    assert received[0].generated_at == 1
    assert t.counters["exposed_messages"] == 2
