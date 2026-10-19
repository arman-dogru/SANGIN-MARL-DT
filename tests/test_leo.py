import copy
import json
from itertools import pairwise

import numpy as np
import pytest

from panama_uav.config import TransportConfig
from panama_uav.leo import (
    EARTH_RATE,
    OrbitTimeline,
    handshake,
    link_geometry,
    observer_ecef,
    teme_to_ecef,
)
from panama_uav.transport import DelayTrace, Transport, import_trace


def timeline_file(tmp_path):
    path = tmp_path / "orbit.npz"
    event = {"start": 5.0, "end": 5.1, "to_satellite": 12}
    np.savez(
        path,
        times=np.arange(21.0),
        propagation_seconds=np.full(21, 0.002),
        blocked_windows=np.array([[5, 5.1, 1], [10, 12, 0]]),
        handover_events=json.dumps([event]),
        metadata=json.dumps({"test": True}),
    )
    return str(path)


def test_teme_rotation_range_and_velocity():
    position = np.array([[7000.0, 0, 0]])
    velocity = np.array([[0.0, 7.5, 0]])
    r, v = teme_to_ecef(position, velocity, np.array([2451545.0]))
    assert np.linalg.norm(r[0]) == pytest.approx(7000)
    assert np.linalg.norm(v[0]) == pytest.approx(7.5 - EARTH_RATE * 7000)
    observer, up = observer_ecef(46.2, -74.6, 0.42)
    elevation, distance, rate = link_geometry(observer + up * 550, up * 2, observer, up)
    assert elevation == pytest.approx(90)
    assert distance == pytest.approx(550)
    assert rate == pytest.approx(2)


def test_handshake_has_individual_packets_and_capacity():
    finish, events = handshake(4, 0.004)
    packets = [e for e in events if "bytes" in e]
    assert [e["direction"] for e in packets] == [
        "uplink",
        "downlink",
        "uplink",
        "downlink",
    ]
    assert sum(e["bytes"] for e in packets) == 384
    assert finish - 4 == pytest.approx(
        0.005 + 4 * 0.004 + 0.010 + 128 * 8 / 10e6 + 256 * 8 / 50e6
    )
    assert all(a["end"] <= b["start"] for a, b in pairwise(events))


def test_orbit_interruptions_no_visibility_and_no_extrapolation(tmp_path):
    timeline = OrbitTimeline.load(timeline_file(tmp_path))
    assert timeline.finish(4.95, 0.2) == pytest.approx(5.25)
    assert timeline.finish(5.02, 0.2) == pytest.approx(5.3)
    assert timeline.finish(4.95, 0.2, False) == pytest.approx(5.15)
    assert timeline.finish(9.9, 0.3, False) == pytest.approx(12.2)
    with pytest.raises(ValueError):
        timeline.finish(20, 0.1)


def test_orbit_backhaul_buffers_but_local_uav_continues_and_resume(tmp_path):
    config = TransportConfig(
        profile="physical",
        leo_timeline=timeline_file(tmp_path),
        contention_enabled=False,
        backhaul_time_offset_seconds=5,
    )
    transport = Transport(config)
    transport.send(0, "local", {}, route=("access_up", "access_down"))
    transport.send(1, "cloud", {}, route=("satellite_up",))
    assert [m.kind for m in transport.advance(0.05)] == ["local"]
    resumed = copy.deepcopy(transport)
    assert transport.advance(0.2) == resumed.advance(0.2)
    assert transport.log == resumed.log
    tx = next(
        x
        for x in transport.log
        if x["event"] == "transmit" and x["link"] == "satellite_up"
    )
    assert tx["handover_seconds"] == pytest.approx(0.1)
    assert tx["propagation_seconds"] == pytest.approx(0.002)
    assert (
        transport.counters["bytes_delivered"] == transport.counters["bytes_generated"]
    )


def test_actual_dataset_headers_origin_and_trace_replacement(tmp_path):
    source = tmp_path / "ul.csv"
    source.write_text(
        "seq_nr,ul_t0,owd_t0_t1\n1,5000000,30000000\n3,1000000000,60000000\n"
    )
    output = tmp_path / "ul.npz"
    metadata = import_trace(source, output, "uplink", origin_seconds=0)
    assert metadata["rows"] == 2
    assert metadata["missing_sequences_are_loss"] is False
    trace = DelayTrace.load(output)
    assert trace.timestamps[0] == pytest.approx(0.005)
    with pytest.raises(ValueError):
        trace.delay(0)
    transport = Transport(
        TransportConfig(
            profile="physical",
            uplink_trace=str(output),
            contention_enabled=False,
            backhaul_time_offset_seconds=0.1,
            one_way_propagation_seconds=10,
        )
    )
    transport.send(0, "test", {}, route=("satellite_up",))
    assert len(transport.advance(0.031)) == 1
    assert not any(x["event"] == "transmit" for x in transport.log)
    with pytest.raises(ValueError, match="bulk model"):
        transport.send(0, "model", {}, route=("satellite_up",))
    with pytest.raises(ValueError, match="double-count"):
        TransportConfig(leo_timeline="orbit.npz", uplink_trace=str(output))
