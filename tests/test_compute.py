import pytest

from panama_uav.compute import ComputePool
from panama_uav.config import TransportConfig
from panama_uav.transport import Transport


def test_overlapping_work_and_stale_completion_tokens():
    t = Transport(TransportConfig(profile="perfect"))
    complete = []
    p = ComputePool("edge", 100, t, lambda j: complete.append((j.job_id, t.now)))
    t.compute_pools["edge"] = p
    p.submit("a", 100, 1, None)
    t.advance(.5)
    assert p.snapshot()[0]["remaining_cycles"] == pytest.approx(50)
    p.submit("b", 50, 1, None)
    assert sum(p.rates.values()) == pytest.approx(100)
    t.advance(1)
    assert not complete  # original a-at-1 event was cancelled by its token
    t.advance(1.5)
    assert complete == [("a", 1.5), ("b", 1.5)]
    assert not p.jobs


def test_zero_cycle_job_completes_without_consuming_capacity():
    t = Transport(TransportConfig(profile="perfect"))
    complete = []
    p = ComputePool("edge", 100, t, lambda j: complete.append(j.job_id))
    t.compute_pools["edge"] = p
    p.submit("empty", 0, 1, None)
    t.advance(0)
    assert complete == ["empty"]
