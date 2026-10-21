from dataclasses import replace

import numpy as np
import pytest
import torch

from panama_uav.allocation import AllocationSnapshot, AllocationTask, compare_allocations
from panama_uav.compute import ComputePool
from panama_uav.config import TransportConfig, load_config
from panama_uav.models import DQN
from panama_uav.training import load_model
from panama_uav.transport import Transport
from panama_uav.validation import independent_primal, sanity_scenarios


@pytest.mark.parametrize("name,snapshot,expected", sanity_scenarios())
def test_predictable_routes_and_independent_primal(name, snapshot, expected):
    result = compare_allocations(snapshot)
    independent = independent_primal(snapshot)
    assert [int(r["x"]) for r in result["binary"]["tasks"]] == expected
    assert result["binary"]["objective"] == pytest.approx(independent["objective"], rel=1e-6, abs=1e-8)


def test_dvfs_free_rate_and_observed_energy():
    task = AllocationTask("a", 1e5, 1000, 1, 50e6, 10e6, fixed_x=1)
    snapshot = AllocationSnapshot((task,), drone_cycles_per_second=30e9, drone_kappa=1e-28)
    result = compare_allocations(snapshot)["binary"]["tasks"][0]
    expected = (1/(2e-28))**(1/3)
    assert result["drone_cycles_per_second"] == pytest.approx(expected)
    t = Transport(TransportConfig(profile="perfect"))
    done = []
    pool = ComputePool("drone", 30e9, t, lambda j: done.append(j), kappa=1e-28)
    t.compute_pools["drone"] = pool
    pool.submit("a", task.cycles, 1, None)
    finish = task.cycles/expected
    t.advance(finish/2)
    assert pool.energy() == pytest.approx(1e-28*task.cycles*expected**2/2)
    t.advance(finish+1e-8)
    assert len(done) == 1
    assert done[0].compute_energy_j == pytest.approx(result["drone_compute_energy_j"])
    assert pool.energy() == pytest.approx(done[0].compute_energy_j)


def test_checkpoint_line_endings_only(tmp_path):
    import hashlib
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1]/"configs/base.toml")
    raw = config.resolve("maps/random-32-32-10.map").read_bytes().replace(b"\r\n", b"\n")
    payload = dict(communication_schema=4, policy_family="panama_0.1_task",
        task_reward=config.to_dict()["reward"], map_sha256=hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest(),
        spatial_shape=(3, 15, 15), vector_length=12, model_state_dict=DQN((3, 15, 15), 12).state_dict())
    file = tmp_path/"model.pt"
    torch.save(payload, file)
    _, loaded = load_model(file, config, torch.device("cpu"))
    assert loaded["map_compatibility"]["normalized_lf_sha256"] == hashlib.sha256(raw).hexdigest()
    payload["map_sha256"] = hashlib.sha256(raw.replace(b".", b"@", 1)).hexdigest()
    torch.save(payload, file)
    with pytest.raises(ValueError, match="map mismatch"):
        load_model(file, config, torch.device("cpu"))
