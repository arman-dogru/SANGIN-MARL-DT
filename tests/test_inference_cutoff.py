import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.information import MissionCommunication

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inference_runner", ROOT / "scripts/run_baseline_inference.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def environments():
    config = load_config(ROOT / "configs/base.toml")
    fresh = replace(config, communication=replace(config.communication, profile="perfect"))
    cutoff = replace(config, communication=replace(
        config.communication, profile="controlled", delay_steps=0,
        stale_information=("intentions", "priorities"),
        contention_enabled=False, handover_enabled=False))
    envs = [ExperimentEnvironment(c, ExperimentMode.UAV, "maps/random-32-32-10.map", 4, 250, 123)
            for c in (fresh, cutoff)]
    for env in envs:
        env.reset()
    return envs


def test_cutoff_only_ages_peer_inputs_recovers_and_keeps_physics():
    fresh, cutoff = environments()
    original = MissionCommunication._receive
    for agent in range(4):
        cutoff.communication.transport.windows[agent] = [(1, 4)]
    with runner.peer_cutoffs(MissionCommunication):
        for tick in range(1, 6):
            actions = {0: 0, 1: 1, 2: 3, 3: 4}
            a, b = fresh.step(actions), cutoff.step(actions)
            assert a.rewards == b.rewards
            assert a.info["agent_positions"] == b.info["agent_positions"]
            for agent in range(4):
                np.testing.assert_array_equal(a.observations[agent]["vector"], b.observations[agent]["vector"])
                ages = cutoff.communication.ages()[agent]
                assert ages["task"] == 0
                assert ages["intentions"] == ages["priorities"] == (tick if tick < 4 else 0)
    assert MissionCommunication._receive is original
    assert any(e["event"] == "peer_cutoff_drop" for e in cutoff.communication.transport.log)


def test_cutoff_without_exposure_reproduces_fresh_observations():
    fresh, cutoff = environments()
    cutoff.communication.transport.windows = {agent: [] for agent in range(4)}
    with runner.peer_cutoffs(MissionCommunication):
        for _ in range(6):
            actions = {0: 0, 1: 1, 2: 3, 3: 4}
            a, b = fresh.step(actions), cutoff.step(actions)
            for agent in range(4):
                for field in ("spatial", "vector"):
                    np.testing.assert_array_equal(a.observations[agent][field], b.observations[agent][field])
