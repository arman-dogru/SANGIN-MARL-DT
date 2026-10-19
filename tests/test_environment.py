import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.information import Guidance, InformationCache

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/base.toml")


def environment(c=None, agents=4, seed=123):
    return ExperimentEnvironment(
        c or config(), ExperimentMode.UAV, "maps/random-32-32-10.map", agents, 250, seed
    )


@pytest.mark.parametrize("route", [None, "uav", "cloud"])
def test_zero_delay_exactly_matches_retained_01(route):
    c = config()
    if route:
        c = replace(
            c,
            communication=replace(
                c.communication,
                mission_route=route,
                stale_information=("task", "intentions", "priorities"),
                delay_steps=0,
            ),
        )
    env = environment(c)
    env.reset()
    golden = json.loads((ROOT / "tests/fixtures/zero_delay_01.json").read_text())
    for frame in golden["frames"]:
        result = env.step({int(k): v for k, v in frame["actions"].items()})
        assert result.info["agent_positions"] == frame["positions"]
        assert result.rewards == {int(k): v for k, v in frame["rewards"].items()}
        assert result.terminated == frame["terminated"]
        for agent, observation in result.observations.items():
            np.testing.assert_array_equal(
                observation["spatial"], frame["spatial"][str(agent)]
            )
            np.testing.assert_array_equal(
                observation["vector"], frame["vector"][str(agent)]
            )
        np.testing.assert_array_equal(env.uav.position, frame["uav"])


def test_guidance_absolute_consumption_without_replanning():
    guidance = Guidance([(1, 1), (1, 2), (1, 3), (2, 3)], 3)
    assert guidance.consume((1, 1)) == [(1, 1), (1, 2), (1, 3), (2, 3)]
    assert guidance.consume((1, 2)) == [(1, 2), (1, 3), (2, 3)]
    assert guidance.consume((2, 2)) == [(2, 2), (1, 3), (2, 3)]
    assert guidance.reported_distance == 3
    assert guidance.consume((2, 3)) == [(2, 3)]


def test_cache_cannot_reverse():
    cache = InformationCache()
    assert cache.accept(2, "fresh", 2)
    assert not cache.accept(1, "late", 3)
    assert not cache.accept(2, "old", 1)
    assert cache.value == "fresh"


@pytest.mark.parametrize("field", ["task", "intentions", "priorities"])
def test_selected_input_isolation_and_truth(field):
    base = config()
    delayed = replace(
        base,
        communication=replace(
            base.communication, stale_information=(field,), delay_steps=10
        ),
    )
    fresh = environment(base)
    stale = environment(delayed)
    fresh.reset()
    stale.reset()
    for agent in range(4):
        stale.communication.transport.windows[agent] = [(0, 100)]
    for tick in range(8):
        actions = {0: 0, 1: 1, 2: 3, 3: 4}
        a = fresh.step(actions)
        b = stale.step(actions)
        assert a.rewards == b.rewards
        assert a.info["agent_positions"] == b.info["agent_positions"]
        for agent in range(4):
            for key in ("task", "intentions", "priorities"):
                if key == field:
                    continue
                left = fresh.communication.caches[agent][key].value
                right = stale.communication.caches[agent][key].value
                if key == "task":
                    assert (
                        left.route == right.route
                        and left.reported_distance == right.reported_distance
                    )
                else:
                    assert left == right
        if field != "task":
            for agent in range(4):
                np.testing.assert_array_equal(
                    a.observations[agent]["vector"], b.observations[agent]["vector"]
                )
        if field == "task":
            for agent in range(4):
                np.testing.assert_array_equal(
                    a.observations[agent]["spatial"], b.observations[agent]["spatial"]
                )
    assert stale.communication.ages()[0][field] == 8


def test_goal_occupancy_and_cap():
    env = environment()
    env.reset()
    for agent in env.agents:
        agent.ever_reached_goal = True
    assert not env.all_agents_at_goals
    for agent in env.agents:
        agent.position = agent.goal
    assert env.all_agents_at_goals
    env.reset()
    for _ in range(250):
        result = env.step({a: 4 for a in range(4)})
    assert result.truncated and env.step_count == 250
    with pytest.raises(ValueError):
        ExperimentEnvironment(
            config(), ExperimentMode.UAV, "maps/random-32-32-10.map", 2, 251, 42
        )


def test_uav_uses_received_position_reports():
    c = config()
    c = replace(
        c,
        communication=replace(
            c.communication, profile="physical", stale_information=("task",)
        ),
    )
    env = environment(c)
    env.reset()
    original = [a.position for a in env.agents]
    env.step({0: 0, 1: 1, 2: 3, 3: 4})
    assert [env.communication.reports[a]["position"] for a in range(4)] == original
    assert [a.position for a in env.agents] != original
