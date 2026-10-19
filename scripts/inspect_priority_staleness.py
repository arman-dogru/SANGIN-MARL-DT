"""Replay recorded actions to measure priority staleness's observation exposure.

No neural policy is rerun, no outcome record is edited, and fresh-priority
counterfactual observations are discarded before the next simulator step.
"""
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from run_baseline_inference import reference_config


def main():
    root = Path("output/learning-pilots/fast-curriculum-20261005-01").resolve()
    inference = Path("output/inference/baseline-20261006-01")
    sys.path.insert(0, str(root/"code"))
    from panama_uav.config import ExperimentMode, FIXED_MAP
    from panama_uav.environment import ExperimentEnvironment
    config, _ = reference_config(root)
    config = replace(config, communication=replace(
        config.communication, profile="controlled", mission_route="uav",
        stale_information=("priorities",), delay_steps=20,
        handover_enabled=False, contention_enabled=True))
    directory = inference/"planned/controlled_uav_priorities_20"
    results = []
    for count in (2, 4, 6, 8):
        totals = {"agents": count, "cases": 0, "agent_decision_steps": 0,
                  "agent_steps_with_aged_priorities": 0,
                  "agent_steps_with_changed_intention_channel": 0,
                  "cases_with_changed_intention_channel": 0,
                  "cases_with_realized_priority_age": 0,
                  "recorded_positions_reproduced": True}
        for path in sorted(directory.glob(f"agents-{count}-case-*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            case = trace["case"]
            env = ExperimentEnvironment(config, ExperimentMode.UAV, FIXED_MAP, count, 250, case["seed"])
            observations, _ = env.reset()
            changed = aged = False
            for frame in trace["frames"][1:]:
                totals["agent_decision_steps"] += count
                for agent in env.agents:
                    ages = env.communication.ages()[agent.id]
                    if ages["priorities"] > 0:
                        aged = True
                        totals["agent_steps_with_aged_priorities"] += 1
                    cache = env.communication.caches[agent.id]["priorities"]
                    original = cache.value
                    cache.value = {peer.id: peer.priority for peer in env.agents}
                    fresh = env.observation(agent)
                    cache.value = original
                    assert np.array_equal(fresh["vector"], observations[agent.id]["vector"])
                    if not np.array_equal(fresh["spatial"][2], observations[agent.id]["spatial"][2]):
                        changed = True
                        totals["agent_steps_with_changed_intention_channel"] += 1
                result = env.step({int(a): action for a, action in frame["actions"].items()})
                assert [list(a.position) for a in env.agents] == frame["positions"]
                observations = result.observations
            totals["cases"] += 1
            totals["cases_with_changed_intention_channel"] += int(changed)
            totals["cases_with_realized_priority_age"] += int(aged)
        assert totals["cases"] == 50
        totals["changed_intention_channel_fraction"] = totals["agent_steps_with_changed_intention_channel"]/totals["agent_decision_steps"]
        results.append(totals)
        print(json.dumps(totals), flush=True)
    (inference/"priority_observation_exposure.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
