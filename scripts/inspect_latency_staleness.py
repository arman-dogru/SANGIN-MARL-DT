"""Reproduce a recorded latency failure and inspect the cached policy inputs.

Uses recorded actions, not a new policy run. Saves an independent diagnostic;
never modifies experiment records, timing, guidance, or learned behavior.
"""
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from run_baseline_inference import reference_config


def main():
    training = Path("output/learning-pilots/fast-curriculum-20261005-01").resolve()
    root = Path("output/inference/baseline-20261006-01")
    sys.path.insert(0, str(training/"code"))
    from panama_uav.config import ExperimentMode, FIXED_MAP
    from panama_uav.environment import ExperimentEnvironment
    config, _ = reference_config(training)
    config = replace(config, communication=replace(
        config.communication, profile="physical", mission_route="uav",
        stale_information=("task", "intentions", "priorities"),
        contention_enabled=False, handover_enabled=True))
    directory = root/"planned/physical_uav_handover"
    rows = [json.loads(s) for s in (directory/"episodes.jsonl").read_text().splitlines()]
    case = next(r for r in rows if r["num_agents"] == 2 and not r["success"])
    with gzip.open(directory/(case["case_id"]+".json.gz"), "rt", encoding="utf-8") as f:
        trace = json.load(f)
    env = ExperimentEnvironment(config, ExperimentMode.UAV, FIXED_MAP, 2, 250, case["seed"])
    observations, _ = env.reset()
    diagnostics = []
    last_timestamps = {a.id:0. for a in env.agents}
    for frame in trace["frames"][1:]:
        for agent in env.agents:
            cache = env.communication.caches[agent.id]["task"]
            assert cache.timestamp >= last_timestamps[agent.id]
            last_timestamps[agent.id] = cache.timestamp
            route = cache.value.consume(agent.position)
            fresh_path = list(agent.path)
            local_other_present = bool(observations[agent.id]["spatial"][1].any())
            diagnostics.append({
                "decision_step": env.step_count, "agent":agent.id,
                "position":list(agent.position), "cached_next":list(route[1]) if len(route)>1 else None,
                "fresh_next":list(fresh_path[1]) if len(fresh_path)>1 else None,
                "cached_route_contains_current_after_consumption": agent.position in cache.value.route,
                "task_source_time":cache.timestamp, "task_age":cache.age(env.communication.transport.now),
                "route_next_differs": route[1:2] != fresh_path[1:2],
                "local_other_present":local_other_present,
                "recorded_action":frame["actions"][str(agent.id)]})
        result = env.step({int(a):v for a,v in frame["actions"].items()})
        observations = result.observations
        assert [list(a.position) for a in env.agents] == frame["positions"]
        assert env.communication.snapshot()["cached_guidance"] == {
            int(a):[tuple(p) for p in route] for a,route in frame["cached_guidance"].items()}
    tx = [e for e in trace["transport_events"] if e["event"] == "transmit"]
    physical_ages = [d["task_age"] for d in diagnostics]
    stale_example = next(d for d in diagnostics if d["route_next_differs"] and not d["local_other_present"])
    output = {"case_id":case["case_id"], "recorded_actions_and_guidance_reproduced":True,
        "cache_timestamps_monotonic":True, "satellite_transmissions":sum(e["link"].startswith("satellite") for e in tx),
        "maximum_task_age_at_action_seconds":max(physical_ages),
        "action_inputs_with_different_next_waypoint":sum(d["route_next_differs"] for d in diagnostics),
        "example_without_nearby_robot":stale_example,
        "explanation":"Reports sent at the integer robot boundary arrive after that boundary; the next action uses the previous report. A learned off-route move can be followed by guidance pointing back, causing a persistent oscillation. No satellite hop is used by this UAV-local treatment.",
        "decisions":diagnostics}
    (root/"physical_latency_diagnostic.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k != "decisions"}))


if __name__ == "__main__":
    main()
