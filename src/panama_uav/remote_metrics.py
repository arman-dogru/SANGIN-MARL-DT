"""Remote-command accounting with explicit accepted/used/censored populations."""
import numpy as np

REMOTE_METRICS = ("tasks_attempted", "tasks_computed", "commands_generated", "commands_delivered",
    "commands_accepted", "commands_used", "commands_rejected_obsolete", "tasks_accepted", "tasks_used",
    "command_cutoff_misses", "stale_source_commands_used", "completed_local_control_steps",
    "task_deadline_misses", "deadline_censored_tasks", "mean_command_source_age_seconds",
    "mean_peer_source_age_seconds", "missed_physical_conflicts", "false_positive_conflicts",
    "zero_relevant_pair_decisions", "action_dt_ordering_errors", "action_order_known_pairs",
    "vertex_relevant_pairs", "edge_relevant_pairs", "physical_conflict_events", "collision_failure",
    "oracle_offload_switches", "mean_truth_scored_regret", "action_shadow_disagreements", "action_shadow_comparisons",
    "executed_drone_tasks", "executed_cloud_tasks", "radio_queue_seconds", "synchronization_wire_bits",
    "synchronization_energy_j", "post_goal_travel_steps", "post_goal_wait_steps", "physical_distance_cells",
    "task_latency_p50_seconds", "task_latency_p95_seconds", "task_latency_p99_seconds",
    "command_transport_p50_seconds", "command_transport_p95_seconds", "command_transport_p99_seconds",
    "access_rf_limited_fraction", "backhaul_rf_limited_fraction", "wrong_local_yield_steps", "unnecessary_wait_steps",
    "physical_distance_m", "mean_first_completion_time_seconds")


def remote_metrics(trace):
    events, tasks = trace["remote_events"], trace["resource_tasks"]
    decisions = [e for e in events if e["event"] == "decision_event"]
    arrivals = [e for e in events if e["event"] == "command_arrived"]
    used = [e for e in decisions if e["command"] is not None and not e["completed_local_control"]]
    pairs = [e["summary"] for e in events if e["event"] == "consistency_event"]
    shadows = [e for e in trace["oracle_events"] if e["event"] == "allocation_shadow"]
    action_shadows = [e for e in trace["oracle_events"] if e["event"] == "action_shadow"]
    local_shadows = [e for e in trace["oracle_events"] if e["event"] == "local_control_shadow"]
    shadow_actions = {e["job_id"]: e for e in action_shadows}
    transmitted = [e for e in trace["transport_events"] if e["event"] == "remote_transmit"]
    def mean(values):
        return float(np.mean(values)) if values else None
    metrics = dict(tasks_attempted=len(tasks), tasks_computed=sum("compute_completed_at" in t for t in tasks),
        commands_generated=sum(e["event"] == "remote_inference" for e in events),
        commands_delivered=len(arrivals), commands_accepted=sum(e["accepted"] for e in arrivals),
        commands_used=len(used), commands_rejected_obsolete=sum(e["reason"] == "rejected_obsolete" for e in arrivals),
        tasks_accepted=sum(t.get("command_accepted", False) for t in tasks), tasks_used=sum("used_at" in t for t in tasks),
        command_cutoff_misses=sum(e["command_cutoff_missed"] for e in decisions),
        stale_source_commands_used=sum(e["stale_source"] for e in used),
        completed_local_control_steps=sum(e["completed_local_control"] for e in decisions),
        task_deadline_misses=sum(t["task_deadline_missed"] for t in tasks),
        deadline_censored_tasks=sum(t["deadline_censored"] for t in tasks),
        mean_command_source_age_seconds=mean([e["cutoff_time"]-min(e["command"]["sources"].values()) for e in used]),
        mean_peer_source_age_seconds=mean([p["mean_peer_source_age_seconds"] for p in pairs]),
        missed_physical_conflicts=sum(p["missed_physical_conflicts"] for p in pairs),
        false_positive_conflicts=sum(p["false_positive_conflicts"] for p in pairs),
        zero_relevant_pair_decisions=sum(p["zero_relevant_pair_case"] for p in pairs),
        action_dt_ordering_errors=sum(p["action_dt_ordering_errors"] for p in pairs),
        action_order_known_pairs=sum(p["action_order_known_pairs"] for p in pairs),
        vertex_relevant_pairs=sum(p["vertex_relevant_pairs"] for p in pairs),
        edge_relevant_pairs=sum(p["edge_relevant_pairs"] for p in pairs),
        physical_conflict_events=sum(e["event"] in ("physical_collision", "physical_conflict") for e in events),
        collision_failure=int(trace["outcome"]["failure_reason"] == "robot_collision"),
        oracle_offload_switches=sum(len(e["offload_switches"]) for e in shadows),
        mean_truth_scored_regret=mean([e["truth_scored_regret"] for e in shadows]),
        action_shadow_disagreements=sum(e["disagrees"] for e in action_shadows),
        action_shadow_comparisons=len(action_shadows),
        wrong_local_yield_steps=sum(e["wrong_yield"] for e in local_shadows),
        unnecessary_wait_steps=sum(e["unnecessary_wait"] for e in local_shadows)+sum(
            e["action"] == 4 and shadow_actions.get(e["command"]["job_id"], {}).get("oracle_action", 4) != 4 for e in used),
        executed_drone_tasks=sum(t.get("executed_x") == 1 for t in tasks),
        executed_cloud_tasks=sum(t.get("executed_x") == 0 for t in tasks),
        radio_queue_seconds=sum(e["queue_seconds"] for e in transmitted),
        synchronization_wire_bits=sum(e["wire_bits"] for e in transmitted if e["category"] == "synchronization"),
        synchronization_energy_j=sum(e["energy_j"] for e in transmitted if e["category"] == "synchronization"),
        post_goal_travel_steps=0, post_goal_wait_steps=0, physical_distance_cells=0)
    for before, after in zip(trace["frames"], trace["frames"][1:]):
        for robot, done in before["completed_flags"].items():
            index = int(robot)
            moved = sum(abs(a-b) for a, b in zip(before["positions"][index], after["positions"][index]))
            metrics["physical_distance_cells"] += moved
            if done:
                metrics["post_goal_travel_steps" if moved else "post_goal_wait_steps"] += 1
    for name, values in (("task_latency", [t["observed_latency_seconds"] for t in tasks if "delivered_at" in t]),
                         ("command_transport", [e["arrived_at"]-e["command"]["generated_at"] for e in arrivals])):
        for percentile in (50, 95, 99):
            metrics[f"{name}_p{percentile}_seconds"] = float(np.quantile(values, percentile/100)) if values else None
    metrics["physical_distance_m"] = metrics["physical_distance_cells"]*trace["remote_settings"]["meters_per_cell"]
    motion = [e for e in events if e["event"] == "motion_event"]
    metrics["mean_first_completion_time_seconds"] = mean([v for v in motion[-1]["first_completion_times"].values()
        if v is not None]) if motion else None
    for name, budgets in (("access", [t["access_radio"] for t in tasks]),
                          ("backhaul", [t["backhaul_radio"] for t in tasks if "backhaul_radio" in t])):
        valid = [b for b in budgets if b.get("rate_bound_bps") is not None and b.get("available", True)]
        metrics[f"{name}_rf_limited_fraction"] = mean([float(b["rate_bound_bps"] < b["configured_cap_bps"]-1e-6)
            for b in valid])
        metrics[f"{name}_rf_budget_samples"] = len(valid)
    return metrics
