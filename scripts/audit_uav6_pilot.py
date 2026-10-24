"""Read-only audit and active-load strata for completed parameter-only pilots."""
from collections import defaultdict
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from panama_uav.communication_report import csv_table
from panama_uav.records import sha256, write_json


def compare_frames(original, current, capacity_changed=False):
    # Capacity is serialized in allocation_event snapshots, so changing an
    # unused UAV capacity also changes the event-chain checksum in cloud runs.
    # No physical, cache, command, timing or other frame field is omitted.
    if capacity_changed:
        original = [{k: v for k, v in frame.items() if k != "event_checksum"} for frame in original]
        current = [{k: v for k, v in frame.items() if k != "event_checksum"} for frame in current]
    assert original == current


def same_records(root, packets, name, field, identity):
    expected = {}
    for packet in packets:
        values = packet[field] if isinstance(packet[field], list) else [packet[field]]
        for row in values:
            key = tuple(row[k] for k in identity)
            assert key not in expected
            expected[key] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    found = set()
    with (root / (name + ".jsonl")).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in identity)
            assert key not in found and key in expected
            assert hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() == expected[key]
            found.add(key)
    assert found == set(expected)
    return len(found)


def audit(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    spec = importlib.util.spec_from_file_location("frozen_uav6", root / "code/run_uav6_pilot.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest = helper.verify_inputs(root)
    assert json.loads((root / "status.json").read_text())["status"] == "complete"
    packets = [json.loads(p.read_text()) for p in sorted((root / "episodes").glob("*/*.result.json"))]
    counts = {}
    for name, field, identity in (("episodes", "row", ("condition", "case_id")),
        ("tasks", "tasks", ("condition", "case_id", "job_id")),
        ("rounds", "rounds", ("condition", "case_id", "cycle")),
        ("audits", "audit", ("condition", "case_id"))):
        counts[name] = same_records(root, packets, name, field, identity)
    cohorts = defaultdict(list)
    first_round = []
    for packet in packets:
        row, tasks = packet["row"], packet["tasks"]
        cycle_counts = {r["cycle"]: r["tasks"] for r in packet["rounds"]}
        for task in tasks:
            cohorts[row["condition"], row["num_agents"], cycle_counts[task["cycle"]]].append(task)
        group = [t for t in tasks if t["cycle"] == 0]
        first_round.append(dict(label=helper.LABEL, condition=row["condition"], num_agents=row["num_agents"],
            case_id=row["case_id"], unavailable_admissions=sum(t["backhaul_available_at_admission"] is False for t in group),
            **helper.describe(group)))
    strata = []
    for (condition, population, active), tasks in sorted(cohorts.items()):
        for available in ("all", "available", "unavailable"):
            subset = [t for t in tasks if available == "all" or
                      t["backhaul_available_at_admission"] == (available == "available")]
            if subset:
                strata.append(dict(label=helper.LABEL, condition=condition, num_agents=population,
                    active_robots=active, backhaul_admission=available,
                    cutoff_waits=sum(t["cutoff_induced_wait"] for t in subset), **helper.describe(subset)))
    csv_table(root / "placement_by_active_count.csv", strata)
    csv_table(root / "first_round.csv", first_round)
    handover = []
    grouped = defaultdict(list)
    for packet in packets:
        grouped[packet["row"]["condition"], packet["row"]["num_agents"]].extend(packet["tasks"])
    for (condition, population), tasks in sorted(grouped.items()):
        for name, subset in (("handover_at_sensing", [t for t in tasks if t["handover_at_sensing"]]),
            ("outside_handover_at_sensing", [t for t in tasks if not t["handover_at_sensing"]]),
            ("unavailable_at_admission", [t for t in tasks if t["backhaul_available_at_admission"] is False]),
            ("available_at_admission", [t for t in tasks if t["backhaul_available_at_admission"] is True])):
            elapsed = [t["compute_elapsed_seconds"] for t in subset if t.get("compute_elapsed_seconds") is not None]
            handover.append(dict(label=helper.LABEL, condition=condition, num_agents=population, subset=name,
                cutoff_waits=sum(t["cutoff_induced_wait"] for t in subset),
                compute_elapsed_mean_s=sum(elapsed)/len(elapsed) if elapsed else None, **helper.describe(subset)))
    csv_table(root / "handover_details.csv", handover)
    references = []
    for packet in packets:
        row = packet["row"]
        if row["condition"] not in ("allocated-1ghz", "cloud-6ghz"):
            continue
        route = "allocated__nominal" if row["condition"] == "allocated-1ghz" else "cloud__nominal"
        old = previous / "episodes" / route / (row["case_id"] + ".json.gz")
        with gzip.open(old, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        with gzip.open(packet["audit"]["archive"], "rt", encoding="utf-8") as stream:
            current = json.load(stream)
        compare_frames(trace["frames"], current["frames"], capacity_changed=row["condition"] == "cloud-6ghz")
        old_tasks = {t["job_id"]: t for t in trace["resource_tasks"]}
        assert len(old_tasks) == len(current["resource_tasks"])
        for task in current["resource_tasks"]:
            original = old_tasks[task["job_id"]]
            for field in ("executed_x", "source_time", "delivered_at", "predicted_latency_seconds", "observed_latency_seconds"):
                left, right = task.get(field), original.get(field)
                assert left == right or (isinstance(left, (int, float)) and isinstance(right, (int, float))
                                        and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)), (row["case_id"], field)
        references.append(dict(condition=row["condition"], case_id=row["case_id"], matched=True,
            historical_archive=str(old), historical_archive_sha256=sha256(old)))
    result = dict(status="passed", label=helper.LABEL, aggregate_records=counts,
        aggregate_transactions_identical=True, historical_reference_cases=len(references),
        baseline_frames_identical=True, all_cloud_frames_identical_except_capacity_event_checksum=True,
        placement_and_command_times_identical=True,
        active_count_stratification=True, script_sha256=sha256(__file__),
        manifest_sha256=sha256(root / "manifest.json"))
    write_json(root / "independent_audit.json", result)
    write_json(root / "baseline_reproduction.json", references)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("output/communication_effects/inference-20261023-01"))
    args = parser.parse_args()
    audit(args.root, args.previous)
