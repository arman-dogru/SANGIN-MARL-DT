"""Independently replay recorded PANAMA decisions and audit causal allocation."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from panama_uav.config import ProjectConfig
from panama_uav.models import select_actions
from panama_uav.records import sha256, write_json
from panama_uav.training import load_model


def check_task_accounting(job, config):
    """Check physical identities from saved measurements, without solver helpers."""
    close = lambda a,b: np.isclose(a,b,rtol=1e-7,atol=1e-8)
    assert close(job["required_cycles"], job["input_bits"]*job["cycles_per_bit"])
    stages=job.get("service_stages", {})
    access=stages.get("paper_input:access_up", {}).get("transmission_seconds",0)
    backhaul=stages.get("paper_cloud_input:satellite_up", {}).get("transmission_seconds",0)
    assert close(job["robot_energy_j"], config.resource_allocation.robot_power_w*access)
    assert close(job["backhaul_energy_j"], config.resource_allocation.drone_power_w*backhaul)
    if "predicted_allocation" in job:
        p=job["predicted_allocation"]
        assert p["x"] in (0,1)
        assert close(p["access_seconds"], job["input_bits"]/job["access_radio"]["applied_bps"])
        assert close(p["latency_seconds"], sum(p[k] for k in ("access_seconds","backhaul_seconds",
            "propagation_seconds","drone_compute_seconds","cloud_compute_seconds")))
        if p["x"]:
            assert close(p["drone_compute_seconds"],job["required_cycles"]/p["drone_cycles_per_second"])
            assert close(p["drone_compute_energy_j"],config.resource_allocation.drone_kappa*
                job["required_cycles"]*p["drone_cycles_per_second"]**2)
        else:
            radio=job["backhaul_radio"]
            assert close(p["backhaul_seconds"],job["input_bits"]/radio["applied_bps"])
            assert close(p["cloud_compute_seconds"],job["required_cycles"]/p["cloud_cycles_per_second"])
            assert close(p["propagation_seconds"],radio["round_trip_seconds"])
            assert close(radio["round_trip_seconds"],2*(radio["air_to_space_one_way_seconds"]+
                radio["satellite_gateway_one_way_seconds"]+radio["ground_one_way_seconds"]))
    if "delivered_at" in job:
        elapsed=sum(job["uplink_"+k] for k in ("transmission_seconds","queue_seconds",
            "propagation_seconds","interruption_seconds","guard_seconds"))
        elapsed+=job["batch_wait_seconds"]+job["compute_elapsed_seconds"]+job["response_seconds"]
        assert close(job["observed_latency_seconds"],job["delivered_at"]-job["source_time"])
        assert close(job["observed_latency_seconds"],elapsed), (job["job_id"],elapsed,job["observed_latency_seconds"])


def audit(study, output, cases_per_population=1):
    study, output = Path(study), Path(output)
    if json.loads((study/"status.json").read_text())["status"] != "complete":
        raise ValueError("Independent study audits require a completed run")
    if cases_per_population < 0:
        raise ValueError("Use zero for all cases or a positive per-population sample")
    manifest = json.loads((study/"manifest.json").read_text())
    case_ids={r["case_id"] for r in json.loads((study/"held_out_cases.json").read_text())}
    config = ProjectConfig.from_dict(manifest["configuration"])
    device = torch.device("cpu")
    torch.set_num_threads(1)
    model, _ = load_model(manifest["model"]["path"], config, device, require_qualified=not manifest["diagnostic"])
    rows = []
    for condition in manifest["conditions"]:
        directory = study/condition["name"]
        committed=[json.loads(s)["case_id"] for s in (directory/"episodes.jsonl").read_text().splitlines()]
        if len(committed)!=len(case_ids) or set(committed)!=case_ids:
            raise ValueError(f"Incomplete or duplicate case bank: {condition['name']}")
        files = [directory/f"{case_id}.json.gz" for case_id in sorted(case_ids)]
        if cases_per_population:
            files = [p for p in files if int(p.name.split("case-")[1].split(".")[0]) < cases_per_population]
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            for step in trace["allocation_loop"]:
                observations = {int(a): {key: np.asarray(value, np.float32) for key, value in values.items()}
                                for a, values in step["observations"].items()}
                actions = select_actions({0: model}, {a: 0 for a in observations}, observations, 0,
                                         np.random.default_rng(42), device)
                assert actions == {int(a): value for a, value in step["actions"].items()}
                for peers in step["input_cache_sources"].values():
                    assert all(item["source_time"] <= item["generation_time"] <= step["decision_at"]+1e-9
                               for item in peers.values())
            for frame in trace["frames"]:
                consistency = frame["consistency"]
                assert consistency["xi_fov"] == consistency["fov_errors"]/max(1, consistency["relevant_pairs"])
                for pair in consistency["pairs"]:
                    assert pair["e2e_ordering_error"] <= pair["uplink_ordering_error"]+pair["downlink_ordering_error"]
            for job in trace["resource_tasks"]:
                check_task_accounting(job, config)
                if "allocation_at" in job:
                    assert job["source_time"] <= job["access_arrived_at"] <= job["allocation_at"]
                    assert job["executed_x"] in (0, 1)
                if "delivered_at" in job:
                    assert job["compute_arrived_at"] <= job["compute_completed_at"] <= job["delivered_at"]
            allocations = [e for e in trace["transport_events"] if e["event"] == "paper_allocation"]
            assert all(e["raw_objective_difference"] >= -1e-6*max(1, abs(e["binary"]["objective"])) for e in allocations)
            for event in allocations:
                assert event["relaxed"]["is_dcp"]
                for mode in ("relaxed","binary"):
                    solution=event[mode]
                    assert solution["residual"]<=1e-6
                    assert sum(t["drone_share"] for t in solution["tasks"])<=1+1e-6
                    assert sum(t["cloud_share"] for t in solution["tasks"])<=1+1e-6
                    for task in solution["tasks"]:
                        x,a,b=task["x"],task["drone_share"],task["cloud_share"]
                        assert -1e-6<=x<=1+1e-6 and -1e-6<=a<=x+1e-6 and -1e-6<=b<=1-x+1e-6
                        if mode=="binary": assert x in (0,1)
            rows.append(dict(condition=condition["name"], case_id=trace["case"]["case_id"],
                decisions_replayed=len(trace["allocation_loop"]), allocation_snapshots=len(allocations),
                maximum_solver_residual=max((e["relaxed"]["residual"] for e in allocations),default=0),
                maximum_relative_relaxation_gap=max((e["relative_gap"] for e in allocations),default=0),
                tasks=len(trace["resource_tasks"]), trace_sha256=sha256(file), status="passed"))
    report = dict(status="passed", checkpoint_sha256=manifest["model"]["sha256"],
        manifest_sha256=sha256(study/"manifest.json"), audit_code_sha256=sha256(Path(__file__)),
        cases_per_population=cases_per_population, conditions_audited=len(manifest["conditions"]),
        episodes_audited=len(rows), decisions_replayed=sum(r["decisions_replayed"] for r in rows),
        allocation_snapshots=sum(r["allocation_snapshots"] for r in rows),
        maximum_solver_residual=max((r["maximum_solver_residual"] for r in rows),default=0),
        maximum_relative_relaxation_gap=max((r["maximum_relative_relaxation_gap"] for r in rows),default=0),
        task_accounting_checks=sum(r["tasks"] for r in rows), rows=rows)
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases-per-population", type=int, default=1)
    args = parser.parse_args()
    result = audit(args.study, args.output, args.cases_per_population)
    print(f"Replayed {result['decisions_replayed']} decisions across {result['episodes_audited']} traces.")
