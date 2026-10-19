"""Verify completed paired inference records without modifying raw outputs."""
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("output/inference/baseline-20261006-01")
    study = json.loads((root/"study_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((root/"status.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    training = Path("output/learning-pilots/fast-curriculum-20261005-01")
    assert digest(training/"clean/final_model.pt") == study["model_sha256"]
    assert digest(root/"runner.py") == study["runner_sha256"]
    if (root/"resume_launch.json").exists():
        resumed = json.loads((root/"resume_launch.json").read_text(encoding="utf-8-sig"))
        assert digest(Path(resumed["wrapper"])) == resumed["wrapper_sha256"]
        assert resumed["original_runner_sha256"] == study["runner_sha256"]
    assert {p.name:digest(p) for p in (training/"code/panama_uav").glob("*.py")} == study["reference_code_sha256"]
    suites = {}
    perfect = None
    bank_hash = None
    total = 0
    for suite, expected in (("planned",49),("orbital",9),("peer-cutoff",6),("latency-ablation",4)):
        folder = root/suite
        manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["diagnostic"] and manifest["model_sha256"] == study["model_sha256"]
        bank = json.loads((folder/"held_out_cases.json").read_text(encoding="utf-8"))
        if bank_hash is None:
            bank_hash = digest(folder/"held_out_cases.json")
        assert digest(folder/"held_out_cases.json") == bank_hash
        reference = {r["case_id"]:r for r in bank["cases"]}
        assert len(reference) == 200
        assert len(manifest["conditions"]) == expected
        checks = []
        for condition in manifest["conditions"]:
            name = condition["name"]
            directory = folder/name
            rows = [json.loads(line) for line in (directory/"episodes.jsonl").read_text().splitlines()]
            assert len(rows) == 200 and {r["case_id"] for r in rows} == set(reference)
            assert all(sum(r["num_agents"] == n for r in rows) == 50 for n in (2,4,6,8))
            for row in rows:
                assert row["diagnostic"] and row["steps"] <= 250
                assert row["par2"] == (row["steps"] if row["success"] else 500)
                assert all(row[k] == reference[row["case_id"]][k] for k in ("seed","starts","goals","num_agents"))
                assert (directory/(row["case_id"]+".json.gz")).exists()
                assert row["stale_information"] == condition["transport"]["stale_information"]
                assert row["mission_route"] == condition["transport"]["mission_route"]
                for field in ("task","intentions","priorities"):
                    if field not in row["stale_information"]:
                        assert row[f"mean_{field}_age_seconds"] == 0
                        if f"maximum_{field}_age_seconds" in row:
                            assert row[f"maximum_{field}_age_seconds"] == 0
            if name == "perfect":
                comparable = [{k:r[k] for k in ("case_id","success","steps","par2","collision_attempts","uav_travel_m")} for r in rows]
                if perfect is None:
                    perfect = comparable
                assert comparable == perfect
            summaries = json.loads((directory/"summary.json").read_text())
            for summary in summaries:
                group = [r for r in rows if r["num_agents"] == summary["num_agents"]]
                assert summary["cases"] == summary["paired_cases"] == 50
                assert abs(sum(r["success"] for r in group)/50-summary["success_rate"]) < 1e-12
                assert abs(sum(r["par2"] for r in group)/50-summary["par2"]) < 1e-9
            with gzip.open(directory/(rows[0]["case_id"]+".json.gz"), "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            assert len(trace["frames"]) == rows[0]["steps"]+1
            versions = {v for f in trace["frames"] for v in f["model_versions"].values()}
            assert len(versions) == 1
            checks.append({"condition":name,"episodes":len(rows),"successes":sum(r["success"] for r in rows),
                           "trace_sample_verified":True,"frozen_model_version":next(iter(versions))})
            total += len(rows)
        suites[suite] = checks
    ablation = root/"latency-ablation"
    assert json.loads((ablation/"status.json").read_text())["status"] == "complete"
    ablation_study = json.loads((ablation/"study_manifest.json").read_text())
    assert ablation_study["model_sha256"] == study["model_sha256"]
    assert digest(ablation/"runner.py") == ablation_study["runner_sha256"]
    assert ablation_study["reference_code_sha256"] == study["reference_code_sha256"]
    assert total == 13600
    # Satellite scheduling cannot change UAV-local mission traffic. Check
    # all outcomes against a control with the same ordinary access latency.
    for suite in ("planned","orbital"):
        controls = []
        for name in ("physical_uav_latency","physical_uav_handover"):
            rows = [json.loads(s) for s in (root/suite/name/"episodes.jsonl").read_text().splitlines()]
            controls.append([{k:r[k] for k in ("case_id","steps","success","par2","collision_attempts","uav_travel_m")} for r in rows])
        assert controls[0] == controls[1]
    audit = {"status":"passed","episode_records":total,"distinct_paired_cases":200,
             "all_episode_caps_250_verified":True,"selected_input_isolation_verified":True,
             "case_bank_hash":bank_hash,"perfect_references_identical":True,
             "model_and_frozen_code_hashes_unchanged":True,"suites":suites,
             "uav_local_handover_matches_latency_control":True,
             "metadata_io_recovery": "Windows sharing-lock retry wrapper; frozen runner and policy unchanged"
                 if (root/"resume_launch.json").exists() else None,
             "qualification":"exploratory partial Clean checkpoint; no graduation rules changed"}
    (root/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k != "suites"}))


if __name__ == "__main__":
    main()
