"""Paired field-by-field physical-latency diagnostic of the frozen Clean model."""
from dataclasses import replace
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from run_baseline_inference import reference_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/inference/baseline-20261006-01/latency-ablation"))
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    training = Path("output/learning-pilots/fast-curriculum-20261005-01").resolve()
    sys.path.insert(0,str(training/"code"))
    import panama_uav.evaluation as evaluation
    config, manifest = reference_config(training)
    model = training/"clean/final_model.pt"
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    identity = {"model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_code_sha256":manifest["code_sha256"], "cases_per_count":50,
        "purpose":"exploratory field-isolation diagnostic; positive physical access latency with no satellite traffic, handover, or reservation holds",
        "paired_to":"same held-out bank as baseline-20261006-01", "qualification":"partial Clean baseline"}
    identity_path = output/"study_manifest.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text(encoding="utf-8")) == identity
    write_json(identity_path,identity)
    original = evaluation.treatments
    def treatments(cfg):
        choices = [original(cfg)[0]]
        for field in ("task","intentions","priorities"):
            choices.append((f"physical_uav_{field}",replace(cfg.communication,
                profile="physical",mission_route="uav",stale_information=(field,),
                handover_enabled=False,contention_enabled=False)))
        return choices
    evaluation.treatments = treatments
    evaluation.gzip = SimpleNamespace(open=lambda *a,**kw:gzip.open(*a,compresslevel=1,**kw))
    append = evaluation.append_jsonl
    start = time.monotonic(); completed = 0
    def record(path,row):
        nonlocal completed
        append(path,row)
        if Path(path).name == "episodes.jsonl":
            completed += 1
            write_json(output/"status.json",{"status":"running","condition":Path(path).parent.name,
                "completed_episodes_this_invocation":completed,"last_case":row["case_id"],"elapsed_seconds":time.monotonic()-start})
            if completed % 50 == 0:
                print(f"{completed}/800 episodes; {Path(path).parent.name}; {time.monotonic()-start:.1f}s",flush=True)
    evaluation.append_jsonl = record
    try:
        evaluation.evaluate_suite(config,model,output,device_name=args.device,
                                  cases=50,counts=(2,4,6,8),diagnostic=True)
        write_json(output/"status.json",{"status":"complete","completed_episodes_this_invocation":completed,"elapsed_seconds":time.monotonic()-start})
    except BaseException as error:
        write_json(output/"status.json",{"status":"error","error":repr(error),"elapsed_seconds":time.monotonic()-start})
        raise


if __name__ == "__main__":
    main()
