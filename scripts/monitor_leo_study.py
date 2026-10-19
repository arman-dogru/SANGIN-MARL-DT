"""Read-only process/measurement audit; appends compact monitoring history."""
import argparse
import json
from datetime import UTC,datetime
from pathlib import Path

import psutil


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def last(path):
    if not path.exists():return None
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-131072))
        lines=stream.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):continue
    return None


def verify(identity):
    if not identity:return {"verified":False,"reason":"identity file not yet present"}
    try:
        process=psutil.Process(identity["pid"])
        created=datetime.fromtimestamp(process.create_time(),UTC)
        recorded=datetime.fromisoformat(identity["creation_utc"])
        command=process.cmdline()
        expected=identity["command_args"]
        # The Windows venv launcher can use the base executable. Arguments and
        # creation time still have to match; process name alone is insufficient.
        same_arguments=[x.replace("\\","/").lower() for x in command[1:]]==[x.replace("\\","/").lower() for x in expected[1:]]
        same_executable=Path(command[0]).name.lower()==Path(expected[0]).name.lower()
        verified=abs((created-recorded).total_seconds())<1 and same_arguments and same_executable
        return {"verified":verified,"pid":process.pid,"creation_utc":created.isoformat(),
                "command_args":command,"reason":None if verified else "identity mismatch"}
    except (psutil.NoSuchProcess,psutil.AccessDenied) as error:
        return {"verified":False,"pid":identity["pid"],"reason":str(error)}


def health(root, condition, arm, state, now, stall_seconds=1200):
    """Process existence is not progress. Validation/checkpoint writes count too."""
    if arm["status"]:
        return {"state": "qualified" if arm["status"].get("qualified") else "partial",
                "reason": arm["status"].get("status")}
    if state and state.get("status") == "error":
        return {"state": "failed", "reason": state.get("error", "Controller failed")}
    if state and state.get("status") == "complete" and condition in state.get("completed", []):
        return {"state": "failed", "reason": "Completed arm has no terminal status"}
    if not state or state.get("active_condition") != condition:
        return {"state": "queued"}
    if state.get("status") != "running":
        return {"state": state.get("status", "unknown")}
    if not arm["process"]["verified"]:
        return {"state": "failed", "reason": "Active worker identity is absent or mismatched"}
    path = Path(root)/condition
    files = [path/name for name in ("learning_steps.jsonl", "training.jsonl",
             "validation_cases.jsonl", "validation.jsonl", "checkpoint.pt", "checkpoint.pt.tmp")]
    stamps = [p.stat().st_mtime for p in files if p.exists()]
    if not stamps:
        stamps = [datetime.fromisoformat(arm["process"]["creation_utc"]).timestamp()]
    idle = max(0., now.timestamp()-max(stamps))
    return {"state": "stalled" if idle > stall_seconds else "progressing",
            "seconds_without_progress": round(idle, 1), "stall_threshold_seconds": stall_seconds,
            "reason": "No generated-step, validation-case or checkpoint progress" if idle > stall_seconds else None}


def snapshot(root):
    root=Path(root)
    state=read(root/"controller_status.json")
    now=datetime.now(UTC)
    result={"checked_utc":now.isoformat(),"controller_status":state,
            "controller":verify(read(root/"launch.json")),"arms":{}}
    for condition in ("clean","combined"):
        path=root/condition
        identity=read(root/f"worker-identity-{condition}.json") or read(root/f"launch-{condition}.json")
        step=last(path/"learning_steps.jsonl")
        result["arms"][condition]={"process":verify(identity),"status":read(path/"status.json"),
            "latest_training_episode":last(path/"training.jsonl"),
            "latest_validation":last(path/"validation.jsonl"),
            "latest_step":{key:step.get(key) for key in ("generated_joint_steps","stage_index","global_episode",
                "delivered_experiences","learner_updates","model_version_lag","deployed_model_age_seconds",
                "experience_delivery_age_seconds","queue_bytes","leo_state","wall_seconds")} if step else None,
            "checkpoint_updated_utc":datetime.fromtimestamp((path/"checkpoint.pt").stat().st_mtime,UTC).isoformat()
                                      if (path/"checkpoint.pt").exists() else None}
        result["arms"][condition]["health"]=health(root,condition,result["arms"][condition],state,now)
    result["alerts"]=[{"condition":name,**arm["health"]} for name,arm in result["arms"].items()
                      if arm["health"]["state"] in ("stalled","failed")]
    if state and state.get("status") == "running" and not result["controller"]["verified"]:
        result["alerts"].append({"condition":"controller","state":"failed",
                                 "reason":"Running controller identity is absent or mismatched"})
    with (root/"monitor_history.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(result)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    print(json.dumps(snapshot(parser.parse_args().root),indent=2))
