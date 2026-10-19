"""Fresh, sequential perfect/full-LEO comparison with a frozen implementation."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC,datetime
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    os.replace(temp,path)


def utc():
    return datetime.now(UTC).isoformat()


def created(pid):
    # psutil is available with the retained Torch runtime.
    import psutil
    return datetime.fromtimestamp(psutil.Process(pid).create_time(),UTC).isoformat()


def worker(root,condition,resume=False):
    write(root/f"worker-identity-{condition}.json",{"pid":os.getpid(),"creation_utc":created(os.getpid()),
          "command_args":[sys.executable,"-u",*sys.argv]})
    sys.path.insert(0,str(root/"code"))
    from panama_uav.config import (ProjectConfig,MapConfig,RewardConfig,TrainingConfig,
        CurriculumStage,UAVConfig,TransportConfig,EvaluationConfig)
    from panama_uav.training import train_policy
    plan=json.loads((root/"plan.json").read_text())
    v=plan["configuration"]
    comm=dict(v["communication"],stale_information=tuple(v["communication"]["stale_information"]))
    ev=dict(v["evaluation"])
    ev.update(agent_counts=tuple(ev["agent_counts"]),delay_steps=tuple(ev["delay_steps"]))
    config=ProjectConfig(Path(v["project_root"]),MapConfig(**v["map"]),RewardConfig(**v["reward"]),
        TrainingConfig(**v["training"]),tuple(CurriculumStage(**s) for s in v["curriculum"]),
        UAVConfig(**v["uav"]),TransportConfig(**comm),EvaluationConfig(**ev))
    actual={p.name:digest(p) for p in (root/"code/panama_uav").glob("*.py")}
    if actual!=plan["code_sha256"] or digest(comm["leo_timeline"])!=plan["timeline_sha256"]:
        raise ValueError("Frozen source/data identity mismatch")
    result=train_policy(config,condition,42,root/condition,resume=resume,device_name="cuda",run_id=root.name)
    print(json.dumps({"condition":condition,"status":result.status}),flush=True)


def supervised_wait(process,root,condition,stall_seconds=600):
    """Bound a busy-loop failure; preserve the last complete checkpoint on disk."""
    import time
    import psutil
    start=time.time()
    while True:
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            path=root/condition
            activity=[p.stat().st_mtime for name in ("learning_steps.jsonl","training.jsonl",
                "validation_cases.jsonl","validation.jsonl","checkpoint.pt","checkpoint.pt.tmp","status.json")
                if (p:=path/name).exists()]
            idle=max(0.,time.time()-max(activity+[start]))
            write(root/"watchdog.json",{"checked_utc":utc(),"condition":condition,
                "seconds_without_progress":idle,"stall_threshold_seconds":stall_seconds})
            if idle>stall_seconds:
                error={"status":"stalled","condition":condition,"time":utc(),
                    "seconds_without_progress":idle,"checkpoint_preserved":str(path/"checkpoint.pt")}
                write(root/"watchdog_failure.json",error)
                parent=psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                try:parent.terminate()
                except psutil.NoSuchProcess:pass
                process.wait(timeout=20)
                raise RuntimeError(f"{condition} stalled for {idle:.0f}s; last valid checkpoint preserved")


def queue(args):
    sys.path.insert(0,str(PROJECT/"src"))
    from dataclasses import replace
    from panama_uav.config import load_config
    import importlib.metadata
    config=load_config(PROJECT/"configs/leo_learning.toml")
    root=(PROJECT/"output/leo-learning"/args.run_id).resolve()
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(PROJECT/"src/panama_uav",root/"code/panama_uav",ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("run_leo_study.py","report_leo_learning.py"):
        shutil.copy2(PROJECT/"scripts"/name,root/"code"/name)
    (root/"frozen_workspace/maps").mkdir(parents=True)
    shutil.copy2(PROJECT/"maps/random-32-32-10.map",root/"frozen_workspace/maps/random-32-32-10.map")
    (root/"datasets").mkdir()
    timeline=PROJECT/config.communication.leo_timeline
    shutil.copy2(timeline,root/"datasets/orbit.npz")
    shutil.copy2(PROJECT/"data/leo/orbit_source.json",root/"datasets/orbit_source.json")
    shutil.copy2(PROJECT/"data/leo/starlink_owd/manifest.json",root/"datasets/measurement_source.json")
    config=replace(config,project_root=root/"frozen_workspace",
        communication=replace(config.communication,leo_timeline=str(root/"datasets/orbit.npz")))
    if args.seconds is not None:
        config=replace(config,training=replace(config.training,budget_seconds=args.seconds,
                                              shutdown_reserve_seconds=min(5,args.seconds/10)))
    if args.steps is not None:
        config=replace(config,training=replace(config.training,target_environment_steps=args.steps))
    plan={"run_id":root.name,"conditions":["clean","combined"],"seed":42,
          "fresh_initializations":True,"stages":[2,4,6,8],"configuration":config.to_dict(),
          "budget_seconds_per_arm":config.training.budget_seconds,
          "mission_information":"fresh; learning uploads/deployments are the intervention",
          "code_sha256":{p.name:digest(p) for p in (root/"code/panama_uav").glob("*.py")},
          "timeline_sha256":digest(timeline),"map_sha256":digest(PROJECT/"maps/random-32-32-10.map"),
          "libraries":{name:importlib.metadata.version(name) for name in ("torch","numpy","sgp4","skyfield","simpy","itur","astropy")},
          "created_utc":utc(),"automatic_mission_inference":False}
    write(root/"plan.json",plan)
    write(root/"launch.json",{"pid":os.getpid(),"creation_utc":created(os.getpid()),
        "command_args":[sys.executable,"-u",*sys.argv],"command":subprocess.list2cmdline([sys.executable,"-u",*sys.argv])})
    state={"status":"running","active_condition":None,"completed":[],"queued":["clean","combined"]}
    try:
        for condition in plan["conditions"]:
            command=[sys.executable,"-u",str(root/"code/run_leo_study.py"),"--root",str(root),"--worker",condition]
            state.update(active_condition=condition,queued=[c for c in plan["conditions"] if c not in state["completed"] and c!=condition])
            with (root/f"{condition}.stdout.log").open("wb") as stdout,(root/f"{condition}.stderr.log").open("wb") as stderr:
                process=subprocess.Popen(command,cwd=PROJECT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                write(root/f"launch-{condition}.json",{"pid":process.pid,"creation_utc":created(process.pid),
                    "command_args":command,"command":subprocess.list2cmdline(command)})
                state["active_pid"]=process.pid
                write(root/"controller_status.json",state)
                print("START "+condition,flush=True)
                code=supervised_wait(process,root,condition)
            if code:
                raise RuntimeError(f"{condition} exited with {code}")
            result=json.loads((root/condition/"status.json").read_text())
            manifest=json.loads((root/condition/"manifest.json").read_text())
            if condition=="combined":
                reference=json.loads((root/"clean/manifest.json").read_text())
                for key in ("initial_weights_sha256","map_sha256","code_sha256","config"):
                    if manifest[key]!=reference[key]: raise ValueError(f"Matched-arm mismatch: {key}")
            state["completed"].append(condition)
            state.update(active_pid=None,last_result=result)
            write(root/"controller_status.json",state)
            subprocess.run([sys.executable,str(root/"code/report_leo_learning.py"),"--root",str(root)],check=True,cwd=PROJECT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        state.update(status="complete",active_condition=None,finished_utc=utc())
        write(root/"controller_status.json",state)
    except BaseException as error:
        state.update(status="error",error=str(error),failed_utc=utc())
        write(root/"controller_status.json",state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--worker",choices=("clean","combined"))
    parser.add_argument("--seconds",type=float)
    parser.add_argument("--steps",type=int)
    args=parser.parse_args()
    if args.worker:
        worker(args.root.resolve(),args.worker)
    elif args.run_id:
        queue(args)
    else: parser.error("--run-id or --worker/--root is required")


if __name__=="__main__": main()
