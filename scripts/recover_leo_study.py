"""Audited checkpoint continuation after the confirmed LEO frame-boundary stall.

Preserves the original experiment and qualified Clean reference. Copies only the
checkpoint-consistent log prefix into a new run, with a numerical scheduler fix.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_leo_study import PROJECT, created, digest, supervised_wait, utc, worker, write


def copy_prefix(source, target, size):
    if source.stat().st_size < size:
        raise ValueError(f"Checkpoint offset exceeds log length: {source}")
    with source.open("rb") as reader, target.open("xb") as writer:
        remaining = size
        while remaining:
            chunk = reader.read(min(remaining, 1024*1024))
            if not chunk:
                raise ValueError("Unexpected end of checkpoint log prefix")
            writer.write(chunk)
            remaining -= len(chunk)


def prepare(source, run_id):
    import torch
    import psutil
    sys.path.insert(0, str(PROJECT/"src"))
    source = source.resolve()
    original = json.loads((source/"plan.json").read_text())
    state = json.loads((source/"controller_status.json").read_text())
    watchdog_failure = source/"watchdog_failure.json"
    if state["status"] != "stopped_for_recovery" and not (
            state["status"] == "error" and watchdog_failure.exists()):
        raise ValueError("Source must be explicitly stopped or stopped by its watchdog")
    for name in ("launch.json", "worker-identity-combined.json", "launch-combined.json"):
        identity = json.loads((source/name).read_text())
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time()-__import__("datetime").datetime.fromisoformat(identity["creation_utc"]).timestamp()) < 1:
                raise ValueError(f"Original process remains alive: {identity['pid']}")
        except psutil.NoSuchProcess:
            pass
    root = (PROJECT/"output/leo-learning"/run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/"code/panama_uav", root/"code/panama_uav",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("leo.py", "training.py"):
        shutil.copy2(PROJECT/"src/panama_uav"/name, root/"code/panama_uav"/name)
    hashes = {p.name: digest(p) for p in (root/"code/panama_uav").glob("*.py")}
    differences = [name for name in hashes if hashes[name] != original["code_sha256"][name]]
    if not differences or not set(differences).issubset({"leo.py", "training.py"}):
        raise ValueError(f"Recovery must change only scheduler and CUDA RNG restoration: {differences}")
    for name in ("run_leo_study.py", "recover_leo_study.py", "report_leo_learning.py"):
        shutil.copy2(PROJECT/"scripts"/name, root/"code"/name)
    shutil.copytree(source/"clean", root/"clean")
    (root/"combined").mkdir()
    checkpoint_source = source/"combined/checkpoint.pt"
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    session = checkpoint["session"]
    if session.run_id != source.name or session.condition.value != "combined":
        raise ValueError("Checkpoint identity does not match the source arm")
    lineage = {
        "source_root": str(source), "source_run_id": source.name, "run_id": run_id,
        "created_utc": utc(), "reason": "Numerical scheduler boundary recovery; replayed interruption end previously rounded back to the same absolute cursor",
        "source_checkpoint_sha256": digest(checkpoint_source),
        "checkpoint_steps": session.generated_steps, "checkpoint_episodes": session.episodes,
        "checkpoint_elapsed_seconds": session.elapsed_seconds,
        "checkpoint_log_offsets": session.log_offsets,
        "code_changes": {name: {"before": original["code_sha256"][name], "after": hashes[name]} for name in differences},
        "checkpoint_migration": "Only session.run_id changes; models, optimizer, replay, pending messages, clocks, counters and saved RNG states are retained",
        "original_artifacts": "Retained; post-checkpoint rows are not appended a second time to the continuation",
        "clean_reference": "Previously qualified measured Clean arm, copied unchanged; not retrained",
        "budget": "Original 12h allocation minus checkpoint elapsed time; infrastructure hang and recovery work excluded and reported separately",
        "scientific_parameters_changed": False,
        "parent_recovery": json.loads((source/"recovery.json").read_text()) if (source/"recovery.json").exists() else None,
    }
    for name, size in session.log_offsets.items():
        copy_prefix(source/"combined"/name, root/"combined"/name, size)
    for stage in range(session.stage_index):
        model = source/"combined"/f"stage_{stage}_model.pt"
        if model.exists():
            shutil.copy2(model, root/"combined"/model.name)
    session.run_id = run_id
    torch.save(checkpoint, root/"combined/checkpoint.pt")
    lineage["continuation_checkpoint_sha256"] = digest(root/"combined/checkpoint.pt")
    previous = json.loads((source/"combined/manifest.json").read_text())
    previous.update(run_id=run_id, code_sha256=hashes)
    write(root/"combined/manifest.json", previous)
    plan = dict(original, run_id=run_id, code_sha256=hashes, created_utc=utc(),
                continuation_of=source.name, clean_reference_reused=True,
                active_conditions=["combined"], fresh_initializations=False)
    write(root/"plan.json", plan)
    write(root/"recovery.json", lineage)
    write(root/"launch.json", {"pid": os.getpid(), "creation_utc": created(os.getpid()),
          "command_args": [sys.executable, "-u", *sys.argv]})
    state.update(recovered_as=str(root))
    write(source/"controller_status.json", state)
    return root


def run(root):
    state = {"status": "running", "active_condition": "combined", "completed": ["clean"],
             "queued": [], "recovered_from": json.loads((root/"recovery.json").read_text())["source_root"]}
    command = [sys.executable, "-u", str(root/"code/recover_leo_study.py"), "--root", str(root), "--worker"]
    try:
        with (root/"combined.stdout.log").open("wb") as stdout, (root/"combined.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            write(root/"launch-combined.json", {"pid": process.pid, "creation_utc": created(process.pid),
                  "command_args": command})
            state["active_pid"] = process.pid
            write(root/"controller_status.json", state)
            code = supervised_wait(process, root, "combined")
        if code:
            raise RuntimeError(f"Recovered learner exited with {code}")
        result = json.loads((root/"combined/status.json").read_text())
        clean = json.loads((root/"clean/manifest.json").read_text())
        combined = json.loads((root/"combined/manifest.json").read_text())
        for key in ("initial_weights_sha256", "map_sha256", "config"):
            if clean[key] != combined[key]:
                raise ValueError(f"Recovery comparison mismatch: {key}")
        state.update(status="complete", active_condition=None, active_pid=None,
                     completed=["clean", "combined"], last_result=result, finished_utc=utc())
        write(root/"controller_status.json", state)
        subprocess.run([sys.executable, str(root/"code/report_leo_learning.py"), "--root", str(root)],
                       check=True, cwd=PROJECT, creationflags=subprocess.CREATE_NO_WINDOW)
    except BaseException as error:
        state.update(status="error", error=str(error), failed_utc=utc())
        write(root/"controller_status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.root.resolve(), "combined", resume=True)
    elif args.source and args.run_id:
        run(prepare(args.source, args.run_id))
    else:
        parser.error("--source and --run-id, or --root and --worker, are required")
