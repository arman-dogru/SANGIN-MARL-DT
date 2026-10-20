"""Measured, paired inference tests of the preserved Clean checkpoint.

This is explicitly exploratory when the checkpoint is not fully qualified.
No training, policy, reward, collision, or local-sensing behavior is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(24):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 23:
                raise
            # Windows readers may briefly omit FILE_SHARE_DELETE. This is
            # metadata I/O only; no simulated communication time advances.
            time.sleep(min(.5,.05*2**min(attempt,4)))


@contextmanager
def peer_cutoffs(communication_class):
    """Drop peer-information receipts during seeded windows; keep task/local data fresh."""
    receive = communication_class._receive

    def blocked_receive(self, packet):
        if packet.kind == "information":
            fields = set(packet.payload)
            if fields and fields <= {"intentions", "priorities"}:
                for start, end in self.transport.windows.get(packet.agent, []):
                    if start <= self.transport.now < end:
                        self.transport.log.append({
                            "event": "peer_cutoff_drop", "time": self.transport.now,
                            "agent": packet.agent, "id": packet.id,
                            "fields": sorted(fields), "start": start, "end": end,
                        })
                        return
        return receive(self, packet)

    communication_class._receive = blocked_receive
    try:
        yield
    finally:
        communication_class._receive = receive


def reference_config(root):
    from panama_uav.config import (
        CurriculumStage, EvaluationConfig, MapConfig, ProjectConfig,
        RewardConfig, TrainingConfig, TransportConfig, UAVConfig,
    )
    manifest = json.loads((root / "clean/manifest.json").read_text(encoding="utf-8"))
    values = manifest["config"]
    comm = dict(values["communication"])
    comm["stale_information"] = tuple(comm["stale_information"])
    evaluation = dict(values["evaluation"])
    for field in ("agent_counts", "delay_steps"):
        evaluation[field] = tuple(evaluation[field])
    config = ProjectConfig(
        Path(values["project_root"]), MapConfig(**values["map"]),
        RewardConfig(**values["reward"]), TrainingConfig(**values["training"]),
        tuple(CurriculumStage(**stage) for stage in values["curriculum"]),
        UAVConfig(**values["uav"]), TransportConfig(**comm),
        EvaluationConfig(**evaluation),
    )
    if hasattr(ProjectConfig, "from_dict"):
        config = ProjectConfig.from_dict(values)
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / "code/panama_uav").glob("*.py")}
    if actual != manifest["code_sha256"]:
        raise ValueError("Frozen reference package hash mismatch")
    return config, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path,
                        default=Path("output/learning-pilots/fast-curriculum-20261005-01"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/inference/baseline-20261006-01"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--agents", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--suites", nargs="+", choices=["planned", "orbital", "peer-cutoff", "empirical"],
                        default=["planned", "orbital", "peer-cutoff"])
    args = parser.parse_args()
    root = args.training_root.resolve()
    sys.path.insert(0, str(root / "code"))
    import panama_uav.evaluation as evaluation
    from panama_uav.information import MissionCommunication
    config, reference = reference_config(root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = root / "clean/final_model.pt"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    previous = output / "study_manifest.json"
    identity = {"model_sha256": model_hash, "cases_per_count": args.cases,
                "agent_counts": args.agents, "suites": args.suites,
                "reference_run": root.name, "reference_code_sha256": reference["code_sha256"],
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "trace_compression_level": 1}
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing inference identity differs; use a new output directory")
    write_json(previous, {**identity, "qualification": "partial: Clean did not qualify eight agents",
               "purpose": "user-authorized exploratory inference; not the qualified confirmatory study",
               "paired_cases": f"same {args.cases} held-out starts/goals/seeds per agent count",
               "episode_cap": 250, "policy": "frozen greedy Clean final model",
               "peer_cutoff": "discard intentions/priorities during 1-2 seeded windows; goals, guidance and local sensing remain fresh",
               "orbital_source": "saved Mont-Tremblant timeline from archived orbital data; handshake/capacity assumptions retained"})
    start = time.monotonic()
    state = {"status": "running", "completed_suites": [], "active_suite": None,
             "active_condition": None, "completed_episodes_this_invocation": 0}
    status_path = output / "status.json"
    original_append = evaluation.append_jsonl
    original_episode = evaluation.run_episode
    original_treatments = evaluation.treatments
    original_gzip = evaluation.gzip
    current_suite = None

    def record_episode(path, row):
        original_append(path, row)
        if Path(path).name == "episodes.jsonl":
            state.update(active_condition=Path(path).parent.name,
                         latest_agent_count=row["num_agents"], latest_case=row["case_id"],
                         elapsed_seconds=time.monotonic() - start)
            state["completed_episodes_this_invocation"] += 1
            write_json(status_path, state)
            if state["completed_episodes_this_invocation"] % 50 == 0:
                print(json.dumps(state), flush=True)

    def measured_episode(*positional, **keywords):
        row, trace = original_episode(*positional, **keywords)
        for field in ("task", "intentions", "priorities"):
            ages = [age[field] for frame in trace["frames"]
                    for age in frame["information_age"].values()]
            row[f"maximum_{field}_age_seconds"] = max(ages)
        row["maximum_queue_bytes"] = max(sum(f["queue_bytes"].values()) for f in trace["frames"])
        transmissions = [e for e in trace["transport_events"] if e["event"] == "transmit"]
        satellite = [e for e in transmissions if e["link"].startswith("satellite")]
        # Floating-point subtraction on an access hop is not a handover.
        row["transmissions_interrupted_by_handover"] = int(sum(e.get("handover_seconds", 0) > 1e-9 for e in satellite))
        row["handover_wait_seconds"] = float(sum(e["handover_seconds"] for e in satellite if e.get("handover_seconds", 0) > 1e-9))
        row["handover_windows_elapsed"] = sum(e["event"] in ("handover_start", "orbit_handover") for e in trace["transport_events"])
        if current_suite == "peer-cutoff":
            drops = [e for e in trace["transport_events"] if e["event"] == "peer_cutoff_drop"]
            row["peer_messages_dropped"] = len(drops)
            row["exposed_agents"] = len({e["agent"] for e in drops})
            row["exposed_messages"] = len(drops)
            row["exposure_by_agent"] = {a: sum(e["agent"] == a for e in drops)
                                        for a in range(row["num_agents"])}
            row["realized_windows_by_agent"] = {
                a: len({(e["start"], e["end"]) for e in drops if e["agent"] == a})
                for a in range(row["num_agents"])}
        return row, trace

    def expanded_treatments(cfg):
        if current_suite == "peer-cutoff":
            choices = [original_treatments(cfg)[0]]
            for severity in cfg.evaluation.delay_steps:
                choices.append((f"peer_cutoff_{severity}", replace(
                    cfg.communication, profile="controlled", mission_route="uav",
                    stale_information=("intentions", "priorities"),
                    contention_enabled=False, handover_enabled=False,
                    delay_steps=0, reservation_steps=int(severity))))
            return choices
        choices = original_treatments(cfg)
        if current_suite in ("orbital", "empirical"):
            choices = [item for item in choices if item[0] == "perfect" or item[0].startswith("physical_")]
        if current_suite == "empirical":
            # Trace spikes do not identify handovers. Only no-handshake latency
            # and controlled resource contention are tested for imported traces.
            choices = [item for item in choices if item[0] == "perfect" or item[0].endswith("_contention")]
        for route in ("uav", "cloud"):
            choices.append((f"physical_{route}_latency", replace(
                cfg.communication, profile="physical", mission_route=route,
                stale_information=("task", "intentions", "priorities"),
                handover_enabled=False, contention_enabled=False)))
        return choices

    evaluation.append_jsonl = record_episode
    evaluation.run_episode = measured_episode
    evaluation.treatments = expanded_treatments
    # Lossless compression level only affects output writing, never simulation.
    evaluation.gzip = SimpleNamespace(open=lambda *a, **kw: gzip.open(*a, compresslevel=1, **kw))
    try:
        for suite in args.suites:
            current_suite = suite
            state.update(active_suite=suite, active_condition=None)
            write_json(status_path, state)
            variant = config
            if suite == "orbital":
                timeline = config.resolve("data/leo/tremblant_6h.npz")
                if not timeline.exists():
                    raise FileNotFoundError(timeline)
                variant = replace(config, communication=replace(
                    config.communication, leo_timeline=str(timeline),
                    backhaul_time_offset_seconds=1000.0))
            elif suite == "empirical":
                import numpy as np
                for direction in ("uplink", "downlink"):
                    trace_path = config.resolve(f"data/leo/starlink_owd/{direction}.npz")
                    with np.load(trace_path, allow_pickle=False) as trace:
                        if trace["timestamps"][0] > 1 or trace["timestamps"][-1] < 250:
                            raise ValueError(
                                f"{direction} trace covers {trace['timestamps'][0]:.3f}–"
                                f"{trace['timestamps'][-1]:.3f} seconds, insufficient for "
                                "250-step paired missions; provide a longer recording. "
                                "Measurements will not be looped or extrapolated.")
                variant = replace(config, communication=replace(
                    config.communication, leo_timeline=None, handover_enabled=False,
                    uplink_trace=str(config.resolve("data/leo/starlink_owd/uplink.npz")),
                    downlink_trace=str(config.resolve("data/leo/starlink_owd/downlink.npz"))))
            options = dict(device_name=args.device, cases=args.cases,
                           counts=tuple(args.agents), diagnostic=True)
            if suite == "peer-cutoff":
                with peer_cutoffs(MissionCommunication):
                    evaluation.evaluate_suite(variant, model_path, output / suite, **options)
            else:
                evaluation.evaluate_suite(variant, model_path, output / suite, **options)
            state["completed_suites"].append(suite)
            write_json(status_path, state)
        state.update(status="complete", active_suite=None, active_condition=None,
                     elapsed_seconds=time.monotonic() - start)
        write_json(status_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="error", error=f"{type(error).__name__}: {error}",
                     elapsed_seconds=time.monotonic() - start)
        write_json(status_path, state)
        raise
    finally:
        evaluation.append_jsonl = original_append
        evaluation.run_episode = original_episode
        evaluation.treatments = original_treatments
        evaluation.gzip = original_gzip


if __name__ == "__main__":
    main()
