from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from .config import LearningCondition, load_config
from .evaluation import evaluate_suite, treatments
from .replay_viewer import create_replay
from .reporting import report_evaluation, report_training
from .training import compare_training, train_policy
from .transport import import_trace


def serve(path, port):
    import mimetypes
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse

    path = Path(path).resolve()
    root = path.parent

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = unquote(urlparse(self.path).path).lstrip("/") or path.name
            target = (root / name).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Replay: http://127.0.0.1:{port}/{path.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parser():
    result = argparse.ArgumentParser(
        description="UAV-PANAMA measured communication staleness study"
    )
    result.add_argument("--config", default="configs/base.toml")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("train", "compare-training", "study", "evaluate", "replay"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-id", required=True)
        sub.add_argument("--output-root", default="output/communication")
        sub.add_argument("--device", default="auto")
        if name in ("train", "compare-training", "study"):
            sub.add_argument("--seed", type=int, default=42, choices=(42,))
            sub.add_argument("--profile", choices=("full", "smoke"), default="full")
            sub.add_argument(
                "--communication-profile",
                choices=("controlled", "physical"),
                default="controlled",
            )
            sub.add_argument("--delay-steps", type=float, default=5)
            sub.add_argument("--no-resume", action="store_true")
            if name == "train":
                sub.add_argument(
                    "--condition",
                    choices=[c.value for c in LearningCondition],
                    default="clean",
                )
        if name in ("evaluate", "replay"):
            sub.add_argument("--model")
            sub.add_argument(
                "--diagnostic",
                action="store_true",
                help="Allow an unqualified model; outputs clearly labeled diagnostic",
            )
        if name == "evaluate":
            sub.add_argument("--cases", type=int, default=50)
            sub.add_argument(
                "--agents",
                type=int,
                nargs="+",
                choices=(2, 4, 6, 8),
                default=[2, 4, 6, 8],
            )
            sub.add_argument(
                "--conditions",
                nargs="+",
                help="Treatment names, e.g. controlled_cloud_task_5; perfect is always included",
            )
        if name == "replay":
            sub.add_argument("--agents", type=int, choices=(2, 4, 6, 8), default=2)
            sub.add_argument("--delay-steps", type=float, default=10)
            sub.add_argument(
                "--mission-route", choices=("uav", "cloud"), default="cloud"
            )
    sub = commands.add_parser("report")
    sub.add_argument("--path", required=True)
    sub.add_argument("--kind", choices=("training", "inference"), default="training")
    sub = commands.add_parser("import-trace")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--direction", choices=("uplink", "downlink"), required=True)
    sub.add_argument("--unit", choices=("ns", "us", "ms", "s"), default="ns")
    sub.add_argument("--start", type=float)
    sub.add_argument("--end", type=float)
    sub = commands.add_parser("serve-replay")
    sub.add_argument("--path", required=True)
    sub.add_argument("--port", type=int, default=8765)
    sub = commands.add_parser("allocation-snapshot", help="Solve and compare a fixed Equation 16 snapshot")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub = commands.add_parser("allocation-sanity", help="Run predictable independently verified allocation cases")
    sub.add_argument("--output", required=True)
    sub = commands.add_parser("remote-audit", help="Rebuild and verify a remote-DT event archive")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output")
    sub.add_argument("--model", help="Also verify frozen DQN actions with this checkpoint")
    sub.add_argument("--device", default="cpu")
    sub = commands.add_parser("resource-conditions", help="List resource experiment conditions")
    sub.add_argument("--primary-only", action="store_true")
    sub = commands.add_parser("resource-report", help="Rebuild resource summaries and figures")
    sub.add_argument("--path", required=True)
    sub = commands.add_parser("resource-study", help="Run paired frozen-policy resource experiments")
    sub.add_argument("--run-id", required=True)
    sub.add_argument("--output-root", default="output/resources")
    sub.add_argument("--model")
    sub.add_argument("--timeline")
    sub.add_argument("--device", default="cpu")
    sub.add_argument("--cases", type=int, default=50)
    sub.add_argument("--agents", type=int, nargs="+", choices=(2, 4, 6, 8), default=[2, 4, 6, 8])
    sub.add_argument("--max-steps", type=int, default=250)
    sub.add_argument("--diagnostic", action="store_true")
    sub.add_argument("--primary-only", action="store_true")
    sub.add_argument("--conditions", nargs="+")
    sub.add_argument("--workers", type=int, default=1)
    sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("deadline-study", help="Run paired 100/250/300/500 ms remote command-cutoff sensitivity")
    sub.add_argument("--run-id", required=True)
    sub.add_argument("--output-root", default="output/deadlines")
    sub.add_argument("--model", required=True)
    sub.add_argument("--timeline")
    sub.add_argument("--device", default="cpu")
    sub.add_argument("--cases", type=int, default=1)
    sub.add_argument("--agents", type=int, nargs="+", choices=(2, 4, 6, 8), default=[2, 4, 6, 8])
    sub.add_argument("--max-steps", type=int, default=250)
    sub.add_argument("--workers", type=int, default=1)
    sub.add_argument("--diagnostic", action="store_true")
    sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("deadline-report", help="Verify pairing and rebuild deadline comparison tables and plots")
    sub.add_argument("--path", required=True)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "remote-audit":
        from .remote_audit import audit_archive
        inference = None
        if arguments.model:
            import numpy as np
            from .training import load_model
            from .models import choose_device, select_action
            config, device = load_config(arguments.config), choose_device(arguments.device)
            model, _ = load_model(arguments.model, config, device, require_qualified=False)
            def inference(location, robot, observation):
                return select_action(model, observation, 0.0, np.random.default_rng(0), device)
        from .records import sha256
        result = audit_archive(arguments.input, inference, sha256(arguments.model) if arguments.model else None)
        # Full directed records can be exported, while terminal output stays compact.
        compact = {k: v for k, v in result.items() if k not in ("pair_consistency", "assignments")}
        if arguments.output:
            from .records import write_json
            write_json(arguments.output, result)
        print(json.dumps(compact, indent=2))
        return 0
    if arguments.command == "allocation-sanity":
        from .validation import run_sanity
        rows = run_sanity(arguments.output)
        print(json.dumps(rows, indent=2))
        return 0
    if arguments.command == "allocation-snapshot":
        from .resource_study import optimize_snapshot
        result = optimize_snapshot(arguments.input, arguments.output)
        print(json.dumps({"output": arguments.output, "absolute_gap": result["absolute_gap"]}, indent=2))
        return 0
    if arguments.command == "resource-conditions":
        from .resource_study import conditions
        config = load_config(arguments.config)
        print("\n".join(name for name, _, _ in conditions(config, not arguments.primary_only)))
        return 0
    if arguments.command == "resource-report":
        from .resource_study import summarize, plot_results
        summarize(arguments.path)
        print("\n".join(plot_results(arguments.path)))
        return 0
    if arguments.command == "deadline-report":
        from .deadline_study import report_study
        rows = report_study(arguments.path)
        print(f"Verified {len(rows)} deadline/treatment/population comparisons in {arguments.path}")
        return 0
    if arguments.command == "serve-replay":
        serve(arguments.path, arguments.port)
        return 0
    if arguments.command == "import-trace":
        print(
            json.dumps(
                import_trace(
                    arguments.input,
                    arguments.output,
                    arguments.direction,
                    arguments.unit,
                    arguments.start,
                    arguments.end,
                ),
                indent=2,
            )
        )
        return 0
    if arguments.command == "report":
        output = (
            report_training(arguments.path)
            if arguments.kind == "training"
            else report_evaluation(arguments.path)
        )
        print(output)
        return 0
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", arguments.run_id):
        raise ValueError(
            "Use a fresh run ID with letters, numbers, underscores or hyphens"
        )
    config = load_config(arguments.config)
    root = Path(arguments.output_root) / arguments.run_id
    if arguments.command == "deadline-study":
        from .deadline_study import run_deadline_study
        if arguments.timeline:
            config = replace(config,communication=replace(config.communication,leo_timeline=arguments.timeline))
        rows = run_deadline_study(config,root,arguments.model,diagnostic=arguments.diagnostic,
            cases=arguments.cases,counts=tuple(arguments.agents),max_steps=arguments.max_steps,
            workers=arguments.workers,device_name=arguments.device,resume=arguments.resume)
        print(f"Measured {len(rows)} deadline/treatment/population comparisons in {root}")
        return 0
    if arguments.command == "resource-study":
        from .resource_study import run_study
        if arguments.timeline:
            config = replace(config, communication=replace(config.communication,
                leo_timeline=str(Path(arguments.timeline).resolve())))
        summary = run_study(config, root, arguments.model, arguments.diagnostic,
            arguments.cases, tuple(arguments.agents), not arguments.primary_only,
            arguments.conditions, arguments.max_steps, arguments.device, arguments.workers, arguments.resume)
        print(f"Measured {len(summary)} resource condition/count summaries in {root}")
        return 0
    if arguments.command in ("train", "compare-training", "study"):
        config = config.profile(arguments.profile)
        config = replace(
            config,
            communication=replace(
                config.communication,
                profile=arguments.communication_profile,
                delay_steps=arguments.delay_steps,
            ),
        )
        if arguments.no_resume and root.exists():
            raise ValueError("Run ID exists; choose a new run ID")
        if arguments.command == "train":
            outcome = train_policy(
                config,
                arguments.condition,
                arguments.seed,
                root / arguments.condition,
                not arguments.no_resume,
                arguments.device,
                arguments.run_id,
            )
            print(
                json.dumps(
                    {
                        "status": outcome.status,
                        "generated_steps": outcome.environment_steps,
                        "episodes": outcome.episodes,
                        "final_model": str(outcome.final_model),
                    },
                    indent=2,
                )
            )
        else:
            outcomes = compare_training(config, arguments.seed, root, arguments.device)
            report_training(root)
            print(
                json.dumps(
                    [
                        {
                            "condition": o.run_directory.name,
                            "status": o.status,
                            "generated_steps": o.environment_steps,
                        }
                        for o in outcomes
                    ],
                    indent=2,
                )
            )
            if (
                arguments.command == "study"
                and outcomes
                and outcomes[0].status == "qualified"
            ):
                evaluate_suite(
                    config,
                    root / "clean/final_model.pt",
                    root / "inference",
                    arguments.device,
                )
                report_evaluation(root / "inference")
        return 0
    model = Path(arguments.model) if arguments.model else root / "clean/final_model.pt"
    if arguments.command == "evaluate":
        if not 1 <= arguments.cases <= 50:
            raise ValueError("Cases must be between 1 and 50")
        if arguments.conditions and set(arguments.conditions) - {
            name for name, _ in treatments(config)
        }:
            raise ValueError("Unknown treatment name")
        summaries = evaluate_suite(
            config,
            model,
            root / "inference",
            arguments.device,
            arguments.cases,
            tuple(arguments.agents),
            arguments.diagnostic,
            arguments.conditions,
        )
        print(
            f"Measured {len(summaries)} condition/count summaries in {root / 'inference'}"
        )
    elif arguments.command == "replay":
        path = create_replay(
            config,
            model,
            root / "replay.html",
            arguments.agents,
            arguments.delay_steps,
            arguments.mission_route,
            arguments.diagnostic,
            arguments.device,
        )
        print(path.resolve())
    return 0
