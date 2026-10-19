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
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
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
