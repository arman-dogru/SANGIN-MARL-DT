"""Transport-only workload experiment and empirical packet-delay plots.

This does not claim policy success or learning improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.models import DQN, cpu_state_dict
from panama_uav.records import append_jsonl, export_csv, sha256, write_json
from panama_uav.transport import DelayTrace, Transport, encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", default="data/leo/tremblant_6h.npz")
    parser.add_argument("--output", default="output/leo/tremblant-data-20261005-01")
    parser.add_argument("--config", default="configs/base.toml")
    args = parser.parse_args()
    output = PROJECT / args.output
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(PROJECT / args.config)
    timeline_path = (PROJECT / args.timeline).resolve()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    weights = {k: v.numpy() for k, v in cpu_state_dict(DQN((3, 15, 15), 12)).items()}
    model_wire = encode({"weights": weights, "stage": 0})
    rows = []
    # Matched capacities in both conditions isolate the incremental handover effect.
    for agents in (2, 4, 6, 8):
        env = ExperimentEnvironment(
            config, ExperimentMode.UAV, config.curriculum[0].map_path, agents, 250, 42
        )
        obs, _ = env.reset()
        for handovers in (False, True):
            comm = replace(
                config.communication,
                profile="physical",
                contention_enabled=False,
                leo_timeline=str(timeline_path),
                handover_enabled=handovers,
                backhaul_time_offset_seconds=1000,
            )
            transport = Transport(comm)
            received = []

            def receiver(packet, received=received, transport=transport):
                if packet.kind == "model_relay":
                    for receiver_agent in range(agents):
                        transport.send(receiver_agent,"model",wire=packet.wire,version=packet.version,
                                       generated_at=packet.generated_at,route=("access_down",),delayed=False)
                received.append(
                    {
                        "agent": packet.agent,
                        "kind": packet.kind,
                        "age_seconds": transport.now - packet.generated_at,
                        "version": packet.version,
                    }
                )

            for step in range(1, 251):
                transport.advance(float(step), receiver)
                for agent in range(agents):
                    # Retained immutable tuple encoding and observation shapes.
                    transition = (obs[agent], 4, -0.01, obs[agent], False)
                    transport.send(
                        agent,
                        "experience",
                        {
                            "transition": transition,
                            "stage": 0,
                            "joint_step": step,
                            "episode": 1,
                            "deployed_version": 0,
                        },
                        route=("access_up", "satellite_up"),
                    )
                    if step % 16 == 0 and config.communication.model_distribution == "per_agent":
                        transport.send(
                            agent,
                            "model",
                            wire=model_wire,
                            version=step // 16,
                            route=("satellite_down", "access_down"),
                        )
                if step%16 == 0 and config.communication.model_distribution == "uav_fanout":
                    transport.send(-1,"model_relay",wire=model_wire,version=step//16,
                                   route=("satellite_down",))
                transport.advance(float(step), receiver)
                append_jsonl(
                    output / "queue_timelines.jsonl",
                    {
                        "num_agents": agents,
                        "handovers": handovers,
                        "step": step,
                        "queue_bytes": transport.occupancy(),
                    },
                )
            transport.advance(280, receiver)
            if (
                transport.counters["messages_generated"]
                != transport.counters["messages_delivered"]
            ):
                raise ValueError("Transport benchmark did not drain all messages")
            for kind in ("experience", "model"):
                ages = np.array(
                    [r["age_seconds"] for r in received if r["kind"] == kind]
                )
                row = {
                    "num_agents": agents,
                    "handovers": handovers,
                    "kind": kind,
                    "count": len(ages),
                    "mean_age_seconds": float(ages.mean()),
                    "p95_age_seconds": float(np.percentile(ages, 95)),
                    "max_age_seconds": float(ages.max()),
                }
                rows.append(row)
                append_jsonl(output / "summary.jsonl", row)
            for event in transport.log:
                append_jsonl(
                    output / "events.jsonl",
                    {"num_agents": agents, "handovers": handovers, **event},
                )
    write_json(
        output / "manifest.json",
        {
            "purpose": "transport-only workload simulation",
            "mission_success_measured": False,
            "source_timeline_sha256": sha256(timeline_path),
            "communication": {**config.to_dict()["communication"], "leo_timeline": str(timeline_path)},
            "modeled_model_wire_bytes": len(model_wire),
            "initial_model_seed": 42,
            "joint_steps": 250,
            "model_publication_interval": 16,
            "satellite_handshake_comparison": "same visibility, propagation, and capacities; handshake enabled/disabled",
            "access_capacity_mbps": 50,
            "satellite_uplink_mbps": 10,
            "satellite_downlink_mbps": 50,
            "satellite_model_transfer": config.communication.model_distribution+", store and forward",
            "model_weights_are_fixed_for_transport_workload": True,
            "event_backend": "SimPy 4.1.1",
            "scheduling": "per-agent FIFO, round-robin service chunks and satellite frame guards",
        },
    )
    export_csv(output / "summary.jsonl", output / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, kind in zip(axes, ("experience", "model")):
        for enabled, label in (
            (False, "Same geometry, no handshake interruption"),
            (True, "Orbit-driven handshakes"),
        ):
            chosen = [
                r for r in rows if r["kind"] == kind and r["handovers"] == enabled
            ]
            ax.plot(
                [r["num_agents"] for r in chosen],
                [r["mean_age_seconds"] for r in chosen],
                "o-",
                label=label,
            )
        ax.set(
            xlabel="Agents",
            ylabel="Mean delivery age (seconds)",
            title=kind.capitalize(),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "delivery_age.png", dpi=180)
    plt.close(fig)
    with np.load(timeline_path, allow_pickle=False) as data:
        t = data["times"] / 60
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 7), sharex=True, layout="constrained"
        )
        axes[0].plot(t, data["visible_satellites"])
        axes[0].set_ylabel("Visible satellites")
        axes[1].plot(t, data["uplink_doppler_hz"] / 1e3, label="Uplink, 14.25 GHz")
        axes[1].plot(
            t, data["downlink_doppler_hz"] / 1e3, label="Downlink, 11.7 GHz", alpha=0.7
        )
        axes[1].set_ylabel("Raw Doppler (kHz)")
        axes[1].legend(fontsize=8)
        axes[2].plot(t, data["propagation_seconds"] * 1e3)
        axes[2].set(
            xlabel="Archived simulation time (minutes)", ylabel="One-way delay (ms)"
        )
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.savefig(output / "orbital_geometry.png", dpi=180)
        plt.close(fig)
    trace_summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for direction in ("uplink", "downlink"):
        trace = DelayTrace.load(
            PROJECT / "data" / "leo" / "starlink_owd" / f"{direction}.npz"
        )
        milliseconds = trace.delays * 1e3
        # The ECDF includes every measured packet; timeline is thinned only for rendering.
        axes[0].plot(
            trace.timestamps[::100], milliseconds[::100], label=direction, alpha=0.7
        )
        ordered = np.sort(milliseconds)
        axes[1].plot(
            ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=direction
        )
        trace_summary[direction] = {
            "samples": len(milliseconds),
            "duration_seconds": float(trace.timestamps[-1]),
            "mean_ms": float(milliseconds.mean()),
            "p95_ms": float(np.percentile(milliseconds, 95)),
            "min_ms": float(milliseconds.min()),
            "max_ms": float(milliseconds.max()),
            "source_sha256": trace.metadata["source_sha256"],
        }
    axes[0].set(
        xlabel="Measured recording time (seconds)",
        ylabel="One-way delay (ms)",
        title="Karlstad packet measurements",
    )
    axes[1].set(
        xlabel="One-way delay (ms)",
        ylabel="Cumulative fraction",
        title="All measured packets",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "measured_one_way_delay.png", dpi=180)
    plt.close(fig)
    write_json(output / "empirical_delay_summary.json", trace_summary)
    print(json.dumps({"transport": rows, "empirical": trace_summary}, indent=2))


if __name__ == "__main__":
    main()
