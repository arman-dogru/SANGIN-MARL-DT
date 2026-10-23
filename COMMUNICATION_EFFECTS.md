# Frozen PANAMA inference over SAGIN

`scripts/run_communication_effects.py` compares the production allocator, all-UAV
execution, all-cloud execution through the archived LEO route, and instant communication. The
policy, starts/goals, seeds, map, control interval, movement rules, CPU capacities,
task cycles and radio budgets are identical across paired conditions. No learning
updates occur. The allocated arm uses the unchanged production offloading solver.
All-UAV/all-cloud placement is fixed; the existing binary CPU-allocation reference
and finite priority-sharing pools are reused.

## Conditions

The default study uses 10 cases at each of 2/4/6/8 robots, a 250-step episode cap,
a one-second control interval and a 250 ms sensing-to-command cutoff. The optimizer
has no hard deadline constraint. Forty-one conditions yield 1,640 simulations.
Commands available by the cutoff are used at the same control tick. Latency
changes can therefore affect mission timing through cutoff WAITs or changed
observations/actions, while sub-cutoff latency changes can leave movement unchanged.
Three matched ideal-network references retain the finite UAV/cloud CPU stages;
their contrasts isolate communication costs. The instant ideal reference also
removes modeled compute service and uses the existing fresh-state reference.
Matched controls retain normal causal DT updates.

| Control | Additional levels | Applied to |
| --- | --- | --- |
| Access uplink transit | 50, 150 ms | Allocated, all-UAV and all-cloud |
| Access downlink transit | 50, 150 ms | Allocated, all-UAV and all-cloud |
| Peer-state availability hold | 1, 3 control steps | Actual remote inference inputs |
| Inference service overhead | 50, 150 ms | After the unchanged finite CPU stage |
| Communication interruption | 250, 1,000 ms per 10 s | Access and backhaul service |
| LEO transit | 50, 150 ms per direction | Cloud-capable arms |

Every sweep changes one factor from its route's nominal configuration. Additional
transit uses the existing event-calendar processing hops. Interruptions suspend
and resume transmission; overlapping native/injected backhaul windows are unioned.
RF energy accrues during active transmission. The original orbital geometry,
RF limits, queues, frame guards and propagation remain in effect. These injected
values are experimental controls, not measured network conditions.
The allocator continues to use its native radio/propagation snapshot; injected
transit and inference overhead are applied to execution, without retuning the
allocation algorithm or adding hard solver deadlines.

The input remains 100,000 logical bits and 1,000 cycles/bit. UAV/cloud capacities
remain 1e9/1e10 cycles/s; robot/UAV uplink conducted powers remain 0.2/2 W.
The base configuration retains its antenna/EIRP, energy and propagation assumptions.
Logical bits, serialized payloads and wire overhead are recorded separately.

Nominal parameters come from [`paper_remote_dt_250ms.toml`](configs/paper_remote_dt_250ms.toml):

| Parameter | Value |
| --- | --- |
| Access uplink/downlink capacity caps | 50/50 Mbps |
| LEO uplink/downlink capacity caps | 10/50 Mbps |
| UAV backhaul EIRP | 35 dBW; distinct from 2 W conducted power |
| UAV dynamic-compute coefficient | 1e-28 |
| Resource-weight range | 1–4, current delivered DT priorities |
| Energy/latency objective coefficients | 1/1 |

Rate bounds and propagation use the archived timeline and existing link budgets.
CPU service times use the configured logical cycles and finite processor pools;
they are separate from host wall time spent evaluating the frozen DQN.

## Execution

Provide an archived qualified checkpoint and orbital timeline through the base
configuration. Preparation copies inputs and source into a new output directory.
Existing output directories cannot be prepared again.

```powershell
.venv\Scripts\python.exe scripts/run_communication_effects.py --root output/communication_effects/inference-01 --model PATH_TO_CHECKPOINT --prepare --cases 10 --counts 2 4 6 8 --cap 250 --workers 4
.venv\Scripts\python.exe scripts/run_communication_effects.py --root output/communication_effects/inference-01 --launch
```

The worker imports its frozen `code/` tree and performs greedy CPU inference with
one thread per worker. Checkpoint/source/map/timeline/case hashes and package
versions are saved in `manifest.json`. Accepted per-case transactions support
recovery of interrupted aggregate writes without repeating completed simulations.
The default runtime ceiling is four hours. Progress is stored in `status.json`;
the workflow creates no recurring monitoring task.

## Measurements and artifacts

Timing records distinguish sensing-to-inference-ready, per-control-round inference
completion, command arrival, ingress, CPU service, additional inference service,
egress, queuing and interruptions. Command arrivals are partitioned into on-time,
late and undelivered, with separate censoring and cutoff-WAIT tags. Source and
generation timestamps are retained through atomic DT updates and command delivery.

Priority measurements include directed FoV exposures/errors, peer AoI, cached/DT/
physical ordering and stale-versus-fresh action disagreements. CPU utilization is
integrated from actual allocation events. Energy covers robot access uplink,
UAV backhaul uplink and configured UAV CPU energy. Cloud-server, receiver and
propulsion power are not parameterized.
Peer AoI includes retained completed-robot states after reporting stops. Source
timestamps and raw peer states support a separate active/completed breakdown.

- `paired_cases.json`, `manifest.json`, `assets/`, `code/`: reproducibility inputs.
- `episodes.csv`, `tasks.csv`, `rounds.csv`: mission, task and inference-round telemetry.
- `summary.csv`, `paired_comparisons.csv`: comparisons with case-paired 95% intervals
  and explicit sample/exposure counts. Absolute success intervals use Wilson bounds.
  Paired intervals use 2,000 case-cluster bootstrap resamples with seed 20261023.
- `primary_summary.csv`, `primary_paired_comparisons.csv`: the four primary arms
  and all six pairwise contrasts, by robot population.
- `episodes/`: compressed event/observation archives and audited per-case transactions.
- `validation.json`, `audits.jsonl`: causal replay, pairing and integrity checks.
- `figures/`: vector PDF/SVG and 600 dpi PNG comparisons.
  `network_penalties` shows each nominal arm against its finite-CPU ideal-network
  reference with paired 95% intervals.
- `REPORT.md`: numerical overview.

Reporting verifies all frozen inputs and complete paired episode coverage before
producing tables or figures. Empty exposure denominators remain unavailable.
Rebuild reports using the run's saved `code/` tree and recorded package versions.
Historical experiments, checkpoints and published bundles are not edited.

## Published results

The completed 1,640-episode study is in
[`results/communication/`](results/communication/README.md). It includes the
four primary arms, all controlled sensitivities, paired confidence intervals,
seven figure sets and complete compressed task/round tables. The report retains
negative and mixed findings, including the all-UAV CPU-service limit and
allocation sensitivity to the phase of brief orbital interruptions.

Publication copies preserve original numerical values.
[`results/files.csv`](results/files.csv) records source and published hashes;
`bundle_validation.json` verifies compressed bytes and record counts. Model
files, archived orbital inputs and raw replay archives remain local.
