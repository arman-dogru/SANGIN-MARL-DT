# Movement frequency, command cutoffs and cached coordination

This extension runs frozen PANAMA inference with the existing binary allocator,
compute pools, radio model and modeled-energy definitions. No model updates are
performed. Historical runs and launchers remain unchanged.

## Local audit

The pre-redesign implementation and uncommitted source delta are archived under
`output/redesign/audit-20261026-01`. The cached-fallback experiment stopped at
279/375 completed episodes; its saved records remain unchanged.

| Component | Existing behavior | Extension |
|---|---|---|
| Motion | One grid move after each cutoff | One or two discrete movement rounds per global step |
| Priority | Remaining A* distance, completed robots last | Same implementation, refreshed after physical moves |
| Missed command | WAIT or opt-in delivered cached-path successor | Same cutoff rule, held path instruction and source labels |
| Task deadline | Source-to-first-complete-arrival classification | Same records; separate cycle and command-transport latencies |
| DT decomposition | Historical cloud reference | Explicit command processor reference; UAV ingress when no command exists |
| Latency plots | Delivered-command empirical step CDF | Shared CDF helper plus log CCDF; configurable cutoff markers |
| Energy | Communication plus UAV computation | Same scope; no locomotion, propulsion or extra local-controller energy |

The published successful-mission W1 pooled p95 values are 116.99 ms (8,
allocator), 149.68 ms (16, allocator), 267.92 ms (16, all-UAV), 242.25 ms
(32, allocator), and 341.12 ms (32, all-cloud). The mean of per-episode p95
values is a different estimand: approximately 108.85, 161.18, 267.62, 244.75
and 345.37 ms respectively. The published task table and summary reconcile;
neither distribution is assumed to remain valid at the shorter interval.

## Discrete movement and command scheduling

Slow robots execute the previous step's remote command, normal robots execute
the current step's command, and fast robots have two ordered movement rounds.
Each action permits at most one adjacent cell. Scenario-seeded, balanced class
assignments are independent of policy and treatment. Every round checks all
simultaneous targets and edge swaps using the existing delivered-priority
resolver. Pipeline warm-up, controller WAIT, physical blocking and
unusable cached paths have separate reasons. No movement clock, accumulated
distance or additional time event is introduced.

Reports are generated at `n * interval`; actions execute at
`n * interval + deadline`. Every active robot submits exactly one normal request
per step. Slow robots consume source cycle `n-1` at execution cycle `n`; cycle
zero is a warm-up WAIT. No current-cycle or older command substitutes for that
specific previous command. If it is missing, cached treatments use the accepted
cache and WAIT treatments use WAIT. A slow command arriving after its nominal
delivery deadline may remain applicable until its explicitly scheduled next-step
cutoff. It remains **late** in task-level deadline accounting. Closed execution
epochs cannot be changed by later arrivals.

A fast task calls the unchanged frozen DQN twice at the selected processor.
The first call uses the uploaded observation and delivered DT replica. The
second uses a hypothetical self-position after action one, the static map and
the same delivered DT peer information. DT peers are held at delivered positions
for this one-action prediction. The original A* state builder computes the
hypothetical self path and distance-based priority; this state is never published.
Both actions and their DT priority provenance return in **one response** under
one existing task and deadline. Logical input bits, required CPU cycles and CPU
energy accounting stay unchanged. Host time for both calls is measured; actual
response serialization includes the additional bundle metadata.

Fast robots execute both directions at the shared cutoff, checking collisions
after each cell. A prediction mismatch after the first physical resolution is
logged; it does not trigger another inference request. If the current response
misses its cutoff, each round follows the treatment's cached or WAIT rule. Goal
completion immediately activates the existing completed-robot local controller.

The classes are discrete execution schedules, **not** fixed cells/second. Slow
means one-epoch command lag, not half-rate movement; after warm-up its maximum
throughput equals normal. Fast permits two adjacent moves per step. Changing the
global interval therefore changes nominal movement rates. This protocol replaces
the archived continuous-clock and alternate-step drafts. Neither is part of the
registered A0–A7 runs. There is no bridge treatment.

UAV positioning retains a nominal one-update-per-second cadence and the existing
metres-per-update limit, sampled at global sensing steps using received reports.
Radio/CPU/propagation clocks stay in seconds.

Only epochs with a cutoff at or before 250 seconds generate tasks. The primary
350/250 ms condition has 714 eligible epochs (`n=0..713`). Physical termination
stops the calendar at the exact completion event or at 250 seconds. No new tasks
are generated after termination; outstanding tasks remain censored.

A0 retains the historical execution path for exact reproduction. A1–A7 retain
discrete movement at the cutoff and add only the registered step frequency and
fallback behavior. No timing bridge treatment is added.

## Registered matrix and pairing

| Treatment | Speeds | Interval (ms) | Cutoff (ms) | Miss fallback |
|---|---|---:|---:|---|
| A0 | Uniform | 1000 | 250 | WAIT, historical motion |
| A1 | Uniform | 350 | 250 | WAIT |
| A2 | Heterogeneous | 350 | 250 | WAIT |
| A3 | Uniform | 350 | 250 | Cached path |
| A4 | Heterogeneous | 350 | 250 | Cached path |
| A5 | Heterogeneous | 1000 | 250 | Cached path |
| A6 | Heterogeneous | 350 | 150 | Cached path |
| A7 | Heterogeneous | 350 | 300 | Cached path |

Phase A begins with five original cases at 8/16/32 robots (360 episodes), then
uses the first ten pre-registered cases (720 episodes total). Phase B uses A4,
2/4/8/16/32 robots, ten paired cases and W1–W5 (750 episodes). The 90 matching
A4/W1 episodes are shared, giving 1,380 unique primary jobs. A scenario's seed,
map, starts/goals, orbital offset and speed assignment never depend on outcomes.
The five historical cases per population retain their original identities.
Additional cases follow the original deterministic case generator and offset
rule. The 16/32-robot cohorts remain exploratory scale tests.

W1–W5 retain latency/energy weights 1/1, 1/2, 1/4, 2/1 and 4/1. UAV/cloud CPU
capacities remain 6/10 GHz, tasks remain 100,000 bits at 1,000 cycles/bit, and
UAV kappa remains 1e-28. Hard solver deadlines remain disabled.

## Diagnostics and accounting

Action sources are remote command, cached fallback, safety WAIT and completed
local control. A separate fresh-information execution of the same fallback rule
distinguishes stale-cache disagreement from fallback-versus-DQN disagreement.
These diagnostics never enter actual caches, commands or motion. Diagnostic
records stay outside the causal event chain so their presence cannot change
serialized network reference sizes.

Physical movement and completion reversals are counted separately. Active
unordered-pair exposure is measured in pair-seconds. Receiver-specific stale
ordering durations end when accepted cached ordering matches, at another
reversal, or at the observation horizon; censored durations remain explicit.
The original vertex-plus-edge relevance and ordering equations are retained.
Processor-specific decomposition names the actual reference replica. The
vertex-only metric remains a separate exploratory quantity.

Task exports retain original source, generation, arrival and cutoff fields.
E2E latency is arrival minus sensing; cycle latency is arrival minus cycle start;
command transport is arrival minus command generation. Missing arrivals receive
no imputed latency. Cutoff use is evaluated with cycle-relative arrival.

Conditional tables use successful missions only. Success and failures retain all
attempts. Empty conditional cohorts are undefined. Paired conditional effects use
cases completed in both conditions, with case-level bootstrap 95% intervals.
Exact task, decision and relevant-pair denominators accompany rates. Predicted
path conflicts, attempted movement conflicts and post-resolution overlaps are
distinct. CPU averages include idle mission time. Controller WAIT seconds and
unused movement opportunities are distinct physical measures.

## Frozen execution and gates

`scripts/run_redesign.py` creates a new directory with copied source, model,
timeline, map, exact configurations, pre-registered jobs and hashes. Allocation,
compute, radio, transport, environment, model and command-contract source hashes
must match the audited local revision. The historical harness is not edited.

The workflow requires deterministic tests and full regressions, a short discrete
preflight, exact A0 reproduction, then 12/16/18-worker benchmarks. Benchmarking
uses identical representative jobs and includes causal/model audits and archive
cost. The initial pool size is 16, with at most eight 16/32-robot jobs, four
32-robot jobs and an 8 GiB RAM reserve. Conservative per-job peak reservations
can reduce effective concurrency. Numerical libraries and PyTorch use one CPU
thread per worker; each worker loads the model once.

The configurable initial wall cap is 72 hours across stages. This is a maximum,
not a predicted duration. Measured audited throughput determines the full-run
estimate before primary execution. Resuming preserves the original elapsed
lease. Outputs are atomic, jobs cannot overwrite completed transactions, and
exceptions retain trace/snapshot evidence. No seed substitution or silent
failure omission is allowed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/redesign-tests --junitxml=tmp/redesign-tests.xml
.\.venv\Scripts\python.exe -m pytest tests/test_frequency_motion.py tests/test_bundled_actions.py tests/test_redesign_workflow.py tests/test_controlled_staleness.py tests/test_consistency_vertex.py -q --basetemp=tmp/redesign-focused --junitxml=tmp/redesign-focused.xml
.\.venv\Scripts\python.exe scripts/record_redesign_validation.py --full tmp/redesign-tests.xml --focused tmp/redesign-focused.xml
.\.venv\Scripts\python.exe scripts/run_redesign.py --prepare
.\.venv\Scripts\python.exe scripts/run_redesign.py --launch --stage preflight
.\.venv\Scripts\python.exe scripts/run_redesign.py --launch --resume --stage reference
.\.venv\Scripts\python.exe scripts/run_redesign.py --launch --resume --stage benchmark
.\.venv\Scripts\python.exe scripts/run_redesign.py --launch --resume --stage full
```

## A4-first accelerated dispatch

`scripts/run_redesign_accelerated.py` prioritizes A4/W1 at 8/16/32 robots,
three policies and the first five matched cases (45 episodes). Registered
five-case ablations follow, then the ten-case expansion and remaining weights.
The 1,380 unique primary jobs, source package, configurations, assets, physics,
inference and all episode audits remain unchanged. The remaining 16/18-worker
benchmark repetitions are deferred scheduling diagnostics.

The handoff suspends only the old dispatcher. Its running episode processes
finish and commit atomically before their idle pool is retired. Committed
12-worker benchmark episodes are verified and reused. Source and handoff
identities are recorded in `accelerated/manifest.json` and `handoff.json`;
the original frozen source and benchmark outputs remain intact. Dispatch retains
one numerical thread per worker, at most 16 workers, size-class limits and an
8 GiB RAM reserve. The original elapsed wall-time lease is preserved.

`scripts/checkpoint_redesign.py` incrementally caches newly committed episodes
and publishes immutable compact snapshots under `results/redesign/`. Each
snapshot contains all completed mission outcomes, successful-mission conditional
tables, diagnostic PNG figures and provenance. Raw traces, large logs and
mission-success plots stay local. Git publication stages only the designated
experiment sources and snapshot directory; unrelated staged work and branch
divergence block publication without interrupting simulations.

Thirty-minute scheduled checkpoints invoke the publisher once, then report
episode counts, preliminary findings, failures/errors, measured throughput and
ETA. Computation continues independently between checkpoints. No training is
performed. Historical results are immutable.
