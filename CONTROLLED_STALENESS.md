# Controlled peer staleness and coordination density

This is a bounded stress experiment, not a measurement of nominal SAGIN delay.
PANAMA is frozen. Allocation, CPU capacities, task workload, radio scheduling,
deadline accounting and physical movement/resolution rules are reused unchanged.
The existing 1,200-episode evaluation and its outputs are not inputs to this run
and are never rewritten.

## Intervention and controls

The intervention buffers the accepted UAV/cloud replica history **at the peer
read used by remote PANAMA inference**, after computation. A delay of `d` steps
selects the latest accepted peer state whose arrival time is at most
`inference_time - d * 1 s`, for `d = 0,1,2,3,4,5`. It adds a controlled availability
hold after network delivery, not an exact task-source age. Source and generation
timestamps, report IDs, versions and arrival events are preserved. Source AoI is
measured separately and includes natural network delay.

The t=0 briefing remains available during warm-up. No negative-time history or
future state is fabricated. Own uploaded sensing, obstacles, physical occupancy,
own computed path and vector remain unchanged. Only the atomic peer state
(including intentions, priority and completion) used by inference is buffered.
Allocation continues to read the original delivered replica, and receiver caches
and guidance streams are not directly delayed. Once trajectories diverge,
downstream allocations and traffic can legitimately diverge. Command payload
serialization remains physical; changed state references can change wire length.

Zero delay returns the original replica. Diagnostic events use a separate
checksum chain outside the causal event stream, so they cannot shift state-event
IDs or change response wire sizes. Tests require exact zero-delay causal event,
action, motion and arrival equivalence with the existing system.

Three existing treatment configurations are reused:

- `fresh_transport`: the existing ideal transport/compute reference. At zero
  injected delay it is fully fresh. At positive delay its peer reads are still
  buffered, providing a pure controlled-staleness reference; it is not labelled
  fresh information at those delays.
- `equal`: physical transport with matched active-mean equal resource weights.
- `priority`: the same physical transport with current PANAMA resource weights.

All use 250 ms decision/command/task cutoffs, a 1 s control interval and no hard
optimizer deadline. Paired conditions share complete maps, starts/goals, seeds,
orbital offsets and checkpoint bytes. Treatment changes retain the established
ideal-reference exception; physical equal/priority radio parameters are identical.

## Coordination scenarios and scope

`controlled_scenarios.py` defines three 32 by 32 layouts:

- A three-cell-wide intersection with perpendicular and opposing streams.
- Opposing traffic through a one-cell corridor, with two passing bays and open
  rooms at both ends. The topology permits yielding; the unchanged controller may
  still deadlock.
- Two rooms connected by a single shared door, with merging and opposing traffic.

Counts 2/4/6/8 select deterministic endpoint slots. Cases rotate the complete map
and endpoints together; RNG seeds and orbital offsets are saved. These cases
are controlled geometry/role conditions, not independent random maps. Robot count
is a density control, while measured directed-pair exposure is the actual
coordination-demand measure. All endpoints are unique and individually reachable.

The original checkpoint is checked for qualification on its original map. These
new stress maps remain out of distribution and are **not** claimed qualified.
Poor fresh-reference performance limits conclusions about networking. There is
no retraining or change to model inputs or movement.

## Diagnostics, denominators and inference

Every completed inference saves the actual observation and two frozen shadow
actions at the same compute instant, using exactly the same own sensing:

1. Unbuffered delivered peers: isolates the additional injected hold.
2. Ideal current physical peers: measures total stale-peer effects, including
   natural network staleness. This is not a counterfactual mission rollout.

The existing fresh cutoff oracle is retained. Used-command disagreement counts
exclude cutoff-induced WAITs; the active-decision population is partitioned into
used commands and missing-command WAITs. On-time, late, undelivered, censored and
pending task accounting remains the existing sensing-to-command definition.
Thus compute-time stale-input disagreement is distinct from total cutoff-oracle
disagreement, which can also reflect late commands or stale self sensing.

Established vertex-plus-edge FoV ordering equations are evaluated on the
inference view as additional prefixed metrics. Original decision/cache metrics
are retained. Both cached-view relevance and physical-reference relevance are
reported, since stale paths can hide conflicts. Rates retain numerator and
denominator counts; zero exposure means missing, not perfect consistency.
AoI is saved for all peers and physically relevant peers. Vertex-only metrics
remain separately labelled exploratory. Attempted physical vertex conflicts and
edge swaps are recorded before the unchanged delivered-priority resolver.

Exposure includes directed-pair counts, distinct directed robot pairs, exposed
cycles and exposed cycles with movement. Repeated deadlock epochs are not
independent conflict opportunities. Before pilot inspection, the screening rule
is at least 20 physical-reference relevant directed-pair observations, exposure
in both cases, and some progressing exposed cycles for each scenario/density at
zero-delay fresh transport. This screens scenario usefulness, not statistical power.
The field `progressing_exposure_cycles` counts exposed cycles with any physical
movement, including oscillation; it does not establish progress toward goals.

`paired_differences.csv` reports within-treatment delay versus zero, current
priority versus matched equal at each delay, and a priority-mitigation interaction:
`[priority(d)-priority(0)] - [equal(d)-equal(0)]`. Mission contrasts use paired case
means; rate contrasts pool numerator/denominator counts and bootstrap whole
paired cases. AoI contrasts use case means; summary AoI also reports the pooled
peer-observation mean. Interaction rate contrasts use means of each case's changes, with
unsupported cases explicitly excluded. All intervals are pointwise 95% percentile
intervals (2,000 samples, seed 42), with no multiplicity correction. A two-case
pilot cannot provide reliable population inference; degenerate intervals are not
evidence of equivalence. All three treatments and all selected delays are saved.

## Bounded pilot commands

```powershell
.venv\Scripts\python.exe -m panama_uav --config configs/paper_remote_dt_250ms.toml stress-study --run-id peer-staleness-pilot-01 --model artifacts/inputs/final_model.pt --cases 2 --agents 2 4 --delays 0 1 2 3 4 5 --max-steps 60 --workers 2
.venv\Scripts\python.exe -m panama_uav stress-report --path output/controlled_stress/peer-staleness-pilot-01
```

This pilot is 216 episodes: three layouts, two densities, two cases, six delays
and three treatments. The command rejects more than two cases or more than
80 steps; it has no full-study override. Existing output directories are rejected.
Failed solver/audit snapshots are retained and the pilot stops without policy
substitution. Review exposure and fresh performance before defining a full study.

Outputs stay under ignored `output/controlled_stress/`: the manifest includes
source/model/timeline hashes and complete configuration, and the paired bank
includes per-case map hashes. Compressed archives retain inference observations,
raw intentions/timestamps, tasks, allocations, radio events, actions and oracle
diagnostics. `audits.json` records replay and model verification. `episodes.csv`,
`summary.csv`, `paired_differences.csv`, `validation.json` and comparison plots
support review. Post-processing checks archive hashes, pairing and completion.
The test suite covers all six delay values, inclusive availability boundaries,
rollback, provenance, zero equivalence, diagnostic tampering, unchanged self
inputs, reproducible geometry, isolated treatment settings and the full-run gate.
