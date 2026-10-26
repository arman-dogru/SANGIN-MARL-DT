# Cached-path missed-command fallback

The preserved `cached-20261026-01` study stopped at 279 of 375 completed primary
episodes before a separate experimental redesign. Its [partial results](results/cached_fallback/partial-20261026-01/README.md)
retain the completed outcomes, full planned inventory and validation evidence.
The remaining 96 episodes have no completed outcome; the study is not complete.
No automatic resumption is configured.

The remote-DT controller supports `missed_command_fallback = "cached_astar"`.
The default remains `"wait"`. The setting affects active robots only when no
current-cycle command arrives by the configured cutoff. Delivered commands,
completed-robot control and physical conflict resolution retain their existing
behavior.

The fallback follows the next adjacent cell on the robot's delivered cached A*
path, aligned to its locally known current position. It uses cached priorities
and the existing communicated peer-intention horizon to yield to higher-priority
active paths inside the square FoV. Locally visible obstacles and occupancy can
prevent movement. Missing, exhausted or incompatible paths cause WAIT. There is
no local DQN inference, path recomputation, DT refresh, priority-truth lookup or
replay of an obsolete remote command.

Source timestamps and cache ordering remain unchanged. The event trace records
the cache-source times, selected successor, reason, cached priority and local
sensing time. Replay reconstructs the decision from delivered caches and local
FoV sensing. Cached movement does not count as command delivery or deadline
compliance. Misses, requested WAITs, fallback move intents, physical fallback
movement and resolver blocks are recorded separately.

## Paired experiment

The primary matrix uses 2/4/8/16/32 robots, Allocated/All UAV/All Cloud, W1–W5,
five original cases per population and the existing 250-step cap: 375 episodes.
The original 375 WAIT runs remain unchanged. Maps, seeds, starts/goals, orbital
offsets, frozen checkpoint, CPU, radio, task workload and objective coefficients
match the original weight-sensitivity experiment. The primary cutoff remains
250 ms and the control interval remains one second. Hard solver deadlines stay
disabled. Robot movement energy remains excluded from the original communication
plus UAV-computation model; no added local-controller energy term is introduced.
The local rule is evaluated at the cutoff without an added modeled service
delay. It is a fallback execution assumption, rather than a local inference or
hardware power model.

Validation uses one paired case at 2/4/8 robots and all three routes under W1.
WAIT and cached fallback both run at 250 ms and 100 ms, giving 36 pilot episodes.
Nine 250 ms WAIT runs must reproduce the original executed records and solver
certificates. Nine 250 ms cached cases are reused in the primary matrix. The
100 ms conditions remain pilot sensitivity evidence. AoI is measured separately
from command-deadline compliance.

```powershell
.\.venv\Scripts\python.exe scripts/run_cached_fallback.py `
  --root output/cached_fallback/<new-id> --prepare --launch --stage pilot
.\.venv\Scripts\python.exe scripts/run_cached_fallback.py `
  --root output/cached_fallback/<id> --launch --stage full --resume
```

The full stage requires a passing pilot gate. A fixed 24-hour wall budget covers
pilot, simulations, replay audits and report generation; resumptions retain the
original start time. Worker source, assets, configurations, baseline index and
case identities are hashed before launch. Failed simulations retain errors and
solver snapshots and stop the condition. Existing results cannot be overwritten.

Outcome tables and plots use completed missions, with per-cell counts. Paired
outcome comparisons use cases completed by both fallback settings. Mission
success uses all attempted cases. Mission-success figures stay local. New AoI,
ordering, physical-conflict, latency and energy plots are generated from the new
runs. Vertex-only consistency remains separate and exploratory. The 16/32-robot
policy results remain exploratory scale tests.

After completion, the independent audit checks every primary packet/archive
hash and replays 30 stratified primary traces plus all 18 sensitivity traces.
The supplementary analysis verifies that all 375 initial-round allocator input
and binary assignment sets match the preserved WAIT reference, before fallback
motion can change the state. Paired success differences use all attempts; paired
AoI, inconsistency, conflict, latency and energy effects use common completions
with their original exposure denominators.

```powershell
.\.venv\Scripts\python.exe scripts/analyse_cached_fallback.py <run-root>
.\.venv\Scripts\python.exe scripts/audit_cached_fallback.py <run-root> --read-back
```
