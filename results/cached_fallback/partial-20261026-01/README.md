# Cached fallback — partial results

The experiment stopped before redesign: **279/375** primary episodes completed, with **245** successful and **34** unsuccessful missions. The remaining 96 planned episodes have no completed outcome. Interrupted episodes are not counted as mission failures. No runs were restarted for this export.

The matrix uses W1–W5, Allocated/All UAV/All Cloud, 2/4/8/16/32 robots, five paired cases, 250 ms cutoffs and one-second control. Only the missed-command controller changes to cached-path fallback. The original WAIT results remain unchanged. The 16/32-robot cohorts are exploratory scale tests.

`episodes.csv`, task records, summary metrics and figures contain successful missions only. `all_completed_episodes.csv` retains every completed outcome; `inventory.csv` retains all 375 planned identities. Mission-success rates use all completed attempted missions in each partially observed cell. They do not imply that the full five-case matrix finished. Paired differences use identical completed cases; conditional outcome comparisons require success in both conditions. Missing conditional outcomes are undefined.

The saved completed episodes generated 100130 tasks: 57099 on time, 43031 late and 0 undelivered. Cached fallback was invoked 43031 times and produced 5448 movement steps.

Each saved episode passed causal replay with frozen-policy action verification before archival. This export rechecks every transaction, archive hash, binary placement, command partition and energy identity. The 36-episode validation pilot passed, including nine exact original-WAIT reproductions. The recorded regression suite passed 427 tests; four supplementary-analysis tests also passed.

Energy includes communication and UAV computation only. No robot locomotion or UAV propulsion energy is added. Full event traces, checkpoints and mission-success figures remain local. These partial results are not a completed study.

The included manifest, configurations, frozen source, case bank, packet hashes and archive hashes preserve provenance. The separate 100 ms pilot is documented in [PILOT_REPORT.md](PILOT_REPORT.md).
