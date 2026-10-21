# Manuscript implementation, 2026-10-21

The initial push confirmed `main` was synchronized. Work stays on `main` and is
pushed in validated stages.

Stage 1 adds the manuscript's dynamic drone CPU energy to the conic relaxation,
exact binary enumeration, and finite compute service. Existing configurations
retain kappa=0; `configs/paper_study.toml` uses kappa=1e-28 and omits constant robot
upload energy from the optimization while retaining it in physical accounting.
The square-root CPU formula applies to latency-only allocation; the general
latency-plus-DVFS allocation solves its scalar KKT conditions and may idle CPU.

Six predictable scenarios are checked using independently formulated primal
CVXPY problems. The stage's 34 allocation, compute, mission, sanity, and checkpoint
compatibility checks pass. Checkpoint map compatibility accepts only exact LF/CRLF
conversions, preserving rejection of content changes and all qualification gates.

The user-provided qualified policy and six-hour archived LEO timeline have been
verified and hashed in `assets_manifest.json`. The policy passed the clean
200-case qualification gates at 2/4/6/8 agents. Test checkpoints are not inputs.

Stage 2 records physical, latest generated digital-twin, and receiver-cache
orderings separately, including UL/DL/E2E errors and their common FoV denominator.
It saves policy observations/actions, priorities, input provenance, every snapshot,
executed assignments, service stages, CPU utilization and realized DVFS energy.
Independent episode workers preserve paired seeds and case-specific orbital
offsets; all treatments use the same case/channel realization.

The full regression suite passes 106 checks. A qualified-policy pilot completed
all four resource policies and the fresh reference at 2/4/6/8 agents (20 episodes,
12-step pilot cap). Pilot outputs remain explicitly labelled diagnostic.

`physics_audit.json` verifies path loss, effective -167 dBm/Hz noise, conducted
power versus EIRP, gateway bottlenecks, complete round-trip propagation and UAV
movement. `paper_audit.association_average` derives the paper's mean-geometry
15-second association channels from the archived samples, recomputing rates
rather than averaging rates. Source interruptions and both propagation legs are
retained. The original time-varying archive remains unchanged.

Stage 3 adds the editable manuscript under `paper/`, the unchanged original TeX,
and reproducible input preparation, independent policy replay, diagrams, and a
complete-study-only report publisher. The manuscript compiles with portable
Tectonic after the built-in preview's Windows sandbox helper failed. Generated
results are explicitly marked in progress until all 19,400 episodes finish.

The full grid exposed a backhaul-interruption endpoint and nearly tied static
weights that the smaller pilot did not cover. Their original failed snapshots
are retained as regression fixtures. The conic epigraphs are now dimensionally
scaled, unused-route cones removed, and fixed-x CPU shares recovered by the same
convex subproblem without allocation floors. Numerical retries and tolerances
are recorded; feasibility and objective checks remain at 1e-6. A separate fix
preserves AllocationFailure snapshots across worker-process boundaries. Exact
binary execution kernels are AST-checked against the original run revision on
resume; these numerical changes affect the bound calculation only.

The expanded full suite passes 110 checks. Completed episodes and interrupted
allocation-log tails are verified against their complete compressed traces before
resume. The complete sweep remains active; publication tables are gated on its
completion and exact case/record counts.

Stage 4 completes and verifies the 1,000-episode nominal-resource comparison.
It contains 487,862 generated tasks, 483,862 delivered tasks and 74,092 allocation
snapshots. An independent audit reproduced 1,760 PANAMA decisions in 20 sampled
traces. A separate full-grid repeat matched all 1,000 nominal episodes' physical
metrics. The tracked baseline report includes paired confidence intervals and
does not claim directional mission-success gains where the intervals cross zero.

An architecture audit found that the existing implementation offloads guidance
and coordination support, while the action network consumes local sensing and
received caches at decision epochs. The manuscript now consistently describes
that qualified implementation. A serialized remote action network would require
a separate workload, transport and timing experiment.

The expanded independent replay audit verifies physical accounting for 10,944
saved nominal tasks: s/R, C/f, P times measured airtime, the complete two-way
propagation decomposition, and the sum of observed per-stage delay components.
These checks use saved measurements rather than allocation implementation helpers.

Stage 5 extends paired summaries with same-policy nominal-resource comparisons,
so CPU/radio changes have their own matched-case intervals. Seven resource-study
checks pass, including shuffled/missing case identities and the nominal identity
comparison. The manuscript explicitly documents absolute weight scaling, sticky
first-goal completion priority, and guidance-arrival task latency versus separate
coordination-state delivery. Its equations compile without column overflow.

Exhaustive nominal replay now passes for all 1,000 episodes: 81,914 decisions,
487,862 physical task checks and 74,092 allocation snapshots. The earlier sampled
audit remains preserved alongside `baseline_full_loop_audit.json`. The full
sensitivity sweep continues; weight-ratio identity controls will be checked once
those complete conditions are available.

The final full regression run passes 111 tests. Independent study audits now
reject running/incomplete case banks and record their own source/manifest hashes.
The nominal sample also independently verifies normalized CPU capacities,
individual route/share bounds, binary execution and DCP certification. Its largest
reported solver residual is 1.282e-8 against the 1e-6 acceptance threshold.
