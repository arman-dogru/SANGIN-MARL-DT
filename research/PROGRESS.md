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
