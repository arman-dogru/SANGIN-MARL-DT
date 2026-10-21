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

User-provided qualified policy and six-hour archived LEO timeline will be hashed
and used for the complete paired mission study; test checkpoints are not inputs.
