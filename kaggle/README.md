# Kaggle CPU jobs (rs-sangin-marl-dt)

CPU only (enable_gpu=false, no accelerator), at most 2 sessions at once, each under 12 h (runner guard 11 h default, 11.5 h max).
Only kernels named `rs-sangin-marl-dt[-<suffix>]` are ever pushed. Every run needs Arman's approval (`push --approved`).

- `run_template.py`: clones this public repo, installs CPU torch + `.[leo,optimization]`, recreates the box paths
  (`/workspace/leo_study`, `/workspace/panama_models`) with symlinks, runs the job's commands, writes everything to `/kaggle/working`
  (`manifest.json`, `logs/`, `mem.log`, `runs/main|prio/<job>/result.json`).
- `kaggle_job.py build|push|status|pull` (box copy lives in `/workspace/kaggle/`).
- `omega_missing.txt`: the 77 omega episodes (weight, condition, case) missing from the published 298.
