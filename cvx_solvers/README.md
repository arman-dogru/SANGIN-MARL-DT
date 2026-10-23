# Supplied priority-offloading CVX solver

The default run uses the **four-robot validation Cases A-F**, saved in
`scenarios/test_priority_offloading_cvx_stable_supplied.m` and translated into
`scenarios/user_validation.json`. All supplied numerical values are retained,
except every deadline is overridden to **1,000,000 seconds** in the large-deadline configuration.
Case E therefore repeats Case C; its original `1.10*solC.T` deadline experiment
is disabled. Case D retains its forced assignment `[1,1,0,0]`.

Both uploaded files are byte-identical (SHA-256
`31a53647c1d737d9a08c6f6e3647f80ed7b3a45870d9f3b3a39e54ea59e7f87b`).
The two unchanged copies are preserved under `originals/attachment_1` and
`originals/attachment_2`. They define one unique solver.

`run_equivalent.py` executes the same mathematical binary model using
CVXPY/Clarabel, enumerating every assignment. This is explicitly **not native
MATLAB/CVX execution** and does not require MATLAB, CVX or MOSEK. Each result is checked against an independent scalar KKT
calculation over every binary assignment. No PANAMA simulation code is changed.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe cvx_solvers\run_equivalent.py
```

To reproduce the **original test-script deadlines** separately:

```powershell
.\.venv\Scripts\python.exe cvx_solvers\run_equivalent.py --tau-mode original
```

This uses infinite tau for A-D and F. Case E uses `1.10*solC.T`, computed from
Case C during the same run, exactly as in the supplied test script. Results go
to `results/user_validation_original_tau`; the earlier large-tau results remain
in `results/user_validation`. JSON encodes infinity as the string `"Infinity"`;
MAT exports retain numeric infinity. Generated reports remain in the ignored local
results directory.

The earlier fixtures are available only with `--benchmarks`.
`parameters.json` specifies those synthetic fixtures using the project's prior settings:
100,000 input bits, 1,000 cycles/bit, 50/10 Mbit/s access/backhaul, 50 ms full-route
round-trip propagation, 1/10 GHz processors, and 2 W drone forwarding power.
Weights range from 4 to 1. The runner checks 2/4/6/8 tasks with `kappaD=0` and
`kappaD=1e-28`. These are declared fixtures, not measurements or trained priorities.
These fixtures use the large-deadline configuration `tau=1000000` seconds.

The upload differs from the earlier project objective by optionally adding drone
CPU energy `kappaD*C*fD^2`. Its energy objective includes that term and forwarding
energy, but excludes the constant robot-upload energy. Results preserve this exact
objective; the omitted robot-upload constant is reported separately. The original
positive active CPU floors and inactive-denominator epsilon are retained.

Results for the supplied scenarios are local under `results/user_validation`:
`summary.csv`, `summary.json`, `summary.md`, `per_task.csv`, individual
input/solution JSON and MAT files, every assignment's solver status and residual,
and a manifest identifying both original hashes and the numerical backend.
Earlier synthetic fixture results remain separate under `results/cvxpy`.
Generated results and result summaries are excluded from version control.

To execute the actual `.m` files later, configure MATLAB, CVX and its integer-capable
MOSEK solver, then run from MATLAB:

```matlab
addpath('D:/Projects/SANGIN-dev/cvx_solvers');
run_solvers;
% Original test-script deadlines:
run_solvers('D:/Projects/SANGIN-dev/cvx_solvers/results/matlab_original_tau', 'original');
```

The native runner uses the same JSON cases with large tau, invokes both original
attachments separately, and saves their MAT files and logs under `results/matlab`.
It has not been executed here.
