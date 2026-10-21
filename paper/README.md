# Revised manuscript

`main.tex` edits the user-supplied manuscript in a dedicated local LaTeX project.
The original source is retained under `original/main_user.tex`. Open this folder
in the Codex LaTeX editor for source editing and a live PDF preview.

The Simulations and Results section is `sections/simulations.tex`. It states
channel/RF parameters, finite CPU service, dynamic energy, movement, full-route
propagation, control timing, censored metrics and the paired evaluation protocol.
`sections/solver.tex` separates the convex scheduling bound from binary execution
and distinguishes the latency-only square-root CPU rule from the DVFS KKT rule.

The atmospheric reference identifies P.618-13 because that is the revision in the
archived ITU-Rpy 0.4.0 experiment. It does not claim the archive was produced with
the newer P.618-14 recommendation. The 2016/2019 placement references and PANAMA
are linked to author preprints. The nominal kappa is an explicit assumption;
the missing original `li2019energy` bibliography entry is not guessed.

Figures and result tables are generated from source scripts and recorded runs.
The complete run uses 97 conditions, 50 held-out cases per population and 250 steps.
The completed 1,000-episode nominal comparison is already included, with paired
confidence intervals and an independent repeat audit. The generated subsection
explicitly distinguishes those results from the running full sensitivity grid.

The action network uses local sensing and received caches; offloaded finite-CPU
jobs provide guidance and coordination states. The manuscript consistently scopes
its claims to this qualified implementation. Remote action inference is a separate
extension requiring its own workload and transport timing.

Reproduction commands and asset hashes are maintained in `../research/`.
