# LEO / delayed-DT study for PANAMA (box-local, not pushed)

Question: How do communication delays and stale digital-twin information affect PANAMA's
priority-based multi-robot coordination, and can cached fallback reduce their impact?

Everything lives under `/workspace/leo_study/`. The original repo `/workspace/SANGIN-dev`
is untouched (`git status` clean). The patched copy is `repo/`, a plain copy with its own
local git (`git -C repo log -p` shows the diff against the baseline copy).

## Inputs
- `inputs/tremblant_full.npz`: LEO timeline, sha256 `c24a6624…187f`. Matches `timeline_sha256` in `results/communication/manifest.json`.
- `inputs/paper.tex`: paper draft.
- Model: `/workspace/panama_models/model8.pt` (clean-16-01), sha256 `0e999c44…`, loaded in diagnostic mode (`require_qualified=False`).
- Config: `repo/configs/paper_remote_dt_250ms_uav6_cached.toml` (6 GHz UAV, 10 GHz cloud, W1 latency:energy 1:1), with the timeline above.
  Treatments come from `redesign.TREATMENTS`: 350 ms decision interval, 250 ms cutoff. A1 = same speed + WAIT, A2 = mixed speed + WAIT,
  A3 = same speed + cached fallback, A4 = mixed speed + cached fallback.
- Cases: `resource_study.study_cases`, seed = 50,000,000 + N·10,000 + case, on the fixed 32×32 map. Orbital offset is set per case.
  Speed classes are seeded by case seed (classes unchanged from the repo); slow-robot phase is seeded as below.

## Code changes (repo/, see `git -C repo diff HEAD~2`)
1. **Slow robots act every other step** (`frequency_motion.py`). `decisions_per_step=0.5`, `command_lag_steps=0`, and no delayed inbox.
   A slow robot is active when `(cycle + phase) % 2 == 0`. **Phase choice:** `phase ~ Bernoulli(1/2)`, drawn per robot id from
   `np.random.default_rng([case_seed, 918274])`, so slow robots are staggered and reproducible. The class permutation still uses the
   repo's original RNG draw, so class membership is unchanged.
   - On an active step the robot uploads at nΔ and executes the **current** command at nΔ+cutoff.
   - On an off step it does not upload (`remote.update` uses the `_scheduled` hook), takes no decision and gets no DQN shadow. Its
     motion reason is `speed_limited` and it counts in `speed_limited_nonmovement`, not as a WAIT, fallback or cutoff miss.
     It is excluded from `waiting_steps`.
   - Its DT / peer caches age naturally on off steps.
   - Exception: the t=0 initial briefing uploads every robot, so phase-1 slow robots get one upload at cycle 0 that they don't act on.
   - Fast robots are unchanged (two DT actions per step from one request).
2. **Consistency metrics only for receivers that decide this cycle** (`redesign.py::_consistency`, `consistency.summarize_pairs`, factored out
   of `consistency_snapshot` unchanged). Off-step slow robots apply no ordering, so they are not receivers at that epoch.
3. **Per-pair statistics** for figs 1 and 4 (`redesign._pair_stat_keys`, `trace['pair_stats']`):
   - AoI histogram (0.05 s bins of max(self, peer) source-time AoI) × active × FoV-relevant × E2E/UL/DL error.
   - Conflict-detection breakdown (active, cached-predicted, physical, position/path staleness, FoV).
4. Causal audit updated for the new protocol (`redesign_audit.py`, `remote_audit.py`). The audit passed on a real 8-agent heterogeneous episode.
5. `LEO_STUDY_LEAN=1` (set by the harness when no trace is kept) drops audit-only observation payloads from oracle and inference events.
   A re-run of the same job gave a bit-identical row, task list and pair stats.

Tests (patched copy): `475 passed, 2 skipped`. The original suite was 474/2 and one new test was added. Tests that encoded the old
one-step-lag semantics were rewritten (`tests/test_frequency_motion.py`, `tests/test_bundled_actions.py`). Log: `logs/full_suite.log`.

## Harness
- `harness/run_job.py`: one episode → `runs/main/<job>/result.json` (atomic write, resumable).
- `harness/dispatch.py`: main matrix. `harness/dispatch_extra.py`: add-on A3 × edge/cloud (`routes_speed`).
- `runs/trim.json`: budget/memory trims. Trimmed jobs have `trimmed.txt`.
- `harness/analyze.py`: figures and CSVs in `figures/`.
- Perfect-comms baseline: `network=ideal` (`InferenceCondition(..., ideal_communication=True)`): transport instantaneous, CPU stage finite.
- **Delay axis:** added **downlink** delay (`InferenceCondition.downlink_seconds`), inserted on the final `access_down` (UAV→robot) hop of
  every DT→robot message, i.e. both commands and peer-state broadcasts, on all routes. Levels are 0 (nominal LEO), +150 ms and +300 ms.

## Definition fixes / notes
- **missed_physical_conflicts.** The repo counts *directed pair-decisions*, every cycle, where the physical A* intentions conflict inside
  the FoV but the cached intentions do not. These include pairs where a robot has *completed*: completed robots stop uploading
  (`completion_sent`), so their cache stays frozen at the pre-completion state. In a failed 32-robot episode (n32 c00), 976 of 988
  "missed conflicts" came from such pairs, re-counted on each of the 714 cycles while the swarm was stuck. That explains the 1,350
  from the scoping run.
  **Fix used in fig 4:** active pairs only (both robots not completed), normalised per 1,000 active pair-decisions, and shown next to
  the actual attempted movement conflicts (vertex + edge swap, from the motion log). The original and corrected counts are both in
  `fig4_conflict_detection.csv`.
- **AoI reference.** The config keeps `age_reference_seconds = 1.0`, but the paper defines A_ref = Δ = 0.35 s here. With A_ref = Δ, every ξ_AoI
  value scales by 1/0.35 = 2.86× (ranking unchanged). Both are in `fig1_xi_aoi_by_condition.csv`.
- The energy scope is the repo's: robot uplink + UAV→satellite uplink + UAV compute. There is no locomotion energy, so speed classes do not change energy directly.

## Update 03:50 ET (partial results)
- trim.json rule 3: A1 (WAIT, same speed) at +150 ms, N=32 dropped (6 h budget + memory: WAIT +150 ms N=32 episodes ran 6-27 min and grew past 8 GB; A2 c04 and c07 were OOM-killed by the kernel). A2 at that cell is kept (n=8).
- run_job.py: heavy lock now also serializes every N=32 episode (failing 32-agent episodes grow to 8-9 GB).
- Figures carry a footnote listing cells with n<10; full list in figures/partial_cells.csv.
- Refresh after the run: `cd /workspace/leo_study && nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/analyze.py && nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig5b_latency_by_n.py`

## Readability replot (07:30 ET)
- harness/paper_figs.py overwrites fig2, fig4, fig5b, fig6, fig7, fig8 (+ fig2b, fig6a/b, fig8a/b) with paper-style plots (Okabe-Ito colors, >=12 pt, 300 dpi, shared legends, k/n labels, partial cells faded/hatched). Old versions copied to figures/old_0724/.
- Full refresh order once the run finishes: analyze.py -> fig5b_latency_by_n.py -> paper_figs.py (paper_figs must run last so its fig2/4/5b/6/7/8 win):
  `cd /workspace/leo_study && nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/analyze.py && nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig5b_latency_by_n.py && nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/paper_figs.py`
- No-success variant (07:45 ET): `nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/paper_figs.py --no-success` writes figures/no_success/<fig>_ns.{png,pdf,csv} (fig8b, fig6a, fig6b, fig2, fig2b, fig4, fig5b, fig7; fig8a dropped). fig6 there uses stalled decisions (scheduled_cutoff_waits / eligible_decisions) and attempted collisions per 1,000 moves (physical_distance_cells), all episodes.
- 07:48 ET relabel (paper_figs.py only): display label '+150 ms' -> '100 ms deadline', '+300 ms' -> 'real time' (labels only; the runs are still +150 / +300 ms added downlink with the 250 ms cutoff). Subtitles removed; panel letters (a)/(b)/(c) added. Backups: harness/paper_figs.py.bak_0743, .bak_0746. figures/old_0724/ untouched.

## 08:50 ET additions
- paper_figs.py: 'trimmed'/'pending'/'n/a' annotations removed; empty or undefined cells are simply omitted (backup paper_figs.py.bak_0846).
- harness/fig9_aoi.py -> figures/aoi_semantic/ (fig9 fine 0.25-Delta bins merged to >=200 pair decisions, plotted at count-weighted mean AoI; fig9b coarse bins; fig9_by_condition.csv).
- Priority add-on: harness/run_job_prio.py (runtime monkeypatch of RemoteDTCommunication.update/_admit; validated identical to runs/main for nominal A4 allocated N=8 c00), harness/dispatch_prio.py (--phase 1: N=8/16 x 3 routes, 2 workers; --phase 2: N=32 allocated+cloud, 1 worker, 20-min cap), runs/prio/waiter_phase2.sh, harness/prio_figs.py -> figures/priority/.

## Prior-archive AoI extraction (09:10 ET, Oct 27)
- harness/prior_aoi_extract.py RUN_DIR [--name NAME] [--conditions GLOB..]: rebuilds pairs per archive with the run's own frozen
  code (<run>/code/panama_uav; falls back to SANGIN-dev/src) via remote_audit.replay_trace (every snapshot digest-checked),
  aggregates like pair_stats.aoi (0.05 s bins, active = truth-not-completed), Δ = trace remote_settings.seconds_per_step.
  Cache: figures/aoi_semantic/prior/<name>_episodes_aoi.jsonl (resumable); figures <name>_fig9*, csv.
- harness/prior_aoi_validate.py RUN_DIR NAME: cross-checks against archived consistency_event summaries, audit
  ordering_pairs_verified, frames completed_flags, and an un-hooked replay. Validation archives: prior_val/comm_n8_v2 (5 eps),
  prior_val/comm_n8_c7 (2 eps with errors), generated by harness/prior_val_run.py (communication_effects prepare + _episode). ALL PASS.
- Extractor also reads gzip-pickle archives (*.pkl.gz: omega <case>.pkl.gz; redesign .../<case>/trace.pkl.gz; failed_* skipped),
  checks archive sha256 against a sibling result json audit when present, --code overrides the frozen source root.
  Pickle path tested on a re-pickled validation trace (prior_val/pkl_layout).
