#!/workspace/SANGIN-dev/.venv/bin/python
"""Fig 10: run-level semantic (priority-ordering) error rates vs team size N, per network condition.
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig10_run_level.py
Read-only over runs/main/*/result.json, one episode (= run r) at a time.

Per run r (all receivers i, peers j != i, decision epochs n), from pair_stats.aoi (count-weighted):
  R_E2E = sum Xi_E2E / sum_n N_dec(n)(N-1)        (denominator = ordering_pairs_verified = directed_pairs)
  R_UL, R_DL analogous; R_FoV = sum zeta*Xi_E2E / sum zeta  (zeta = 'relevant': predicted coordination conflict)
N_dec(n) = robots that take a decision at epoch n. Same speed: N_dec = N, so the denominator is exactly sum_n N(N-1)
(finished robots included). Mixed speed: slow robots decide every other step, so it is smaller; the literal
steps*N(N-1) grid is also written (R_E2E_grid) for reference.
Outputs figures/aoi_semantic/fig10_run_level_semantic.{png,pdf} (split by speed), fig10b_run_level_semantic_pooled.{png,pdf}
(A1-A4 pooled), fig10_run_level_per_run.csv (one row per run, every metric), fig10_run_level_summary.csv.
"""
import json, glob, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/aoi_semantic'
B = 2000; RNG = np.random.default_rng(20261027); DELTA = .35
NS = (2, 4, 8, 16, 32)
CONDS = ['ideal network', 'nominal LEO', '100 ms deadline', 'real time']
COL = {'ideal network': '#000000', 'nominal LEO': '#56B4E9', '100 ms deadline': '#CC79A7', 'real time': '#D55E00'}
MK = {'ideal network': 'o', 'nominal LEO': 's', '100 ms deadline': 'D', 'real time': '^'}
SPEED = {'A1': 'same speed', 'A3': 'same speed', 'A2': 'mixed speed', 'A4': 'mixed speed'}
POLICY = {'A1': 'WAIT', 'A2': 'WAIT', 'A3': 'cached fallback', 'A4': 'cached fallback'}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.grid': True, 'grid.alpha': .25, 'axes.grid.axis': 'y',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})

def cond_of(j):
    g = j['group']
    if g == 'baseline': return 'ideal network'
    if g == 'nominal': return 'nominal LEO'
    if g == 'delay': return {150: '100 ms deadline', 300: 'real time'}.get(int(round(j['downlink_seconds'] * 1000)))
    if g in ('routes', 'routes_speed'): return 'nominal LEO'
    return None

def rat(a, b): return a / b if b else None

def run_metrics(p):
    j, r = p['job'], p['row']; N = j['num_agents']; ps = p['pair_stats']; F = ps['aoi_fields']; ix = {n: F.index(n) for n in F}
    w = ps['aoi_bin_seconds']; s = dict.fromkeys(('pairs', 'e2e', 'ul', 'dl', 'rel', 'rel_e2e', 'rel_ul', 'rel_dl',
                                                  'act', 'act_e2e', 'act_rel', 'act_rel_e2e', 'aoi_sum', 'rel_aoi_sum', 'act_aoi_sum'), 0)
    for rec in ps['aoi']:
        c = rec[ix['count']]; e, u, d = rec[ix['e2e_error']] * c, rec[ix['ul_error']] * c, rec[ix['dl_error']] * c
        rel, act = rec[ix['relevant']], rec[ix['active']]; a = (rec[ix['aoi_bin']] + .5) * w / DELTA * c
        s['pairs'] += c; s['e2e'] += e; s['ul'] += u; s['dl'] += d; s['aoi_sum'] += a
        if rel: s['rel'] += c; s['rel_e2e'] += e; s['rel_ul'] += u; s['rel_dl'] += d; s['rel_aoi_sum'] += a
        if act:
            s['act'] += c; s['act_e2e'] += e; s['act_aoi_sum'] += a
            if rel: s['act_rel'] += c; s['act_rel_e2e'] += e
    grid = r['steps'] * N * (N - 1)
    sa = json.loads(r['speed_assignments']); nslow = sum(v['decisions_per_step'] < 1 for v in sa.values())
    moves = r['physical_distance_cells']; dec = r['eligible_decisions']
    out = dict(job_id=j['job_id'], group=j['group'], route=j['route'], treatment=j['treatment'], speed=SPEED[j['treatment']],
               policy=POLICY[j['treatment']], condition=cond_of(j), downlink_added_ms=int(round(j.get('downlink_seconds', 0) * 1000)),
               num_agents=N, case=j['case'], slow_robots=nslow, steps=r['steps'],
               # denominators
               pair_decisions=s['pairs'], ordering_pairs_verified=r['ordering_pairs_verified'], directed_pairs=r['directed_pairs'],
               grid_pairs_steps_N_Nm1=grid, pair_decisions_over_grid=rat(s['pairs'], grid),
               # numerators
               e2e_errors=s['e2e'], ul_errors=s['ul'], dl_errors=s['dl'], row_ordering_errors=r['ordering_errors'],
               relevant_pairs=s['rel'], fov_e2e_errors=s['rel_e2e'], fov_ul_errors=s['rel_ul'], fov_dl_errors=s['rel_dl'],
               # Arman's run-level rates
               R_E2E=rat(s['e2e'], s['pairs']), R_UL=rat(s['ul'], s['pairs']), R_DL=rat(s['dl'], s['pairs']),
               R_FoV=rat(s['rel_e2e'], s['rel']), R_FoV_UL=rat(s['rel_ul'], s['rel']), R_FoV_DL=rat(s['rel_dl'], s['rel']),
               R_E2E_grid=rat(s['e2e'], grid), zeta_fraction=rat(s['rel'], s['pairs']),
               R_E2E_active=rat(s['act_e2e'], s['act']), R_FoV_active=rat(s['act_rel_e2e'], s['act_rel']), active_pair_decisions=s['act'],
               mean_aoi_max_delta=rat(s['aoi_sum'], s['pairs']), mean_aoi_max_delta_fov=rat(s['rel_aoi_sum'], s['rel']), mean_aoi_max_delta_active=rat(s['act_aoi_sum'], s['act']),
               # extras that are logged
               vertex_conflict_pairs=r['vertex_relevant_pairs'], edge_swap_conflict_pairs=r['edge_relevant_pairs'],
               vertex_conflict_frac=rat(r['vertex_relevant_pairs'], r['directed_pairs']), edge_swap_conflict_frac=rat(r['edge_relevant_pairs'], r['directed_pairs']),
               vertex_e2e_errors=r['vertex_e2e_errors'], R_vertex=rat(r['vertex_e2e_errors'], r['vertex_relevant_pairs']),
               stale_fresh_dqn_disagreements=r['stale_fresh_dqn_disagreements'], inferences=r['inferences'],
               stale_fresh_disagree_rate=rat(r['stale_fresh_dqn_disagreements'], r['inferences']),
               fallback_stale_fresh_disagreements=r['fallback_stale_fresh_disagreements'], fallback_shadow_opportunities=r['fallback_shadow_opportunities'],
               fallback_stale_fresh_disagree_rate=rat(r['fallback_stale_fresh_disagreements'], r['fallback_shadow_opportunities']),
               eligible_decisions=dec, controller_wait_decisions=r['controller_wait_decisions'], wait_action_frac=rat(r['controller_wait_decisions'], dec),
               safety_wait_decisions=r['safety_wait_decisions'], scheduled_cutoff_waits=r['scheduled_cutoff_waits'],
               stalled_decision_frac=rat(r['scheduled_cutoff_waits'], dec), waiting_steps=r['waiting_steps'],
               attempted_vertex_conflicts=r['attempted_vertex_conflicts'], attempted_edge_swaps=r['attempted_edge_swaps'],
               robot_moves=moves, collision_attempts_per_1000_moves=rat(1000 * (r['attempted_vertex_conflicts'] + r['attempted_edge_swaps']), moves),
               vertex_attempts_per_1000_moves=rat(1000 * r['attempted_vertex_conflicts'], moves), edge_swap_attempts_per_1000_moves=rat(1000 * r['attempted_edge_swaps'], moves),
               physical_resolver_blocks=r['physical_resolver_blocks'], post_resolution_conflicts=r['post_resolution_conflicts'],
               collision_failure=r['collision_failure'], physical_conflict_events=r['physical_conflict_events'],
               success=int(bool(r['success'])), makespan_seconds=r['makespan_seconds'], observed_seconds=r['observed_seconds'],
               completion_fraction=r['completion_fraction'], mean_first_completion_time_seconds=r['mean_first_completion_time_seconds'])
    chk = dict(e2e_ok=s['e2e'] == r['ordering_errors'], pairs_ok=s['pairs'] == r['ordering_pairs_verified'] == r['directed_pairs'],
               ul_ok=s['ul'] == r['uplink_errors'], dl_ok=s['dl'] == r['downlink_errors'],
               rel_ok=s['rel'] == r['relevant_pairs'], fov_ok=s['rel_e2e'] == r['fov_errors'],
               grid_ok=nslow > 0 or grid == r['directed_pairs'])
    return out, chk

def load():
    rows, bad = [], []
    for f in sorted(glob.glob(str(STUDY / 'runs/main/*/result.json'))):
        try: p = json.load(open(f))
        except (json.JSONDecodeError, OSError): continue
        o, c = run_metrics(p); del p
        if o['condition'] is None: continue
        rows.append(o)
        if not all(c.values()): bad.append((o['job_id'], {k: v for k, v in c.items() if not v}))
    return rows, bad

def boot_mean(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (np.nan,) * 3
    m = RNG.choice(v, (B, len(v))).mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

FLOOR = 3e-4  # percent: zero-error runs drawn on this line
def panel(ax, rows, key, ylab, show_counts=True):
    w = .17; recs = []
    for ci, cond in enumerate(CONDS):
        xs, ms, los, his = [], [], [], []
        for gi, n in enumerate(NS):
            v = [100 * r[key] for r in rows if r['condition'] == cond and r['num_agents'] == n and r[key] is not None]
            if not v: continue
            x0 = gi + (ci - 1.5) * w; jit = RNG.uniform(-.05, .05, len(v)); va = np.asarray(v)
            ax.scatter(x0 + jit, np.where(va > 0, va, FLOOR), s=11, color=COL[cond], alpha=.35, lw=0, zorder=2,
                       marker=MK[cond])
            m, lo, hi = boot_mean(v); xs.append(x0); ms.append(m); los.append(lo); his.append(hi)
            num, den = {'R_E2E': ('e2e_errors', 'pair_decisions'), 'R_FoV': ('fov_e2e_errors', 'relevant_pairs')}[key]
            cell = [r for r in rows if r['condition'] == cond and r['num_agents'] == n and r[key] is not None]
            pooled = 100 * sum(r[num] for r in cell) / max(1, sum(r[den] for r in cell))
            recs.append(dict(metric=key, condition=cond, num_agents=n, runs=len(v), zero_runs=int((va == 0).sum()), pooled_ratio_pct=pooled,
                             errors=sum(r[num] for r in cell), denominator=sum(r[den] for r in cell),
                             mean_pct=m, ci_low_pct=lo, ci_high_pct=hi, median_pct=float(np.median(va))))
            if m > 0:
                ax.errorbar(x0, m, yerr=[[m - max(lo, FLOOR)], [hi - m]], color=COL[cond], marker=MK[cond], ms=7.5, mec='white',
                            mew=.8, capsize=3, lw=1.6, zorder=4)
            else:
                ax.plot(x0, FLOOR, marker=MK[cond], ms=7.5, mfc='white', mec=COL[cond], mew=1.6, zorder=4, ls='none')
            if show_counts:
                ax.annotate(f'{len(v)}', (x0, 0), xycoords=('data', 'axes fraction'), textcoords='offset points', xytext=(0, 2),
                            ha='center', va='bottom', fontsize=8, color=COL[cond], alpha=.8)
        ms_ = np.array(ms, float); xs_ = np.array(xs)
        ax.plot(xs_, np.where(ms_ > 0, ms_, np.nan), color=COL[cond], lw=1.4, alpha=.85, zorder=3)
    ax.set_yscale('log'); ax.set_ylim(FLOOR / 4.5, 100)
    ax.axhline(FLOOR, color='0.55', lw=.9, ls=':', zorder=1)
    ax.text(-.62, FLOOR * 1.25, '0 errors', fontsize=10, color='0.4', va='bottom')
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100]); ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlim(-.65, len(NS) - .35)
    ax.set_ylabel(ylab)
    return recs

YL = {'R_E2E': 'All directed pairs, $R_{\\mathrm{E2E}}$\n(% of pair decisions, log)',
      'R_FoV': 'Conflict-relevant pairs, $R_{\\mathrm{FoV}}$\n(% of relevant pairs, log)'}

def legend(fig, y):
    h = [Line2D([], [], color=COL[c], marker=MK[c], lw=1.4, ms=7.5, label=c) for c in CONDS]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, y), ncol=4)

def figure_split(rows):
    """2 x 2: rows = same / mixed speed (WAIT + cached fallback pooled), cols = R_E2E / R_FoV."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True); recs = []
    for ri, sp in enumerate(('same speed', 'mixed speed')):
        sub = [r for r in rows if r['speed'] == sp]
        for ci, key in enumerate(('R_E2E', 'R_FoV')):
            ax = axes[ri, ci]
            rr = panel(ax, sub, key, YL[key])
            for x in rr: x['view'] = sp
            recs += rr
            ax.set_title(f"({chr(97 + 2 * ri + ci)}) {sp.capitalize()}, {'all pairs' if key == 'R_E2E' else 'conflict-relevant pairs'}")
            if ri == 1: ax.set_xlabel('Number of robots N')
    fig.tight_layout(rect=(0, 0, 1, .955)); legend(fig, .955)
    if False: fig.text(.01, -.004, 'Allocator route. WAIT and cached fallback pooled (same speed: A1 + A3; mixed speed: A2 + A4; ideal network: cached fallback only). '
             'Small dots: one run each (jittered); large markers: mean over runs with 95% bootstrap CI. Hollow marker on the dotted line: every run had 0 errors. '
             'Small coloured numbers at the bottom: runs per point (R_FoV is undefined for runs with no conflict-relevant pair).', ha='left', va='top', fontsize=10.5, color='0.35', wrap=True)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'fig10_run_level_semantic.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); return recs

def figure_pooled(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0)); recs = []
    for ci, key in enumerate(('R_E2E', 'R_FoV')):
        rr = panel(axes[ci], rows, key, YL[key])
        for x in rr: x['view'] = 'A1-A4 pooled'
        recs += rr
        axes[ci].set_title(f"({'ab'[ci]}) {'All directed pairs' if key == 'R_E2E' else 'Conflict-relevant pairs'}")
        axes[ci].set_xlabel('Number of robots N')
    fig.tight_layout(rect=(0, 0, 1, .9)); legend(fig, .9)
    if False: fig.text(.01, -.01, 'Allocator route, treatments A1-A4 pooled (ideal network: A3 + A4 only; some delay cells lack A1/A2 at N = 16, 32). '
             'Small dots: runs; large markers: mean, 95% bootstrap CI. Hollow marker on the dotted line: all runs 0 errors. Numbers at the bottom: runs per point.',
             ha='left', va='top', fontsize=10.5, color='0.35', wrap=True)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'fig10b_run_level_semantic_pooled.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); return recs

def write(name, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rr)

def main():
    rows, bad = load()
    print('runs', len(rows), 'check failures', len(bad)); [print(b) for b in bad[:20]]
    write('fig10_run_level_per_run', rows)
    alloc = [r for r in rows if r['route'] == 'allocated']
    recs = figure_split(alloc) + figure_pooled(alloc)
    write('fig10_run_level_summary', recs)
    for r in recs: print(r['view'], r['metric'], r['num_agents'], r['condition'], r['runs'], r['zero_runs'], f"{r['mean_pct']:.4f} [{r['ci_low_pct']:.4f},{r['ci_high_pct']:.4f}]")

if __name__ == '__main__': main()
