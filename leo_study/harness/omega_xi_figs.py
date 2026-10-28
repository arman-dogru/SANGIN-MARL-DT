#!/workspace/SANGIN-dev/.venv/bin/python
"""Run-level semantic ordering-error (Xi) rates for Arman's omega-sensitivity run.
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/omega_xi_figs.py
Read-only over SANGIN-dev/results/omega_sensitivity/episodes.csv (published bundle: completed missions only, 298 runs).
Per run r:  R_E2E = ordering_errors / ordering_pairs_verified   (= sum Xi_E2E / sum_n N(N-1); checked = steps*N*(N-1),
            i.e. every robot incl. finished ones is a receiver at every epoch; same speed only, Delta = 1.0 s)
            R_FoV = fov_errors / fov_relevant_pairs              (= sum zeta*Xi / sum zeta; undefined if no relevant pair)
            R_FoV_UL / R_FoV_DL = fov_uplink_errors / fov_downlink_errors over fov_relevant_pairs (E2E-level UL/DL not in bundle)
Style follows harness/fig10_run_level.py (jittered run dots, mean + 95% bootstrap CI, log y, '0 errors' floor, hollow = all runs 0).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = Path('/workspace/SANGIN-dev/results/omega_sensitivity/episodes.csv')
STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/aoi_semantic/omega'; OUT.mkdir(parents=True, exist_ok=True)
B = 2000; RNG = np.random.default_rng(20261027); DELTA = 1.0; NS = (2, 4, 8, 16, 32); FLOOR = 3e-4
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.grid': True, 'grid.alpha': .25, 'axes.grid.axis': 'y',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
# colours sampled from Arman's legend image (01M5YWBCHC43W683QXM01ZXV61.png)
W_LAB = {'W1': 'ω₁:ω₂ = 1:1', 'W2': 'ω₁:ω₂ = 1:2', 'W3': 'ω₁:ω₂ = 1:4', 'W4': 'ω₁:ω₂ = 2:1', 'W5': 'ω₁:ω₂ = 4:1'}
W_C = {'W1': '#3E5B86', 'W2': '#4F9B8F', 'W3': '#CDA84F', 'W4': '#8865AB', 'W5': '#C96B6E'}
R_LAB = {'allocated': 'allocator', 'cloud': 'all-cloud', 'edge': 'all-UAV'}
R_C = {'allocated': '#0072B2', 'cloud': '#009E73', 'edge': '#E69F00'}
YL = {'R_E2E': 'All directed pairs, $R_{\\mathrm{E2E}}$\n(% of pair decisions, log)',
      'R_FoV': 'Conflict-relevant pairs, $R_{\\mathrm{FoV}}$\n(% of relevant pairs, log)'}

def rat(a, b): return a / b if b else None

def load():
    rows, bad = [], 0
    for r in csv.DictReader(open(SRC)):
        N, st = int(r['num_agents']), int(r['steps']); I = lambda k: int(float(r[k]))
        den, rel = I('ordering_pairs_verified'), I('fov_relevant_pairs')
        bad += den != st * N * (N - 1)
        rows.append(dict(weight_id=r['weight_id'], omega=W_LAB[r['weight_id']], route=r['route'], route_label=R_LAB[r['route']],
            num_agents=N, case_id=r['case_id'], success=r['success'], steps=st, decision_interval_s=DELTA,
            ordering_errors=I('ordering_errors'), ordering_pairs_verified=den, grid_steps_N_Nm1=st * N * (N - 1),
            fov_relevant_pairs=rel, fov_errors=I('fov_errors'), fov_uplink_errors=I('fov_uplink_errors'), fov_downlink_errors=I('fov_downlink_errors'),
            R_E2E=rat(I('ordering_errors'), den), R_FoV=rat(I('fov_errors'), rel),
            R_FoV_UL=rat(I('fov_uplink_errors'), rel), R_FoV_DL=rat(I('fov_downlink_errors'), rel),
            vertex_relevant_pairs=I('vertex_relevant_pairs'), vertex_e2e_errors=I('vertex_e2e_errors'), R_vertex=rat(I('vertex_e2e_errors'), I('vertex_relevant_pairs')),
            edge_relevant_pairs=I('edge_relevant_pairs'), attempted_vertex_conflicts=I('attempted_vertex_conflicts'),
            attempted_edge_swaps=I('attempted_edge_swaps'), stale_fresh_disagreements=I('stale_fresh_disagreements'),
            stale_fresh_rate=r['stale_fresh_rate'], timely_stale_disagreements=I('timely_stale_disagreements'),
            physical_relevant_pairs=I('physical_relevant_pairs'), physical_fov_order_errors=I('physical_fov_order_errors'),
            physical_fov_order_error_rate=r['physical_fov_order_error_rate'], inference_relevant_pairs=I('inference_relevant_pairs'),
            inference_fov_errors=I('inference_fov_errors'), action_dt_ordering_errors=I('action_dt_ordering_errors'),
            action_order_known_pairs=I('action_order_known_pairs'), mean_peer_aoi_s=r['mean_peer_aoi'],
            mean_priorities_age_s=r['mean_priorities_age_seconds'], waiting_steps=I('waiting_steps'), makespan_seconds=r['makespan_seconds']))
    return rows, bad

def boot_mean(v):
    v = np.asarray(v, float)
    if not len(v): return (np.nan,) * 3
    m = RNG.choice(v, (B, len(v))).mean(1); return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def panel(ax, groups, key, ylab):
    """groups: list of (label, colour, marker, rows)."""
    w = .8 / len(groups); recs = []
    for ci, (lab, col, mk, rows) in enumerate(groups):
        xs, ms = [], []
        for gi, n in enumerate(NS):
            cell = [r for r in rows if r['num_agents'] == n and r[key] is not None]
            if not cell: continue
            va = np.array([100 * r[key] for r in cell]); x0 = gi + (ci - (len(groups) - 1) / 2) * w
            ax.scatter(x0 + RNG.uniform(-.04, .04, len(va)), np.where(va > 0, va, FLOOR), s=14, color=col, alpha=.4, lw=0, zorder=2, marker=mk)
            m, lo, hi = boot_mean(va); xs.append(x0); ms.append(m)
            num, den = {'R_E2E': ('ordering_errors', 'ordering_pairs_verified'), 'R_FoV': ('fov_errors', 'fov_relevant_pairs')}[key]
            E, D = sum(r[num] for r in cell), sum(r[den] for r in cell)
            recs.append(dict(metric=key, series=lab, num_agents=n, runs=len(va), zero_runs=int((va == 0).sum()), errors=E, denominator=D,
                             pooled_ratio_pct=100 * E / D if D else None, mean_pct=m, ci_low_pct=lo, ci_high_pct=hi, median_pct=float(np.median(va))))
            if m > 0:
                ax.errorbar(x0, m, yerr=[[m - max(lo, FLOOR)], [hi - m]], color=col, marker=mk, ms=8, mec='white', mew=.8, capsize=3, lw=1.6, zorder=4)
            else:
                ax.plot(x0, FLOOR, marker=mk, ms=8, mfc='white', mec=col, mew=1.7, zorder=4, ls='none')
        ms_ = np.array(ms, float); ax.plot(xs, np.where(ms_ > 0, ms_, np.nan), color=col, lw=1.4, alpha=.85, zorder=3)
    ax.set_yscale('log'); ax.set_ylim(FLOOR / 4.5, 1); ax.axhline(FLOOR, color='0.55', lw=.9, ls=':', zorder=1)
    ax.text(-.62, FLOOR * 1.25, '0 errors', fontsize=10, color='0.4', va='bottom')
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(['0.001', '0.01', '0.1', '1'])
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlim(-.65, len(NS) - .35)
    ax.set_ylabel(ylab); ax.set_xlabel('Number of robots N'); return recs

def figure(groups, name, view):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0)); recs = []
    for ax, key in zip(axes, ('R_E2E', 'R_FoV')):
        rr = panel(ax, groups, key, YL[key]); [x.update(view=view) for x in rr]; recs += rr
    fig.tight_layout(rect=(0, 0, 1, .9))
    h = [Line2D([], [], color=c, marker=m, lw=1.4, ms=8, label=l) for l, c, m, _ in groups]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .9), ncol=len(groups))
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); return recs

def write(name, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rr)

if __name__ == '__main__':
    rows, bad = load(); print('runs', len(rows), 'denominator != steps*N*(N-1):', bad)
    write('omega_xi_per_run', rows)
    alloc = [(W_LAB[w], W_C[w], 'o', [r for r in rows if r['route'] == 'allocated' and r['weight_id'] == w]) for w in W_LAB]
    routes = [(R_LAB[k], R_C[k], 'o', [r for r in rows if r['route'] == k and r['weight_id'] == 'W1']) for k in R_LAB]
    recs = figure(alloc, 'omega_xi_run_level_allocator_by_omega', 'allocator by omega') + figure(routes, 'omega_xi_run_level_routes_W1', 'routes, omega 1:1')
    # FoV UL/DL split (CSV only): pooled and run means per route x omega x N
    for k in R_LAB:
        for w in W_LAB:
            for n in NS:
                cell = [r for r in rows if r['route'] == k and r['weight_id'] == w and r['num_agents'] == n and r['R_FoV'] is not None]
                if not cell: continue
                for key in ('R_FoV_UL', 'R_FoV_DL'):
                    m, lo, hi = boot_mean([100 * r[key] for r in cell])
                    recs.append(dict(metric=key, series=f'{R_LAB[k]} {W_LAB[w]}', num_agents=n, runs=len(cell), mean_pct=m, ci_low_pct=lo, ci_high_pct=hi,
                                     errors=sum(r['fov_uplink_errors' if key.endswith('UL') else 'fov_downlink_errors'] for r in cell),
                                     denominator=sum(r['fov_relevant_pairs'] for r in cell), view='FoV UL/DL split'))
    write('omega_xi_summary', recs)
    for r in recs:
        if r['metric'] in ('R_E2E', 'R_FoV'): print(r['view'], r['metric'], r['series'], r['num_agents'], r['runs'], r['zero_runs'], f"{r['mean_pct']:.4f} [{r['ci_low_pct']:.4f},{r['ci_high_pct']:.4f}] pooled {r['pooled_ratio_pct']:.4f} err {r['errors']}")
