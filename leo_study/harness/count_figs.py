#!/workspace/SANGIN-dev/.venv/bin/python
"""Count versions (raw numbers of ordering errors, no denominators) of
  (1) the combined Xi run-level figures (harness/redesign_xi_figs.py data)  -> figures/aoi_semantic/redesign/*_counts.{png,pdf}
  (2) AoI layout C (points + whiskers, fresh/stale)  (harness/omega_aoi_figs.py data, delta bins) -> figures/aoi_semantic/omega/mockups/C_points_*_counts.png
Statistic: Xi = MEAN errors per run (same-speed N=32 has only 1-3 completed runs vs 5 mixed-speed runs, so totals would not be comparable);
           AoI = TOTAL errors per bin over the 5 episodes of every series (N=8/16: 5 episodes each, failures included).
Whiskers = 95% bootstrap over runs/episodes of that statistic. Read-only on inputs; percentage versions untouched."""
import sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/leo_study/harness')
import redesign_xi_figs as RX
import omega_aoi_figs as OA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11.5,
                     'axes.grid': True, 'grid.alpha': .25, 'axes.grid.axis': 'y', 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
B = 2000; RNG = np.random.default_rng(20261027)
def boot(v, stat):
    v = np.asarray(v, float); s = stat(RNG.choice(v, (B, len(v))), axis=1); return np.percentile(s, 2.5), np.percentile(s, 97.5)

# ---------------- (1) Xi run-level counts ----------------
XK = [('ordering_errors', 'Ordering errors, all directed pairs\n(mean per run)'),
      ('fov_errors', 'Ordering errors, conflict-relevant pairs\n(mean per run)'),
      ('physical_relevant_ordering_errors', 'Ordering errors, physically conflicting pairs\n(mean per run)')]
NS = RX.NS
def xi_cells(groups, key):
    out = []
    for ci, (lab, col, mk, ls, rows) in enumerate(groups):
        for gi, n in enumerate(NS):
            cell = [r for r in rows if r['num_agents'] == n and r[key] is not None]
            if not cell: continue
            va = np.array([r[key] for r in cell], float); lo, hi = boot(va, np.mean)
            out.append(dict(ci=ci, gi=gi, n=n, lab=lab, col=col, mk=mk, ls=ls, va=va, m=va.mean(), lo=lo, hi=hi, total=va.sum()))
    return out
def xi_panel(ax, groups, key, ylab, recs, name):
    cells = xi_cells(groups, key); w = .8 / len(groups)
    pos = [c['m'] for c in cells if c['m'] > 0] + [v for c in cells for v in c['va'] if v > 0]
    logy = bool(pos) and max(pos) / min(pos) >= 50
    floor = (min(pos) / 3 if pos else .1) if logy else 0.
    for ci, (lab, col, mk, ls, rows) in enumerate(groups):
        xs, ms = [], []
        for c in [c for c in cells if c['ci'] == ci]:
            x0 = c['gi'] + (ci - (len(groups) - 1) / 2) * w; va = c['va']
            ax.scatter(x0 + RNG.uniform(-.025, .025, len(va)), np.where(va > 0, va, floor), s=14, color=col, alpha=.4, lw=0, zorder=2, marker=mk)
            m = c['m']; xs.append(x0); ms.append(m)
            if m > 0: ax.errorbar(x0, m, yerr=[[m - max(c['lo'], floor)], [c['hi'] - m]], color=col, marker=mk, ms=7, mec='white', mew=.8, capsize=2.5, lw=1.5, zorder=4)
            else: ax.plot(x0, floor, marker=mk, ms=7, mfc='white', mec=col, mew=1.6, zorder=4, ls='none')
            recs.append(dict(figure=name, metric=key, series=lab, num_agents=c['n'], runs=len(va), mean_errors_per_run=m, ci_low=c['lo'], ci_high=c['hi'],
                             total_errors=c['total'], per_run=' '.join(str(int(v)) for v in va), y_scale='log' if logy else 'linear'))
        ms_ = np.array(ms, float)
        ax.plot(xs, np.where(ms_ > 0, ms_, np.nan) if logy else ms_, color=col, lw=1.4, ls=ls, alpha=.85, zorder=3)
    if logy:
        ax.set_yscale('log'); ax.axhline(floor, color='0.55', lw=.9, ls=':', zorder=1); ax.set_ylim(floor / 2.5, None)
        ax.text(-.62, floor * 1.15, '0 errors', fontsize=10, color='0.4', va='bottom')
        from matplotlib.ticker import FuncFormatter; ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    else:
        top = max([c['hi'] for c in cells] + [v for c in cells for v in c['va']] + [1]); ax.set_ylim(-.04 * top * 1.1, top * 1.1)
        ax.axhline(0, color='0.55', lw=.9, ls=':', zorder=1)
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlim(-.65, len(NS) - .35)
    ax.set_ylabel(ylab); ax.set_xlabel('Number of robots N')
def xi_fig(groups, name, ncol, recs):
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.2))
    for ax, (key, ylab) in zip(axes, XK): xi_panel(ax, groups, key, ylab, recs, name)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls=ls, lw=1.4, ms=7.5, label=l) for l, c, m, ls, _ in groups], loc='lower center', bbox_to_anchor=(.5, .86), ncol=ncol)
    for ext in ('png', 'pdf'): fig.savefig(RX.OUT / f'{name}_counts.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); print(RX.OUT / f'{name}_counts.png')

# ---------------- (2) AoI layout C counts ----------------
BINS = [0, 1]; BL = ['fresh (AoI < 1Δ)', 'stale (1Δ ≤ AoI < 2Δ)']
AP = [(0, 1, OA.MIN_A, 'Ordering errors, all active pairs'), (2, 3, OA.MIN_B, 'Ordering errors, physically conflicting pairs')]
MOCK = Path('/workspace/leo_study/figures/aoi_semantic/omega/mockups')
def aoi_fig(eps, S, name, recs):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True); ns = len(S); tops = [0, 0]
    for ri, n in enumerate((8, 16)):
        for ci, (cr, er, mn, ylab) in enumerate(AP):
            ax = axes[ri, ci]
            for si, (lab, col, mk, sel) in enumerate(S):
                sub = [e for e in eps if e['n'] == n and sel(e)]
                if not sub: continue
                C = np.stack([e['M'][cr] for e in sub]); E = np.stack([e['M'][er] for e in sub]); off = (si - (ns - 1) / 2) * .09
                for b in BINS:
                    if C[:, b].sum() < mn: continue            # same drop rule as the percentage plots
                    y = E[:, b].sum(); lo, hi = boot(E[:, b], np.sum)
                    ax.errorbar(b + off, y, yerr=[[y - lo], [hi - y]], fmt='none', ecolor=col, elinewidth=1.4, capsize=3, zorder=3)
                    ax.scatter(b + off, y, marker=mk, s=52, facecolor=col if y > 0 else 'white', edgecolor=col, lw=1.6, zorder=4)
                    tops[ci] = max(tops[ci], hi, y)
                    recs.append(dict(figure=name, panel=ylab, num_agents=n, series=lab, aoi_bin=BL[b], episodes=len(sub), total_errors=int(y),
                                     ci_low=lo, ci_high=hi, per_episode=' '.join(str(int(v)) for v in E[:, b])))
            ax.set_xlim(-.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(BL); ax.grid(axis='x', visible=False); ax.set_ylabel(ylab)
            ax.axhline(0, color='0.55', lw=.9, ls=':', zorder=1)
            ax.text(.02, .97, f'N = {n}', transform=ax.transAxes, fontsize=12.5, fontweight='bold', va='top', ha='left', zorder=10)
            if ri == 1: ax.set_xlabel('AoI, max(A$_i^{S_j}$, A$_i^{S_i}$) / Δ')
    for (ri, ci), ax in np.ndenumerate(axes): t = max(tops[ci], 1) * 1.2; ax.set_ylim(-.03 * t, t)
    fig.legend(handles=[Line2D([], [], ls='', marker=mk, color=col, ms=8, label=lab) for lab, col, mk, _ in S], loc='lower center', bbox_to_anchor=(.5, .975), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(MOCK / f'{name}_counts.png', bbox_inches='tight', dpi=300); plt.close(fig); print(MOCK / f'{name}_counts.png')

def write(path, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(path, 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rr)

if __name__ == '__main__':
    rows, _ = RX.load(); recs = []
    same = lambda r, k, w='W1': r['series'] == 'same' and r['route'] == k and r['weight_id'] == w and not (k == 'edge' and r['num_agents'] == 32)
    g_route = [(f'{RX.R_LAB[k]}, same speed', RX.R_C[k], 'o', '--', [r for r in rows if same(r, k)]) for k in RX.R_LAB] + \
              [(f'{RX.R_LAB[k]}, mixed speed', RX.R_C[k], 's', '-', [r for r in rows if r['series'] == 'mixed' and r['route'] == k]) for k in RX.R_LAB]
    g_omega = [(RX.W_LAB[w], RX.W_C[w], 'o', '-', [r for r in rows if same(r, 'allocated', w)]) for w in RX.W_LAB] + \
              [('mixed speed', RX.MIX_C, 's', '-', [r for r in rows if r['series'] == 'mixed' and r['route'] == 'allocated'])]
    xi_fig(g_route, 'redesign_xi_run_level', 2, recs); xi_fig(g_omega, 'redesign_xi_run_level_by_omega', 3, recs)
    write(RX.OUT / 'redesign_xi_counts_summary.csv', recs)
    eps = OA.load()
    for e in eps: e['M'], e['X'], e['S'] = OA.binned(e, 'delta')
    RC = {'allocated': ('allocator', '#0072B2'), 'cloud': ('all-cloud', '#009E73'), 'edge': ('all-UAV', '#E69F00')}
    s_route = [(f'{RC[r][0]}, {sp}', RC[r][1], 'o' if not m else 's', (lambda e, r=r, m=m: e['mixed'] == m and e['w'] == 'W1' and e['route'] == r))
               for r in RC for m, sp in ((False, 'same speed'), (True, 'mixed speed'))]
    s_omega = [(f'ω₁:ω₂ = {OA.WL[w]}', OA.WC[w], 'o', (lambda e, w=w: not e['mixed'] and e['w'] == w and e['route'] == 'allocated')) for w in OA.WL] + \
              [('mixed speed', '#222222', 's', (lambda e: e['mixed'] and e['route'] == 'allocated'))]
    arecs = []; aoi_fig(eps, s_route, 'C_points_by_route', arecs); aoi_fig(eps, s_omega, 'C_points_by_omega', arecs)
    write(MOCK / 'C_points_counts_summary.csv', arecs)
    for r in recs: print('XI', r['figure'][18:], r['metric'][:8], r['series'], r['num_agents'], r['runs'], f"{r['mean_errors_per_run']:.2f}", r['per_run'], r['y_scale'])
    for r in arecs: print('AOI', r['figure'], r['panel'][16:], r['num_agents'], r['series'], r['aoi_bin'][:5], r['episodes'], r['total_errors'], r['per_episode'])
    from collections import Counter; print('episodes per AoI series/N:', Counter((r['figure'], r['series'], r['num_agents'], r['episodes']) for r in arecs).keys() and sorted({r['episodes'] for r in arecs}))
