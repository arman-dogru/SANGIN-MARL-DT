#!/workspace/SANGIN-dev/.venv/bin/python
"""Layout mockups for the omega + mixed-speed AoI ordering-error figures (A bars, B dumbbell, C points).
Reads the per-bin estimates and 95% episode-bootstrap CIs already written by harness/omega_aoi_figs.py
(omega_aoi_semantic_bins.csv, x_axis == 'delta'), so rates/CIs are identical to the current plots.
Only the two populated AoI/Δ bins are used: fresh [0,1)Δ, stale [1,2)Δ. Bins the current script drops
(plotted == False, below MIN_A/MIN_B pair decisions) are dropped here too. Read-only on inputs."""
from pathlib import Path
import csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = Path('/workspace/leo_study/figures/aoi_semantic/omega'); OUT = D / 'mockups'; OUT.mkdir(exist_ok=True)
BINS = ['[0,1)', '[1,2)']; BL = ['fresh (AoI < 1Δ)', 'stale (1Δ ≤ AoI < 2Δ)']
DNOTE = 'Δ = 1 s same speed, 0.35 s mixed speed'
PANELS = [('a_dt_e2e_active', 'Ordering error, all active pairs (%)'), ('b_physical_relevant_cache_active', 'Ordering error, conflicting pairs (%)')]
RC = {'allocator': '#0072B2', 'all-cloud': '#009E73', 'all-UAV': '#E69F00'}
WC = {'1:1': '#3E5B86', '1:2': '#4F9B8F', '1:4': '#CDA84F', '2:1': '#8865AB', '4:1': '#C96B6E'}; MIX_C = '#222222'
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12.5, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 11.5,
                     'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})

TAB = {(r['figure'], r['group'], int(r['num_agents']), r['panel'], r['aoi_bin']): (100 * float(r['pr_error']), 100 * float(r['ci_low']), 100 * float(r['ci_high']))
       for r in csv.DictReader(open(D / 'omega_aoi_semantic_bins.csv')) if r['x_axis'] == 'delta' and r['plotted'] == 'True' and r['aoi_bin'] in BINS}

# series: (key, csv figure, csv group, label, colour, marker, group label for bars, speed)
ROUTE = [(f'{r}|{sp}', 'omega_aoi_semantic_by_route_aoi_delta', f'{r}, {sp}', f'{r}, {sp}', RC[r], 'o' if sp == 'same speed' else 's', r, sp)
         for r in RC for sp in ('same speed', 'mixed speed')]
OMEGA = [(w, 'omega_aoi_semantic_by_omega_aoi_delta', f'ω₁:ω₂ = {w}', f'ω₁:ω₂ = {w}', c, 'o', f'ω₁:ω₂ = {w}', 'same speed') for w, c in WC.items()]
OMEGA.append(('mix', 'omega_aoi_semantic_by_omega_aoi_delta', 'mixed speed', 'mixed speed', MIX_C, 's', 'mixed speed', 'mixed speed'))
VERS = {'by_route': ROUTE, 'by_omega': OMEGA}

def get(s, n, panel, b):
    return TAB.get((s[1], s[2], n, panel, b))

def colmax(series):
    return [max([v[2] for s in series for n in (8, 16) for b in BINS if (v := get(s, n, p, b))] + [1e-3]) for p, _ in PANELS]

def tag(ax, n, x=.02): ax.text(x, .97, f'N = {n}', transform=ax.transAxes, fontsize=12.5, fontweight='bold', va='top', ha='left', zorder=10)

def fresh_stale_handles():
    return [Patch(facecolor='white', edgecolor='#555555', lw=1.8, label=BL[0]), Patch(facecolor='#555555', edgecolor='#555555', lw=1.8, label=BL[1])]

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=300); plt.close(fig); print(OUT / f'{name}.png')

# ---------- A: grouped bars ----------
def bars(ver):
    S = VERS[ver]; groups = list(dict.fromkeys(s[6] for s in S)); cm = colmax(S)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True)
    for ri, n in enumerate((8, 16)):
        for ci, (panel, ylab) in enumerate(PANELS):
            ax = axes[ri, ci]
            for gi, g in enumerate(groups):
                mem = [s for s in S if s[6] == g]; nb = 2 * len(mem); w = .8 / nb
                for si, s in enumerate(mem):
                    for bi, b in enumerate(BINS):
                        k = 2 * si + bi; x = gi - .4 + w * (k + .5); v = get(s, n, panel, b)
                        if v is None: continue
                        y, lo, hi = v
                        ax.bar(x, y, w * .9, facecolor=s[4] if bi else 'white', edgecolor=s[4], lw=1.8, zorder=3)
                        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt='none', ecolor='#333333', elinewidth=1.2, capsize=3, zorder=4)
                        if ver == 'by_route':
                            ax.scatter(x, y, marker=s[5], s=34, facecolor='white' if bi == 0 else s[4], edgecolor=s[4] if bi == 0 else 'white', lw=1.3, zorder=5)
            t = cm[ci] * 1.25; ax.set_ylim(-.02 * t, t); ax.set_xlim(-.55, len(groups) - .45)
            ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.replace('ω₁:ω₂ = ', 'ω₁:ω₂ =\n').replace('mixed speed', 'mixed\nspeed') if ver == 'by_omega' else g for g in groups], fontsize=11 if ver == 'by_omega' else 11.5)
            ax.grid(axis='x', visible=False); ax.set_ylabel(ylab); tag(ax, n)
            if ri == 1: ax.set_xlabel('Route' if ver == 'by_route' else 'Priority weights')
    h = fresh_stale_handles()
    if ver == 'by_route':
        h = [Patch(color=c, label=r) for r, c in RC.items()] + h + [Line2D([], [], ls='', marker='o', color='#555555', ms=7, label='same speed (ω₁:ω₂ = 1:1)'),
                                                                Line2D([], [], ls='', marker='s', color='#555555', ms=7, label='mixed speed')]
        h = [h[0], h[3], h[1], h[4], h[2], Line2D([], [], ls='', label=' '), h[5], h[6]]
        fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .975), ncol=4)
    else:
        fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .975), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .975)); save(fig, f'A_bars_{ver}')

# ---------- B: dumbbell ----------
def dumbbell(ver):
    S = VERS[ver]; cm = colmax(S); ny = len(S)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharey=True)
    for ri, n in enumerate((8, 16)):
        for ci, (panel, xlab) in enumerate(PANELS):
            ax = axes[ri, ci]
            for si, s in enumerate(S):
                yy = ny - 1 - si; pts = [get(s, n, panel, b) for b in BINS]
                ok = [(bi, v) for bi, v in enumerate(pts) if v is not None]
                if len(ok) == 2: ax.plot([ok[0][1][0], ok[1][1][0]], [yy, yy], color=s[4], lw=2.2, zorder=2)
                for bi, (y, lo, hi) in ok:
                    dy = .2 if bi else -.2
                    ax.errorbar(y, yy + dy, xerr=[[y - lo], [hi - y]], fmt='none', ecolor=s[4], elinewidth=1.2, capsize=2.5, zorder=3)
                    if bi: ax.scatter(y, yy, marker=s[5], s=46, facecolor=s[4], edgecolor=s[4], lw=1, zorder=5)
                    else: ax.scatter(y, yy, marker=s[5], s=110, facecolor='white', edgecolor=s[4], lw=1.8, zorder=4)
            t = cm[ci] * 1.12; ax.set_xlim(-.04 * t, t); ax.set_ylim(-.6, ny - .3)
            ax.set_yticks(range(ny)); ax.set_yticklabels([s[3] for s in S][::-1]); ax.grid(axis='y', visible=False)
            for lab in ax.get_yticklabels(): lab.set_color('#222222')
            ax.text(.98, .97, f'N = {n}', transform=ax.transAxes, fontsize=12.5, fontweight='bold', va='top', ha='right', zorder=10,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            ax.set_xlabel(xlab)
    h = [Line2D([], [], ls='', marker='o', mfc='white', mec='#555555', mew=1.8, ms=10, label=BL[0]),
         Line2D([], [], ls='', marker='o', color='#555555', ms=7, label=BL[1])]
    if ver == 'by_route':
        h += [Line2D([], [], ls='', marker='o', color='#999999', ms=8, label='same speed (ω₁:ω₂ = 1:1)'), Line2D([], [], ls='', marker='s', color='#999999', ms=8, label='mixed speed')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .975), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .975)); save(fig, f'B_dumbbell_{ver}')

# ---------- C: points with whiskers ----------
def points(ver):
    S = VERS[ver]; cm = colmax(S); ns = len(S)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True)
    for ri, n in enumerate((8, 16)):
        for ci, (panel, ylab) in enumerate(PANELS):
            ax = axes[ri, ci]
            for si, s in enumerate(S):
                off = (si - (ns - 1) / 2) * .09
                for bi, b in enumerate(BINS):
                    v = get(s, n, panel, b)
                    if v is None: continue
                    y, lo, hi = v
                    ax.errorbar(bi + off, y, yerr=[[y - lo], [hi - y]], fmt='none', ecolor=s[4], elinewidth=1.4, capsize=3, zorder=3)
                    ax.scatter(bi + off, y, marker=s[5], s=52, facecolor=s[4] if y > 0 else 'white', edgecolor=s[4], lw=1.6, zorder=4)
            t = cm[ci] * 1.25; ax.set_ylim(-.03 * t, t); ax.set_xlim(-.5, 1.5)
            ax.set_xticks([0, 1]); ax.set_xticklabels(BL); ax.grid(axis='x', visible=False); ax.set_ylabel(ylab); tag(ax, n)
            if ri == 1: ax.set_xlabel(f'AoI, max(A$_i^{{S_j}}$, A$_i^{{S_i}}$) / Δ')
    h = [Line2D([], [], ls='', marker=s[5], color=s[4], ms=8, label=s[3]) for s in S]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .975), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .975)); save(fig, f'C_points_{ver}')

if __name__ == '__main__':
    for v in VERS: bars(v); dumbbell(v); points(v)
