#!/workspace/SANGIN-dev/.venv/bin/python
"""Omega run: Pr(semantic ordering error) vs AoI (decision intervals, Delta = 1 s) from omega_aoi_extract caches.
(a) Xi^E2E on DT consistency pairs (decision-epoch snapshots, frozen replay), active pairs (neither robot finished).
(b) Xi~ physical reference: archived inference shadows, pairs whose TRUE 5-step paths conflict in the receiver FoV,
    excluding pairs whose CACHED peer state already says 'finished' (priority then fixed => error impossible; these are
    the stale [10,inf) pairs); error = cached order used by the robot != true order at the inference instant.
Mixed speed = redesign A4 (Delta = 0.35 s) from Arman's archived traces via harness/redesign_archive_extract.py -> redesign_aoi/A4_archive_episodes.jsonl
(same 0.05 s raw bins, same pair definitions). Two x-axes: AoI/Delta (per-episode Delta) and AoI in seconds.
Bins (Arman's spec) [0,1),[1,2),[2,3),[3,5),[5,10),[10,inf) Delta; AoI = max(self, peer) age / Delta.
Bands = 95% bootstrap over episodes. Bins with < MIN_A (a) / MIN_B (b) pair decisions omitted."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = Path('/workspace/leo_study/figures/aoi_semantic/omega'); DELTA = 1.0; A4F = Path('/workspace/leo_study/redesign_aoi/A4_archive_episodes.jsonl')  # archived traces; rerun copy A4_episodes.jsonl = cross-check only
RAW = 401
SEDGES = np.array([0, .35, .7, 1., 2., 3.5, np.inf]); SLABS = ['[0,0.35)', '[0.35,0.7)', '[0.7,1)', '[1,2)', '[2,3.5)', '[3.5,∞)']
MIX_C = '#222222'
EDGES = np.array([0, 1, 2, 3, 5, 10, np.inf]); LABS = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,10)', '[10,∞)']; K = 6
MODES = {'delta': (EDGES, LABS, 'AoI, max(A$_i^{S_j}$, A$_i^{S_i}$) / Δ   (Δ = 1 s same speed, 0.35 s mixed speed)'),
         'seconds': (SEDGES, SLABS, 'AoI, max(A$_i^{S_j}$, A$_i^{S_i}$)  (s)')}
MIN_A, MIN_B = 200, 50; B = 2000; RNG = np.random.default_rng(20261027)
WL = {'W1': '1:1', 'W2': '1:2', 'W3': '1:4', 'W4': '2:1', 'W5': '4:1'}
WC = {'W1': '#3E5B86', 'W2': '#4F9B8F', 'W3': '#CDA84F', 'W4': '#8865AB', 'W5': '#C96B6E'}
RL = {'allocated': 'allocator', 'cloud': 'all-cloud', 'edge': 'all-UAV'}; RC = {'allocated': '#3E5B86', 'cloud': '#009E73', 'edge': '#E69F00'}
RT = {'allocated-6ghz': 'allocated', 'cloud-6ghz': 'cloud', 'uav-6ghz': 'edge'}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 12, 'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})

def binof(x): return np.searchsorted(EDGES, x, side='right') - 1

def load():
    """Per episode: raw 0.05 s-bin arrays (rows a_n, a_e, b_n, b_e, bdone_n, bdone_e) and its Delta; binning happens per x-axis mode."""
    eps = []
    def ep(r, sh, w, route, delta, mixed):
        Rw = np.zeros((8, RAW))  # rows 6,7: all shadow pairs with cache-active peer
        for ab, act, rel, e2e, ul, dl, cnt in r['aoi']:
            if act: Rw[0, ab] += cnt; Rw[1, ab] += cnt * e2e
        pe = pn = 0
        for ab, pr, oe, ft, fc, sc, cnt in sh:
            pn += cnt * pr; pe += cnt * pr * oe
            if pr and not fc: Rw[2, ab] += cnt; Rw[3, ab] += cnt * oe
            if pr and fc: Rw[4, ab] += cnt; Rw[5, ab] += cnt * oe
            if not fc: Rw[6, ab] += cnt; Rw[7, ab] += cnt * oe
        assert (pn, pe) == (r['totals']['physical_relevant_pairs'], r['totals']['physical_fov_order_errors']), r['case_id']
        return dict(n=r['num_agents'], w=w, route=route, case=r['case_id'], success=r['success'], raw=Rw, delta=delta, mixed=mixed, tot=r['totals'])
    for name in ('uav6_n8', 'uav6_n16', 'omega_w25'):
        sh2 = {(q['condition'], q['case_id']): q for q in map(json.loads, open(D / f'{name}_shadow2.jsonl'))}
        for r in map(json.loads, open(D / f'{name}_episodes.jsonl')):
            c = r['condition'].split('/'); w = c[0] if len(c) == 2 else 'W1'
            eps.append(ep(r, sh2[(r['condition'], r['case_id'])]['shadow'], w, RT[c[-1]], DELTA, False))
    seen = set()
    for r in map(json.loads, open(A4F)):
        k = (r['route'], r['case_id'])
        if k in seen or not r['validated']: continue
        seen.add(k); eps.append(ep(r, r['shadow'], 'W1', r['route'], r['delta_seconds'], True))
    return eps

def binned(e, mode):
    edges = MODES[mode][0]; x = (np.arange(RAW) + .5) * .05 / (e['delta'] if mode == 'delta' else 1.)
    b = np.searchsorted(edges, x, side='right') - 1; M = np.zeros((6, K)); X = np.zeros((2, K))
    for i in range(6): M[i] = np.bincount(b, e['raw'][i], K)
    X[0] = np.bincount(b, e['raw'][0] * x, K); X[1] = np.bincount(b, e['raw'][2] * x, K)
    S = np.stack([np.bincount(b, e['raw'][i], K) for i in (6, 7)])
    return M, X, S

def boot(C, E):
    idx = RNG.integers(0, len(C), (B, len(C))); r = E[idx].sum(1) / np.maximum(C[idx].sum(1), 1e-12)
    r = np.where(C[idx].sum(1) > 0, r, np.nan)
    return np.nanpercentile(r, 2.5, 0), np.nanpercentile(r, 97.5, 0)

def fmt(v):
    v = int(v); return f'{v/1e3:.0f}k' if v >= 1e4 else f'{v/1e3:.1f}k' if v >= 1e3 else str(v)

def figure(eps, groups, fname, recs, mode):
    labs_ = MODES[mode][1]
    for e in eps: e['M'], e['X'], e['S'] = binned(e, mode)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6), sharex=True); colmax = [0, 0]; used = set()
    for ri, n in enumerate((8, 16)):
        for s, (ci, ei, mn, panel) in enumerate(((0, 1, MIN_A, 'DT pairs, Ξ$^{E2E}$'), (2, 3, MIN_B, 'physically conflicting pairs, Ξ̃'))):
            ax = axes[ri, s]; top = 0; labs = {}
            for gi, (gkey, lab, col, sel, mk, ls) in enumerate(groups):
                sub = [e for e in eps if e['n'] == n and sel(e)]
                if not sub: continue
                C = np.stack([e['M'][ci] for e in sub]); E = np.stack([e['M'][ei] for e in sub]); cc, ee = C.sum(0), E.sum(0)
                lo, hi = boot(C, E); est = ee / np.maximum(cc, 1e-12); keep = cc >= mn
                xm = sum(e['X'][s] for e in sub) / np.maximum(cc, 1e-12)
                for b in range(K):
                    if cc[b]: recs.append(dict(figure=fname, x_axis=mode, panel='a_dt_e2e_active' if s == 0 else 'b_physical_relevant_cache_active', num_agents=n, group=lab,
                        episodes=len(sub), aoi_bin=labs_[b], aoi_mean_x=round(xm[b], 3), pair_decisions=int(cc[b]), errors=int(ee[b]),
                        pr_error=est[b], ci_low=lo[b], ci_high=hi[b], plotted=bool(keep[b])))
                if not keep.any(): continue
                used.update(np.where(keep)[0].tolist()); off = (gi - (len(groups) - 1) / 2) * .06; x = np.arange(K)[keep] + off
                y, l, h = 100 * est[keep], 100 * lo[keep], 100 * hi[keep]
                ax.fill_between(x, l, h, color=col, alpha=.13, lw=0)
                ax.plot(x, y, color=col, lw=1.8, ls=ls, zorder=3)
                nz = y > 0
                ax.scatter(x[nz], y[nz], color=col, s=44, marker=mk, zorder=4)
                ax.scatter(x[~nz], y[~nz], facecolor='white', edgecolor=col, s=44, lw=1.6, marker=mk, zorder=4)
                top = max(top, np.nanmax(h))
                for b in np.where(keep)[0]: labs.setdefault(b, []).append((col, f'{fmt(ee[b])}/{fmt(cc[b])}'))
            for b, L in ([] if True else labs.items()):
                for i, (col, t) in enumerate(L):
                    ax.annotate(t, (b, 1), xycoords=('data', 'axes fraction'), textcoords='offset points', xytext=(0, -4 - 11 * i),
                                fontsize=8.5, color=col, ha='center', va='top')
            colmax[s] = max(colmax[s], top)
            ax.text(.02, .97, f'N = {n}', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
            ax.set_ylabel('Ordering error, all active pairs (%)' if s == 0 else 'Ordering error, conflicting pairs (%)')
            if ri == 1: ax.set_xlabel(MODES[mode][2])
    hi = max(used) if used else K - 1
    for (ri, s), ax in np.ndenumerate(axes):
        t = colmax[s] * 1.55; ax.set_ylim(-.03 * t, t); ax.set_xlim(-.45, hi + .45); ax.set_xticks(range(hi + 1)); ax.set_xticklabels(labs_[:hi + 1])
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls=ls, lw=1.8, ms=7, label=l) for _, l, c, _, mk, ls in groups],
               loc='lower center', bbox_to_anchor=(.5, .975), ncol=len(groups) if len(groups) <= 3 else 3)
    fig.tight_layout(rect=(0, 0, 1, .975))
    for ext in ('png', 'pdf'): fig.savefig(D / f'{fname}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    eps = load(); recs = []
    g1 = [(w, f'ω₁:ω₂ = {WL[w]}', WC[w], (lambda e, w=w: not e['mixed'] and e['w'] == w and e['route'] == 'allocated'), 'o', '-') for w in WL]
    g1.append(('A4', 'mixed speed', MIX_C, (lambda e: e['mixed'] and e['route'] == 'allocated'), 's', '-'))
    g2 = [(r + sp, f'{RL[r]}, {lab}', RC[r], (lambda e, r=r, m=m: e['mixed'] == m and e['w'] == 'W1' and e['route'] == r), mk, ls)
          for m, sp, lab, mk, ls in ((False, '', 'same speed', 'o', '--'), (True, 'A4', 'mixed speed', 's', '-')) for r in RL]
    for mode in MODES:
        figure(eps, g1, f'omega_aoi_semantic_by_omega_aoi_{mode}', recs, mode)
        figure(eps, g2, f'omega_aoi_semantic_by_route_aoi_{mode}', recs, mode)
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(D / 'omega_aoi_semantic_bins.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)
    # extras: totals incl. finished peers, and all shadow pairs (not only conflicting)
    tot = []
    for e in eps: e['M'], e['X'], e['S'] = binned(e, 'delta')
    for n in (8, 16):
        for w in list(WL) + ['A4']:
            for r in RL:
                sub = [e for e in eps if e['n'] == n and e['route'] == r and (e['mixed'] if w == 'A4' else (not e['mixed'] and e['w'] == w))]
                if not sub: continue
                M = sum(e['M'] for e in sub); S = sum(e['S'] for e in sub); T = {k: sum(e['tot'][k] for e in sub) for k in sub[0]['tot']}
                tot.append(dict(num_agents=n, omega='mixed speed (A4, Δ=0.35 s)' if w == 'A4' else WL[w], route=RL[r], episodes=len(sub), successes=sum(bool(e['success']) for e in sub), **T,
                    physical_relevant_cache_active=int(M[2].sum()), physical_errors_cache_active=int(M[3].sum()), physical_relevant_cache_finished=int(M[4].sum()), physical_errors_cache_finished=int(M[5].sum()),
                    **{f'shadow_allpairs_cache_active_{LABS[b]}': f'{int(S[1,b])}/{int(S[0,b])}' for b in range(K) if S[0, b]}))
    keys = list(dict.fromkeys(k for r in tot for k in r))
    with open(D / 'omega_aoi_semantic_totals.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(tot)
    for r in recs: print(r['figure'][19:], r['x_axis'], r['panel'][:1], r['num_agents'], r['group'], r['aoi_bin'], r['errors'], r['pair_decisions'], f"{100*r['pr_error']:.2f}% [{100*r['ci_low']:.2f},{100*r['ci_high']:.2f}]", r['plotted'])
    for t in tot: print(t)

if __name__ == '__main__': main()
