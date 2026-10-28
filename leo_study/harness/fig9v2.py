#!/workspace/SANGIN-dev/.venv/bin/python
"""Fig 9 v2: Pr(E2E semantic ordering error) vs AoI (decision intervals, Delta = 0.35 s), split by network condition.
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig9v2.py
Read-only over runs/main/*/result.json. Allocator route, A4 + A3 pooled, active pairs only.
AoI per pair-decision = max(self, peer) source age, stored in 0.05 s bins (bin centre used).
Bins [0.5,1),[1,1.75),[1.75,2),[2,2.75),[2.75,3.5) Delta; points with < 200 pair decisions dropped.
Marker x = decision-weighted mean AoI in the bin; band = 95% bootstrap over episodes.
Rows = N (8, 16, 32); (a) all active pairs (linear %), (b) coordination-relevant pairs (log %, zeros hollow on floor).
"""
import json, glob, sys, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/aoi_semantic'
DELTA = .35; MIN_N = 200; B = 1000; RNG = np.random.default_rng(20261027)
EDGES = np.array([.5, 1, 1.75, 2, 2.75, 3.5]); NS = (8, 16, 32)
CONDS = ['ideal network', 'nominal LEO', '100 ms deadline', 'real time']
COL = {'ideal network': '#000000', 'nominal LEO': '#56B4E9', '100 ms deadline': '#CC79A7', 'real time': '#D55E00'}
MK = {'ideal network': 'o', 'nominal LEO': 's', '100 ms deadline': 'D', 'real time': '^'}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})

def cond_of(j):
    g = j['group']
    if g == 'baseline': return 'ideal network'
    if g == 'nominal': return 'nominal LEO'
    if g == 'delay': return {150: '100 ms deadline', 300: 'real time'}.get(int(round(j['downlink_seconds'] * 1000)))
    return None

def load():
    """per episode: (n, cond, c[2,k], e[2,k], sx[2,k]) with index 0 = all active, 1 = relevant active."""
    eps = []; k = len(EDGES) - 1
    for f in glob.glob(str(STUDY / 'runs/main/*/result.json')):
        try: p = json.load(open(f))
        except (json.JSONDecodeError, OSError): continue
        j = p['job']
        if j['route'] != 'allocated' or j['treatment'] not in ('A4', 'A3') or j['num_agents'] not in NS: continue
        cond = cond_of(j)
        if cond is None: continue
        ps = p['pair_stats']; F = ps['aoi_fields']; w = ps['aoi_bin_seconds']; ix = {n: F.index(n) for n in F}
        c = np.zeros((2, k)); e = np.zeros((2, k)); sx = np.zeros((2, k)); tot = np.zeros(2); toterr = np.zeros(2); allp = alle = 0
        for r in ps['aoi']:
            allp += r[ix['count']]; alle += r[ix['count']] * r[ix['e2e_error']]
            if not r[ix['active']]: continue
            x = (r[ix['aoi_bin']] + .5) * w / DELTA; cnt = r[ix['count']]; er = r[ix['e2e_error']]
            b = np.searchsorted(EDGES, x, side='right') - 1
            for s in ((0, 1) if r[ix['relevant']] else (0,)):
                tot[s] += cnt; toterr[s] += cnt * er
                if 0 <= b < k: c[s, b] += cnt; e[s, b] += cnt * er; sx[s, b] += cnt * x
        eps.append(dict(n=j['num_agents'], cond=cond, tr=j['treatment'], c=c, e=e, sx=sx, tot=tot, toterr=toterr, allp=allp, alle=alle,
                        row_err=p['row'].get('ordering_errors'), row_pairs=p['row'].get('ordering_pairs_verified')))
        del p
    return eps

def boot(c, e):
    idx = RNG.integers(0, len(c), (B, len(c))); r = e[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-12)
    return np.percentile(r, 2.5, 0), np.percentile(r, 97.5, 0)

def fmt(v):
    v = int(v)
    if v >= 1e6: return f'{v / 1e6:.1f}M'
    if v >= 1e4: return f'{v / 1e3:.0f}k'
    if v >= 1e3: return f'{v / 1e3:.1f}k'
    return str(v)

def main():
    eps = load(); print('episodes', len(eps))
    recs, tots = [], []
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
    FLOOR = 3e-3  # percent; zero-error points drawn here
    for ri, n in enumerate(NS):
        for s, panel in enumerate(('all active pairs', 'coordination-relevant pairs')):
            ax = axes[ri, s]; tops = []; labs = {}
            for ci, cond in enumerate(CONDS):
                sub = [ep for ep in eps if ep['n'] == n and ep['cond'] == cond]
                if not sub: continue
                C = np.stack([ep['c'][s] for ep in sub]); E = np.stack([ep['e'][s] for ep in sub]); SX = sum(ep['sx'][s] for ep in sub)
                cc, ee = C.sum(0), E.sum(0); keep = cc >= MIN_N
                tots.append(dict(condition=cond, num_agents=n, panel=panel, episodes=len(sub),
                                 pair_decisions=int(sum(ep['tot'][s] for ep in sub)), errors=int(sum(ep['toterr'][s] for ep in sub))))
                lo, hi = boot(C, E); est = ee / np.maximum(cc, 1e-12); xm = SX / np.maximum(cc, 1e-12)
                for b in range(len(cc)):
                    if cc[b] == 0: continue
                    recs.append(dict(panel=panel, num_agents=n, condition=cond, episodes=len(sub), aoi_lo_delta=EDGES[b], aoi_hi_delta=EDGES[b + 1],
                                     aoi_mean_delta=round(xm[b], 3), pair_decisions=int(cc[b]), errors=int(ee[b]), pr_error=est[b],
                                     ci_low=lo[b], ci_high=hi[b], plotted=bool(keep[b])))
                if not keep.any(): continue
                x, y, l, h = xm[keep], 100 * est[keep], 100 * lo[keep], 100 * hi[keep]
                col = COL[cond]
                if s == 1:
                    yp = np.where(y > 0, y, FLOOR); lp = np.maximum(l, FLOOR); hp = np.maximum(h, FLOOR); nz = y > 0
                    ax.fill_between(x, lp, hp, color=col, alpha=.13, lw=0)
                    yl = np.where(nz, y, np.nan)  # lines only between non-zero points; zeros sit unconnected on the floor
                    ax.plot(x, yl, color=col, lw=1.8, zorder=3)
                    ax.scatter(x[nz], yp[nz], color=col, marker=MK[cond], s=42, zorder=4)
                    ax.scatter(x[~nz], yp[~nz], facecolor='white', edgecolor=col, marker=MK[cond], s=42, lw=1.6, zorder=4)
                    tops.append(hp.max())
                else:
                    ax.plot(x, y, color=col, lw=1.8, marker=MK[cond], ms=6.5, zorder=3)
                    ax.fill_between(x, l, h, color=col, alpha=.13, lw=0); tops.append(h.max())
                for b in np.where(keep)[0]:
                    if b == 0: continue  # AoI < 1 Delta: 0 errors in every cell (footnote)
                    labs.setdefault(b, []).append((cond, f'{fmt(ee[b])}/{fmt(cc[b])}'))
            # stacked count labels per bin, near the top of the panel: left column right-aligned, right column left-aligned
            for b, L in labs.items():
                xa = {1: 1.62, 2: 1.81, 3: 2.62, 4: 2.81}[b]; ha = 'right' if b in (1, 3) else 'left'
                for i, (cond, t) in enumerate(L):
                    ax.annotate(t, (xa, 1), xycoords=('data', 'axes fraction'), textcoords='offset points', xytext=(0, -4 - 11 * i),
                                fontsize=9, color=COL[cond], ha=ha, va='top')
            labs.clear()
            if s == 1:
                ax.set_yscale('log'); ax.set_ylim(FLOOR / 1.8, 400)
                ax.axhline(FLOOR, color='0.5', lw=.9, ls=':'); ax.text(.52, FLOOR * 1.15, '0 errors', fontsize=10.5, color='0.4', va='bottom')
            else:
                ax.set_ylim(0, 1.45 * max(tops))
            ax.set_xlim(.5, 3.5); ax.set_xticks([.5, 1, 1.5, 2, 2.5, 3, 3.5])
            for e_ in EDGES[1:-1]: ax.axvline(e_, color='0.85', lw=.8, zorder=0)
            ax.set_title(f'({chr(97 + 2 * ri + s)}) N = {n}, {panel}')
            if s == 0: ax.set_ylabel('Semantic error rate (%)')
            if ri == 2: ax.set_xlabel('AoI (decision intervals, Δ = 0.35 s)')
    fig.legend(handles=[Line2D([], [], color=COL[c], marker=MK[c], lw=1.8, ms=7, label=c) for c in CONDS],
               loc='lower center', bbox_to_anchor=(.5, .975), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .975))
    if False: fig.text(.01, -.003, 'Labels: errors / pair decisions per bin (in line-colour, top of panel). AoI < 1Δ: 0 errors in every cell. Bands: 95% bootstrap over episodes. Bins with < 200 decisions omitted. '
             'Allocator route, A3 + A4 pooled. Hollow markers = 0 errors.', ha='left', va='top', fontsize=10.5, color='0.35', wrap=True)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'fig9v2_aoi_vs_semantic_error.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    for name, rr in (('fig9v2_aoi_vs_semantic_error', recs), ('fig9v2_totals', tots)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rr)
    # sanity: all-pair (active + finished) totals vs row ordering_errors / ordering_pairs_verified
    bad = sum(1 for ep in eps if ep['row_err'] is not None and (ep['alle'] != ep['row_err'] or ep['allp'] != ep['row_pairs']))
    print('row mismatches', bad, 'of', len(eps))
    for t in tots: print(t)
    for r in recs: print(r['panel'][:3], r['num_agents'], r['condition'], f"[{r['aoi_lo_delta']},{r['aoi_hi_delta']})", r['aoi_mean_delta'], r['errors'], r['pair_decisions'], f"{100*r['pr_error']:.3f}%", r['plotted'])

if __name__ == '__main__': main()
