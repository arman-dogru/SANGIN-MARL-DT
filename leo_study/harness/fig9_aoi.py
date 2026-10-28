#!/workspace/SANGIN-dev/.venv/bin/python
"""Fig 9: Pr(semantic ordering error) vs AoI (in decision intervals, Delta = 0.35 s), from runs/main pair_stats.aoi.
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig9_aoi.py
Pools treatments A4 + A3, route = allocated, all network conditions (ideal, nominal LEO, 100 ms deadline, real time).
AoI per pair-decision = max(self, peer) source age, stored in 0.05 s bins; each 0.05 s bin is assigned by its centre.
(a) active pairs: Pr(E2E error | AoI bin); (b) coordination-relevant active pairs: Pr(E2E error | AoI bin, relevant).
Fine figure: 0.25 Delta bins on [0.5, 3.5) Delta, bins with < 200 pair decisions merged into a neighbour.
Coarse figure (fig9b): bins [0,1), [1,2), [2,3), [3,5), [5,10), [10,inf) Delta. 95% bootstrap over episodes.
"""
import json, glob, math, sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/leo_study/harness')
import paper_figs as PF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/aoi_semantic'; OUT.mkdir(parents=True, exist_ok=True)
DELTA = .35; RNG = np.random.default_rng(20261027); B = 1000; MIN_N = 200
N_C = {8: '#E69F00', 16: '#CC79A7', 32: '#000000'}   # Okabe-Ito; distinct from route colours' meaning (population here)
FINE = list(np.round(np.arange(.5, 3.5001, .25), 2)); COARSE = [0, 1, 2, 3, 5, 10, math.inf]

def cond_of(j):
    if j['network'] == 'ideal': return 'ideal network'
    d = int(round(j.get('downlink_seconds', 0) * 1000))
    return {0: 'nominal LEO', 150: '100 ms deadline', 300: 'real time'}[d]

def load():
    eps = []
    for f in glob.glob(str(STUDY / 'runs/main/*/result.json')):
        p = json.load(open(f)); j = p['job']
        if j['route'] != 'allocated' or j['treatment'] not in ('A4', 'A3') or j['group'] not in ('nominal', 'baseline', 'delay'): continue
        ps = p['pair_stats']; F = ps['aoi_fields']; w = ps['aoi_bin_seconds']
        recs = [dict(zip(F, r)) for r in ps['aoi']]
        x = np.array([(r['aoi_bin'] + .5) * w / DELTA for r in recs]); act = np.array([r['active'] for r in recs], bool)
        rel = np.array([r['relevant'] for r in recs], bool); err = np.array([r['e2e_error'] for r in recs], float)
        cnt = np.array([r['count'] for r in recs], float)
        eps.append(dict(n=j['num_agents'], cond=cond_of(j), tr=j['treatment'], x=x, act=act, rel=rel, err=err, cnt=cnt))
        del p
    return eps

def per_episode(eps, edges, rel_only):
    """matrix [episode, bin] of (pair decisions, errors) for active (and relevant) pairs."""
    k = len(edges) - 1; c = np.zeros((len(eps), k)); e = np.zeros((len(eps), k)); sx = np.zeros(k)
    for i, ep in enumerate(eps):
        m = ep['act'] & (ep['rel'] if rel_only else True)
        b = np.searchsorted(edges, ep['x'][m], side='right') - 1; ok = (b >= 0) & (b < k)
        np.add.at(c[i], b[ok], ep['cnt'][m][ok]); np.add.at(e[i], b[ok], (ep['cnt'] * ep['err'])[m][ok])
        np.add.at(sx, b[ok], (ep['cnt'] * ep['x'])[m][ok])
    per_episode.sx = sx
    return c, e

def merge(edges, c, e):
    """Merge adjacent bins until each has >= MIN_N pooled pair decisions (left to right; short tail joins previous)."""
    tot = c.sum(0); groups, cur = [], []
    for i in range(len(tot)):
        cur.append(i)
        if tot[cur].sum() >= MIN_N: groups.append(cur); cur = []
    if cur:
        if groups: groups[-1] += cur
        elif tot[cur].sum() > 0: groups.append(cur)
    ne = [(edges[g[0]], edges[g[-1] + 1]) for g in groups]
    return ne, np.stack([c[:, g].sum(1) for g in groups], 1) if groups else c[:, :0], np.stack([e[:, g].sum(1) for g in groups], 1) if groups else e[:, :0]

def boot(c, e):
    est = e.sum(0) / np.maximum(c.sum(0), 1e-12); idx = RNG.integers(0, len(c), (B, len(c)))
    r = e[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-12)
    return est, np.percentile(r, 2.5, 0), np.percentile(r, 97.5, 0)

def draw(eps, edges, name, merge_bins, recs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9)); tot_n = {}
    for ax, (rel, title) in zip(axes, ((False, 'All active pairs'), (True, 'Coordination-relevant pairs'))):
        for n in (8, 16, 32):
            sub = [ep for ep in eps if ep['n'] == n]
            if not sub: continue
            c, e = per_episode(sub, np.array(edges, float), rel)
            sx = per_episode.sx; tot0 = c.sum(0)
            if merge_bins:
                bins, c, e = merge(edges, c, e)
                xm = [sx[(np.array(edges[:-1]) >= a - 1e-9) & (np.array(edges[:-1]) < b - 1e-9)].sum() / max(tot0[(np.array(edges[:-1]) >= a - 1e-9) & (np.array(edges[:-1]) < b - 1e-9)].sum(), 1e-12) for a, b in bins]
            else:
                keep = c.sum(0) > 0; bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if keep[i]]; c, e = c[:, keep], e[:, keep]
            if not bins: continue
            est, lo, hi = boot(c, e)
            xc = xm if merge_bins else [COARSE.index(a) for a, b in bins]
            ax.plot(xc, est, color=N_C[n], lw=2.2, marker='o', ms=6); ax.fill_between(xc, lo, hi, color=N_C[n], alpha=.15, lw=0)
            tot_n[(title, n)] = int(c.sum())
            for (a, b), v, l, h, cc, ee in zip(bins, est, lo, hi, c.sum(0), e.sum(0)):
                recs.append(dict(panel=title, num_agents=n, aoi_lo_delta=a, aoi_hi_delta=b, aoi_mean_delta=(xm[bins.index((a, b))] if merge_bins else None), pair_decisions=int(cc), errors=int(ee),
                                 pr_error=v, ci_low=l, ci_high=h, episodes=len(sub)))
        ax.set_title(title); ax.set_xlabel('AoI (decision intervals)'); ax.set_ylim(bottom=0)
        if merge_bins: ax.set_xlim(.5, 3.5)
        else:
            labs = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,10)', '[10,∞)']; ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs); ax.set_xlim(-.5, len(labs) - .5)
        ax.grid(True, axis='both', alpha=.25)
    axes[0].set_ylabel('Pr(semantic ordering error)')
    fig.legend(handles=[Line2D([], [], color=N_C[n], lw=2.4, marker='o', label=f'N = {n}') for n in (8, 16, 32)],
               loc='lower center', bbox_to_anchor=(.5, .9), ncol=3); fig.tight_layout(rect=(0, 0, 1, .9))
    PF_save(fig, name, 'Pair decisions per curve (all / relevant): ' + '; '.join(
        f"N = {n}: {tot_n.get(('All active pairs', n), 0):,} / {tot_n.get(('Coordination-relevant pairs', n), 0):,}" for n in (8, 16, 32))
        + '. 95% bootstrap over episodes. Per-bin counts in CSV.')

def PF_save(fig, name, foot):
    for i, ax in enumerate([a for a in fig.axes if a.get_title()]): ax.set_title(f'({chr(97 + i)}) {ax.get_title()}')
    fig.text(.01, -.01, foot, ha='left', va='top', fontsize=10.5, color='0.35', wrap=True)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

def write_csv(name, recs):
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)

if __name__ == '__main__':
    eps = load(); print('episodes', len(eps))
    r1, r2, r3 = [], [], []
    draw(eps, FINE, 'fig9_aoi_vs_semantic_error', True, r1); write_csv('fig9_aoi_vs_semantic_error', r1)
    draw(eps, COARSE, 'fig9b_coarse', False, r2); write_csv('fig9b_coarse', r2)
    edges = np.array(FINE, float)
    for cond in ('ideal network', 'nominal LEO', '100 ms deadline', 'real time'):
        for n in (8, 16, 32):
            sub = [ep for ep in eps if ep['n'] == n and ep['cond'] == cond]
            if not sub: continue
            for rel, panel in ((False, 'All active pairs'), (True, 'Coordination-relevant pairs')):
                c, e = per_episode(sub, edges, rel)
                for i in range(len(edges) - 1):
                    if c[:, i].sum(): r3.append(dict(condition=cond, num_agents=n, panel=panel, aoi_lo_delta=edges[i], aoi_hi_delta=edges[i + 1],
                        episodes=len(sub), pair_decisions=int(c[:, i].sum()), errors=int(e[:, i].sum()), pr_error=e[:, i].sum() / c[:, i].sum()))
                # outside the fine range
                for lo_, hi_, lab in ((-1, .5, '<0.5'), (3.5, 1e9, '>=3.5')):
                    c2, e2 = per_episode(sub, np.array([lo_, hi_]), rel)
                    if c2.sum(): r3.append(dict(condition=cond, num_agents=n, panel=panel, aoi_lo_delta=lab, aoi_hi_delta='', episodes=len(sub),
                        pair_decisions=int(c2.sum()), errors=int(e2.sum()), pr_error=e2.sum() / c2.sum()))
    write_csv('fig9_by_condition', r3); print('ok')
