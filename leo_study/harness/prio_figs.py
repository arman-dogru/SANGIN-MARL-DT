#!/workspace/SANGIN-dev/.venv/bin/python
"""Priority add-on figures: command latency of high- vs low-priority robots (runs/prio, read-only).
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/prio_figs.py
Priority = true environment priority (distance to goal; lower = higher priority), ranked among ACTIVE robots
each cycle; normalised rank 0 = highest. High = top quarter (norm rank <= 0.25), low = bottom quarter (>= 0.75).
Settled robots excluded. All episodes (failed included). Undelivered commands count as > any latency.
Outputs figures/priority/*.png|pdf|csv.
"""
import json, glob, math, sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/leo_study/harness')
import paper_figs as PF          # shared style (rcParams, Okabe-Ito colours)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/priority'; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261027); B = 1000; CUTOFF = .25
ROUTES = (('allocated', 'allocator'), ('cloud', 'all-cloud'), ('edge', 'all-UAV'))
COND = {0: 'nominal LEO'}   # 100 ms deadline dropped from figures (Arman, 2026-10-27 10:18)
SPEED = {'A4': 'mixed speed', 'A3': 'same speed'}

def load():
    eps = []
    for f in glob.glob(str(STUDY / 'runs/prio/*/result.json')):
        try: p = json.load(open(f))
        except (json.JSONDecodeError, OSError): continue
        j = p['job']; recs = []
        for t in p['tasks']:
            if t.get('completed') or t.get('norm_rank') is None: continue   # settled robots excluded
            v = t.get('cycle_latency_seconds'); recs.append((float(t['norm_rank']), math.inf if v is None else float(v) * 1000))
        eps.append(dict(route=j['route'], tr=j['treatment'], dl=int(round(j.get('downlink_seconds', 0) * 1000)), n=j['num_agents'],
                        case=j['case'], r=np.array([x[0] for x in recs]), lat=np.array([x[1] for x in recs])))
    return eps

def cell(eps, route, tr, dl, n): return [e for e in eps if (e['route'], e['tr'], e['dl'], e['n']) == (route, tr, dl, n)]
def group(e, which): m = e['r'] <= .25 if which == 'high' else e['r'] >= .75; return e['lat'][m]
def q(x, p): return float(np.quantile(x, p, method='inverted_cdf')) if len(x) else math.nan

def boot_mean(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v): return (math.nan,) * 3
    m = v[RNG.integers(0, len(v), (B, len(v)))].mean(1); return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0: return (math.nan,) * 3
    i = RNG.integers(0, len(num), (B, len(num))); r = num[i].sum(1) / np.maximum(den[i].sum(1), 1e-12)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)

def stats(eps):
    """Per-episode p50/p95 (ms) and pooled cutoff-miss fraction for high vs low priority."""
    out = {}
    for w in ('high', 'low'):
        gs = [group(e, w) for e in eps]; gs_ne = [g for g in gs if len(g)]
        out[w] = dict(p50=boot_mean([q(g, .5) for g in gs_ne]), p95=boot_mean([q(g, .95) for g in gs_ne]),
                      miss=boot_ratio([np.sum(g > CUTOFF * 1000) for g in gs], [len(g) for g in gs]),
                      commands=int(sum(len(g) for g in gs)), undelivered=int(sum(np.isinf(g).sum() for g in gs)))
    return out

def write_csv(name, recs):
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)

def save(fig, name):
    titled = [ax for ax in fig.axes if ax.get_title()]
    if len(titled) > 1:
        for i, ax in enumerate(titled): ax.set_title(f'({chr(97 + i)}) {ax.get_title()}')
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

# allocator split by priority; all-cloud / all-UAV do not use priority-aware allocation -> one all-robot series each
SERIES = (('allocated', 'high', 'allocator, high priority (top 25%)', '-', False),
          ('allocated', 'low', 'allocator, low priority (bottom 25%)', '--', True),
          ('cloud', 'all', 'all-cloud (all robots)', '-', False),
          ('edge', 'all', 'all-UAV (all robots)', '-', False))
RLAB = dict(ROUTES)

def group_all(e, which): return e['lat'] if which == 'all' else group(e, which)

def series_stats(eps, which):
    gs = [group_all(e, which) for e in eps]; ne = [g for g in gs if len(g)]
    return dict(p50=boot_mean([q(g, .5) for g in ne]), p95=boot_mean([q(g, .95) for g in ne]),
                miss=boot_ratio([np.sum(g > CUTOFF * 1000) for g in gs], [len(g) for g in gs]),
                commands=int(sum(len(g) for g in gs)), undelivered=int(sum(np.isinf(g).sum() for g in gs)))

def legend_series(fig, ncol=4, y=.9):
    h = [Line2D([], [], color=PF.ROUTE_C[RLAB[r]], lw=2.4, ls=ls, marker='o', ms=8, mfc='white' if op else PF.ROUTE_C[RLAB[r]], mew=1.8, label=lab)
         for r, w, lab, ls, op in SERIES]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, y), ncol=ncol, fontsize=11.5)

def fig_a(eps, counts, recs):
    for tr in ('A4', 'A3'):
        for dl, cname in COND.items():
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.9)); any_ = False
            for ax, metric, title in ((axes[0], 'p50', 'Median latency'), (axes[1], 'p95', 'p95 latency')):
                for gi, n in enumerate(counts):
                    for si, (route, w, lab, ls, op) in enumerate(SERIES):
                        sub = cell(eps, route, tr, dl, n)
                        if not sub: continue
                        s = series_stats(sub, w); any_ = True
                        est, lo, hi = s[metric]; x = gi + (si - 1.5) * .17
                        PF.dot(ax, x, est, lo, hi, PF.ROUTE_C[RLAB[route]], open_=op, n=PF.EXPECTED, marker='o', top=400)
                        if metric == 'p50':
                            recs.append(dict(speed=SPEED[tr], condition=cname, route=RLAB[route], num_agents=n,
                                priority=w if w != 'all' else 'all robots', episodes=len(sub),
                                p50_ms=s['p50'][0], p50_ci_low=s['p50'][1], p50_ci_high=s['p50'][2],
                                p95_ms=s['p95'][0], p95_ci_low=s['p95'][1], p95_ci_high=s['p95'][2],
                                cutoff_miss_pct=100 * s['miss'][0], miss_ci_low_pct=100 * s['miss'][1], miss_ci_high_pct=100 * s['miss'][2],
                                commands=s['commands'], undelivered=s['undelivered']))
                ax.set_xticks(range(len(counts))); ax.set_xticklabels([f'N = {n}' for n in counts]); ax.set_xlim(-.6, len(counts) - .4)
                ax.set_title(title); ax.set_ylabel('Command latency (ms)'); ax.set_ylim(0, 400)
                ax.axhline(250, color='0.3', ls='--', lw=1.2); ax.text(-.55, 255, '250 ms cutoff', fontsize=11, color='0.3', va='bottom')
            if not any_: plt.close(fig); continue
            fig.tight_layout(rect=(0, 0, 1, .86)); legend_series(fig, ncol=2, y=.86)
            tag = f"{'mixed' if tr == 'A4' else 'same'}_{'nominal' if dl == 0 else 'deadline100'}"
            save(fig, f'prio_a_quartiles_{tag}')

def fig_b(eps, recs):
    fig, axes = plt.subplots(1, len(COND), figsize=(7.5 * len(COND), 4.9), sharey=True, squeeze=False); axes = axes[0]
    for ax, (dl, cname) in zip(axes, COND.items()):
        for route, w, lab, ls, op in SERIES:
            sub = cell(eps, route, 'A4', dl, 16)
            if not sub: continue
            lat = np.sort(np.concatenate([group_all(e, w) for e in sub])); fin = lat[np.isfinite(lat)]
            t = np.linspace(0, 450, 901); cdf = np.searchsorted(fin, t, side='right') / max(len(lat), 1)
            ax.plot(t, cdf, color=PF.ROUTE_C[RLAB[route]], ls=ls, lw=2.4)
            recs.append(dict(condition=cname, route=RLAB[route], priority=w if w != 'all' else 'all robots', num_agents=16, episodes=len(sub),
                             commands=len(lat), p50_ms=q(lat, .5), p95_ms=q(lat, .95), frac_le_250ms=float(np.sum(lat <= 250) / max(len(lat), 1))))
        ax.axvline(250, color='0.3', ls='--', lw=1.2); ax.text(255, .05, '250 ms\ncutoff', fontsize=11, color='0.3')
        ax.set_xlabel('Command latency (ms)'); ax.set_xlim(0, 450); ax.set_ylim(0, 1.02)
        ax.grid(True, axis='both', alpha=.25)
    axes[0].set_ylabel('Fraction of commands\nwith latency ≤ x')
    h = [Line2D([], [], color=PF.ROUTE_C[RLAB[r]], lw=2.4, ls=ls, label=lab) for r, w, lab, ls, op in SERIES]
    fig.tight_layout(rect=(0, 0, 1, .8)); fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .8), ncol=2, fontsize=11.5)
    save(fig, 'prio_b_cdf_n16')

def fig_c(eps, counts, recs):
    """Allocator: median latency per priority-rank decile. all-cloud / all-UAV: all-robot median as a flat reference line."""
    edges = np.linspace(0, 1, 11)
    for dl, cname in COND.items():
        fig, axes = plt.subplots(1, len(counts), figsize=(13, 4.9), sharey=True)
        for ax, n in zip(np.atleast_1d(axes), counts):
            sub = cell(eps, 'allocated', 'A4', dl, n)
            if sub:
                xs, est, lo, hi = [], [], [], []
                for dd in range(10):
                    per = []
                    for e in sub:
                        m = (e['r'] >= edges[dd]) & ((e['r'] < edges[dd + 1]) if dd < 9 else (e['r'] <= edges[dd + 1]))
                        if m.any(): per.append(q(e['lat'][m], .5))
                    r = boot_mean(per); xs.append(dd + 1); est.append(r[0]); lo.append(r[1]); hi.append(r[2])
                    recs.append(dict(condition=cname, route='allocator', num_agents=n, rank_decile=dd + 1, episodes=len(sub),
                                     episodes_with_data=len(per), median_ms=r[0], ci_low=r[1], ci_high=r[2]))
                ax.plot(xs, est, color=PF.ROUTE_C['allocator'], lw=2.2, marker='o', ms=6)
                ax.fill_between(xs, lo, hi, color=PF.ROUTE_C['allocator'], alpha=.15, lw=0)
            for route in ('cloud', 'edge'):
                s2 = cell(eps, route, 'A4', dl, n)
                if not s2: continue
                est, lo, hi = boot_mean([q(e['lat'], .5) for e in s2 if len(e['lat'])]); c = PF.ROUTE_C[RLAB[route]]
                ax.axhline(est, color=c, lw=2.2); ax.axhspan(lo, hi, color=c, alpha=.12, lw=0)
                recs.append(dict(condition=cname, route=RLAB[route], num_agents=n, rank_decile='all robots', episodes=len(s2),
                                 episodes_with_data=len(s2), median_ms=est, ci_low=lo, ci_high=hi))
            ax.set_title(f'N = {n}'); ax.set_xticks(range(1, 11)); ax.set_xlabel('Allocator priority-rank decile (1 = highest)')
            ax.axhline(250, color='0.3', ls='--', lw=1.2); ax.set_ylim(bottom=0)
        np.atleast_1d(axes)[0].set_ylabel('Median command latency (ms)')
        h = [Line2D([], [], color=PF.ROUTE_C['allocator'], lw=2.6, marker='o', label='allocator (by priority decile)'),
             Line2D([], [], color=PF.ROUTE_C['all-cloud'], lw=2.6, label='all-cloud (all robots)'),
             Line2D([], [], color=PF.ROUTE_C['all-UAV'], lw=2.6, label='all-UAV (all robots)')]
        fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .9), ncol=3); fig.tight_layout(rect=(0, 0, 1, .9))
        save(fig, f"prio_c_deciles_{'nominal' if dl == 0 else 'deadline100'}")

if __name__ == '__main__':
    eps = load(); print('episodes', len(eps))
    counts = tuple(n for n in (8, 16, 32) if any(e['n'] == n for e in eps))
    ra, rb, rc = [], [], []
    fig_a(eps, counts, ra); write_csv('prio_a_quartiles', ra); print('ok a')
    fig_b(eps, rb); write_csv('prio_b_cdf_n16', rb); print('ok b')
    fig_c(eps, counts, rc); write_csv('prio_c_deciles', rc); print('ok c')
