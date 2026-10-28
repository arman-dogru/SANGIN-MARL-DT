#!/workspace/SANGIN-dev/.venv/bin/python
"""Priority vs command latency in Arman's omega-sensitivity run (read-only over
SANGIN-dev/results/omega_sensitivity/tasks.csv.gz, the published completed-mission bundle).
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/omega_prio_figs.py
Priority = allocator weight (1..4, 4 = closest-to-goal active robot). Rows exist only for active robots.
Per (episode, cycle) robots are ranked by weight (descending, ties -> mid-rank), normalised rank r = rank/(n_active-1);
high = r <= .25, low = r >= .75 (same convention as prio_figs.py). Cycles with one active robot are dropped.
Alternative check: weight bins, high = weight >= 3.25, low = weight <= 1.75.
Statistics as prio_figs.py: per-episode p50/p95, mean over episodes with 95% case bootstrap; cutoff miss pooled
(latency > 250 ms); undelivered commands would count as +inf (none occur in this bundle).
Gap = mean over episodes of (high - low) per-episode quantile, paired case bootstrap.
"""
import csv, gzip, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/leo_study/harness')
import paper_figs as PF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = Path('/workspace/SANGIN-dev/results/omega_sensitivity/tasks.csv.gz')
STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/priority/omega'; OUT.mkdir(parents=True, exist_ok=True)
CACHE = STUDY / 'tmp/omega_tasks_cache.npz'
RNG = np.random.default_rng(20261027); B = 2000; CUTOFF_MS = 250.
NS = (2, 4, 8, 16, 32)
ROUTES = (('allocated', 'allocator'), ('cloud', 'all-cloud'), ('edge', 'all-UAV'))
WS = ('W1', 'W2', 'W3', 'W4', 'W5')
W_C = {'W1': '#000000', 'W2': '#56B4E9', 'W3': '#009E73', 'W4': '#E69F00', 'W5': '#D55E00'}
W_LAB = {}

def load():
    if CACHE.exists() and CACHE.stat().st_mtime > SRC.stat().st_mtime:
        z = np.load(CACHE, allow_pickle=False); return {k: z[k] for k in z.files}
    cols = defaultdict(list); deltas, cuts = set(), set()
    for t in csv.DictReader(gzip.open(SRC, 'rt')):
        lat = t['observed_latency_seconds']; deliv = t['command_delivery_status'] == 'delivered'
        cols['w'].append(t['weight_id']); cols['route'].append(t['route']); cols['n'].append(int(t['num_agents']))
        cols['case'].append(t['case_id']); cols['cycle'].append(int(t['cycle'])); cols['agent'].append(int(t['agent']))
        cols['weight'].append(float(t['weight'])); cols['lat'].append(float(lat) * 1000 if (lat and deliv) else math.inf)
        cols['lc'].append(float(t['latency_coefficient'])); cols['ec'].append(float(t['energy_coefficient']))
        cuts.add(round(float(t['command_cutoff_at']) - float(t['source_time']) if t['source_time'] else float('nan'), 6))
        deltas.add(round(float(t['source_time']) / int(t['cycle']), 6) if int(t['cycle']) else None)
    d = {k: np.array(v) for k, v in cols.items()}
    d['cutoffs'] = np.array(sorted(c for c in cuts if c == c)); d['deltas'] = np.array(sorted(x for x in deltas if x is not None))
    CACHE.parent.mkdir(exist_ok=True); np.savez_compressed(CACHE, **d); return d

def ranks(d):
    """Normalised mid-rank by weight (0 = highest priority) within each (w, route, n, case, cycle)."""
    key = np.char.add(np.char.add(np.char.add(d['w'], d['route']), np.char.add(d['n'].astype(str), d['case'])), d['cycle'].astype(str))
    _, inv = np.unique(key, return_inverse=True); r = np.full(len(inv), np.nan)
    order = np.argsort(inv, kind='stable'); bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for g in np.split(order, bounds):
        m = len(g)
        if m < 2: continue
        w = -d['weight'][g]; srt = np.argsort(w, kind='stable'); rk = np.empty(m); rk[srt] = np.arange(m)
        for v in np.unique(w):                      # ties -> mid-rank
            t = w == v
            if t.sum() > 1: rk[t] = rk[t].mean()
        r[g] = rk / (m - 1)
    return r

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

def episodes(d, masks):
    """{(w, route, n): [ {case, high: lat array, low: lat array}, ... ]} for a given grouping."""
    hi_m, lo_m = masks; out = defaultdict(dict)
    for i in np.flatnonzero(hi_m | lo_m):
        k = (d['w'][i], d['route'][i], int(d['n'][i])); c = d['case'][i]
        e = out[k].setdefault(c, {'high': [], 'low': []}); e['high' if hi_m[i] else 'low'].append(d['lat'][i])
    return {k: [{'case': c, 'high': np.array(v['high']), 'low': np.array(v['low'])} for c, v in sorted(cs.items())] for k, cs in out.items()}

def cell_stats(eps):
    s = {}
    for g in ('high', 'low'):
        gs = [e[g] for e in eps]; ne = [x for x in gs if len(x)]
        s[g] = dict(p50=boot_mean([q(x, .5) for x in ne]), p95=boot_mean([q(x, .95) for x in ne]),
                    miss=boot_ratio([np.sum(x > CUTOFF_MS) for x in gs], [len(x) for x in gs]),
                    commands=int(sum(len(x) for x in gs)), undelivered=int(sum(np.isinf(x).sum() for x in gs)))
    both = [e for e in eps if len(e['high']) and len(e['low'])]
    for p, nm in ((.5, 'gap_p50'), (.95, 'gap_p95')): s[nm] = boot_mean([q(e['high'], p) - q(e['low'], p) for e in both])
    s['gap_miss'] = boot_mean([100 * (np.mean(e['high'] > CUTOFF_MS) - np.mean(e['low'] > CUTOFF_MS)) for e in both])
    s['episodes'] = len(eps); return s

def save(fig, name):
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

def band(ax, xs, est, lo, hi, color, ls, open_, label=None, fill=True):
    xs, est, lo, hi = map(np.asarray, (xs, est, lo, hi)); ok = np.isfinite(est)
    if not ok.any(): return
    if fill: ax.fill_between(xs[ok], lo[ok], hi[ok], color=color, alpha=.13, lw=0)
    ax.plot(xs[ok], est[ok], color=color, ls=ls, lw=2.4, marker='o', ms=7.5, mfc='white' if open_ else color, mec=color, mew=1.8, label=label)

def xaxis(ax, ns):
    ax.set_xscale('log', base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_xlim(ns[0] / 1.35, ns[-1] * 1.35)

def all_robot_stats(d):
    """All-robot (unsplit) per-episode stats for every (w, route, n): all active robots' commands."""
    out = defaultdict(lambda: defaultdict(list))
    for i in range(len(d['w'])): out[(d['w'][i], d['route'][i], int(d['n'][i]))][d['case'][i]].append(d['lat'][i])
    S = {}
    for k, cs in out.items():
        gs = [np.array(v) for _, v in sorted(cs.items())]
        S[k] = dict(p50=boot_mean([q(x, .5) for x in gs]), p95=boot_mean([q(x, .95) for x in gs]),
                    miss=boot_ratio([np.sum(x > CUTOFF_MS) for x in gs], [len(x) for x in gs]),
                    commands=int(sum(len(x) for x in gs)), undelivered=int(sum(np.isinf(x).sum() for x in gs)), episodes=len(gs))
    return S

LINES = (('allocated', 'high', 'allocator, high priority (top 25%)', '-', False),
         ('allocated', 'low', 'allocator, low priority (bottom 25%)', '--', True),
         ('cloud', 'all', 'all-cloud (all robots)', '-', False),
         ('edge', 'all', 'all-UAV (all robots)', '-', False))
COLOR = {'allocated': PF.ROUTE_C['allocator'], 'cloud': PF.ROUTE_C['all-cloud'], 'edge': PF.ROUTE_C['all-UAV']}

def get(S, A, w, route, g, n):
    if g == 'all': return A.get((w, route, n))
    s = S.get((w, route, n)); return None if s is None else s[g]

def cutoff_line(ax):
    ax.axhline(CUTOFF_MS, color='0.3', ls=':', lw=1.4); ax.text(NS[0] / 1.3, CUTOFF_MS + 5, '250 ms cutoff', fontsize=11, color='0.3', va='bottom')

def fig_main(S, A, recs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, metric, ylab in ((axes[0], 'p50', 'Median command latency (ms)'), (axes[1], 'p95', 'p95 command latency (ms)'),
                             (axes[2], 'miss', 'Commands over 250 ms cutoff (%)')):
        for route, g, lab, ls, op in LINES:
            ns = [n for n in NS if get(S, A, 'W1', route, g, n) is not None]
            vals = [get(S, A, 'W1', route, g, n)[metric] for n in ns]
            if metric == 'miss': vals = [tuple(100 * x for x in v) for v in vals]
            band(ax, ns, *zip(*vals), COLOR[route], ls, op)
        xaxis(ax, list(NS)); ax.set_xlabel('Number of robots N'); ax.set_ylabel(ylab)
        if metric != 'miss': ax.set_ylim(0, 400 if metric == 'p95' else 300); cutoff_line(ax)
        else: ax.set_ylim(bottom=-1)
    h = [Line2D([], [], color=COLOR[r], lw=2.4, ls=ls, marker='o', ms=8, mfc='white' if op else COLOR[r], mew=1.8, label=lab) for r, g, lab, ls, op in LINES]
    fig.tight_layout(rect=(0, 0, 1, .88)); fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .88), ncol=4, fontsize=12)
    save(fig, 'omega_prio_latency_vs_n_W1')
    for (w, route, n), s in sorted(S.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        if route != 'allocated': continue
        for g in ('high', 'low'):
            recs.append(dict(weight_id=w, latency_coefficient=W_LAB[w][0], energy_coefficient=W_LAB[w][1], route='allocator',
                             num_agents=n, group=g, episodes=s['episodes'], commands=s[g]['commands'], undelivered=s[g]['undelivered'],
                             p50_ms=s[g]['p50'][0], p50_ci_low=s[g]['p50'][1], p50_ci_high=s[g]['p50'][2],
                             p95_ms=s[g]['p95'][0], p95_ci_low=s[g]['p95'][1], p95_ci_high=s[g]['p95'][2],
                             cutoff_miss_pct=100 * s[g]['miss'][0], miss_ci_low_pct=100 * s[g]['miss'][1], miss_ci_high_pct=100 * s[g]['miss'][2],
                             gap_p50_ms=s['gap_p50'][0], gap_p50_ci_low=s['gap_p50'][1], gap_p50_ci_high=s['gap_p50'][2],
                             gap_p95_ms=s['gap_p95'][0], gap_p95_ci_low=s['gap_p95'][1], gap_p95_ci_high=s['gap_p95'][2],
                             gap_miss_pp=s['gap_miss'][0]))
    for (w, route, n), s in sorted(A.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        recs.append(dict(weight_id=w, latency_coefficient=W_LAB[w][0], energy_coefficient=W_LAB[w][1], route=dict(ROUTES)[route],
                         num_agents=n, group='all robots', episodes=s['episodes'], commands=s['commands'], undelivered=s['undelivered'],
                         p50_ms=s['p50'][0], p50_ci_low=s['p50'][1], p50_ci_high=s['p50'][2],
                         p95_ms=s['p95'][0], p95_ci_low=s['p95'][1], p95_ci_high=s['p95'][2],
                         cutoff_miss_pct=100 * s['miss'][0], miss_ci_low_pct=100 * s['miss'][1], miss_ci_high_pct=100 * s['miss'][2]))

N_C = dict(zip(NS, plt.cm.Purples(np.linspace(.3, 1.0, len(NS)))))   # sequential, distinct from route colours
GROUPS = (('high', 'high priority\n(top 25%)'), ('mid', 'middle 50%'), ('low', 'low priority\n(bottom 25%)'))

def fig_bars(S, M, recs):
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.4), sharey='row')
    wbar = .16
    for col, w in enumerate(WS):
        for row, metric, ylab, top in ((0, 'p50', 'Median command latency (ms)', 300), (1, 'p95', 'p95 command latency (ms)', 400)):
            ax = axes[row, col]
            for gi, (g, glab) in enumerate(GROUPS):
                for ni, n in enumerate(NS):
                    if (w, 'allocated', n) not in S or (g == 'mid' and (w, 'allocated', n) not in M): continue   # N=2 has no middle group
                    est, lo, hi = (M[(w, 'allocated', n)]['high'] if g == 'mid' else S[(w, 'allocated', n)][g])[metric]
                    if not np.isfinite(est): continue
                    x = gi + (ni - 2) * wbar
                    ax.bar(x, est, wbar * .92, color=N_C[n], edgecolor='0.15', lw=.6)
                    ax.errorbar(x, est, yerr=[[max(est - lo, 0)], [max(hi - est, 0)]], color='0.1', capsize=2.5, lw=1.1, ls='none')
                    if row == 0:
                        src = M[(w, 'allocated', n)]['high'] if g == 'mid' else S[(w, 'allocated', n)][g]
                        recs.append(dict(weight_id=w, num_agents=n, group=g, episodes=S[(w, 'allocated', n)]['episodes'], commands=src['commands'],
                                         p50_ms=src['p50'][0], p50_ci_low=src['p50'][1], p50_ci_high=src['p50'][2],
                                         p95_ms=src['p95'][0], p95_ci_low=src['p95'][1], p95_ci_high=src['p95'][2], cutoff_miss_pct=100 * src['miss'][0]))
            ax.axhline(CUTOFF_MS, color='0.3', ls=':', lw=1.4); ax.set_ylim(0, top)
            ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([gl for _, gl in GROUPS] if row == 1 else [''] * len(GROUPS), fontsize=11)
            ax.set_xlim(-.55, len(GROUPS) - .45)
            if col == 0: ax.set_ylabel(ylab); ax.text(-.5, CUTOFF_MS + 5, '250 ms cutoff', fontsize=10.5, color='0.3', va='bottom')
            if row == 0: ax.set_title(f'ω₁:ω₂ = {W_LAB[w][0]:g}:{W_LAB[w][1]:g}' + (' (default)' if w == 'W1' else ''), fontsize=13)   # ω₁ = latency coeff, ω₂ = energy coeff
    from matplotlib.patches import Patch
    h = [Patch(facecolor=N_C[n], edgecolor='0.15', label=f'N = {n}') for n in NS]
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .93), ncol=5, fontsize=12.5)
    save(fig, 'omega_prio_bars_per_omega_allocator')

def write_csv(name, recs):
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)

if __name__ == '__main__':
    d = load(); print('rows', len(d['w']), 'cutoff (s) per row:', d['cutoffs'], 'delta (s):', d['deltas'])
    for w in WS:
        i = np.flatnonzero(d['w'] == w)[0]; W_LAB[w] = (float(d['lc'][i]), float(d['ec'][i]))
    r = ranks(d)
    S = {k: cell_stats(v) for k, v in episodes(d, (r <= .25, r >= .75)).items()}
    Sw = {k: cell_stats(v) for k, v in episodes(d, (np.isfinite(r) & (d['weight'] >= 3.25), np.isfinite(r) & (d['weight'] <= 1.75))).items()}
    A = all_robot_stats(d); recs = []; fig_main(S, A, recs); write_csv('omega_prio_summary_quartiles', recs)
    rm = np.isfinite(r) & (r > .25) & (r < .75)
    M = {k: cell_stats(v) for k, v in episodes(d, (rm, np.zeros_like(rm))).items()}   # middle 50% stored under 'high'
    rb = []; fig_bars(S, M, rb); write_csv('omega_prio_bars_per_omega_allocator', rb)
    agree = []
    for k in sorted(S, key=lambda k: (k[1], k[0], k[2])):
        a, b = S[k], Sw.get(k)
        if b is None: continue
        agree.append(dict(weight_id=k[0], route=dict(ROUTES)[k[1]], num_agents=k[2], episodes=a['episodes'],
                          q_high_p50=a['high']['p50'][0], q_low_p50=a['low']['p50'][0], q_gap_p50=a['gap_p50'][0], q_gap_p95=a['gap_p95'][0],
                          w_high_p50=b['high']['p50'][0], w_low_p50=b['low']['p50'][0], w_gap_p50=b['gap_p50'][0], w_gap_p95=b['gap_p95'][0],
                          q_high_cmds=a['high']['commands'], q_low_cmds=a['low']['commands'], w_high_cmds=b['high']['commands'], w_low_cmds=b['low']['commands'],
                          sign_agree_p50=bool(np.sign(round(a['gap_p50'][0], 1)) == np.sign(round(b['gap_p50'][0], 1))),
                          sign_agree_p95=bool(np.sign(round(a['gap_p95'][0], 1)) == np.sign(round(b['gap_p95'][0], 1)))))
    write_csv('omega_prio_quartile_vs_weightbin_check', agree)
    print('ok')
