#!/workspace/SANGIN-dev/.venv/bin/python
"""Fig 5b/5c: per-N command-latency CDF + log-CCDF panels (replaces p95-only bars).

Read-only over runs/main/*/result.json; safe to run while the study is live:
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/fig5b_latency_by_n.py
Re-run any time (e.g. after the study finishes) to refresh figures/fig5b_*, fig5c_*.
Undelivered commands (cycle_latency_seconds is None) are counted as > all times,
so the CDF tops out below 1 and the CCDF floors at the undelivered fraction.
"""
import json, csv, math, re
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY = Path('/workspace/leo_study'); FIG = STUDY / 'figures'; RUNS = STUDY / 'runs'
CUTOFF = .25; INTERVAL_REPO = .35; INTERVAL_PAPER = 1.0
NS = (8, 16, 32)
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

# (curve label, job-id prefix template, colour)
ROUTES = [('Allocator', 'nominal__A4__allocated__physical__dl0__n{n}__', 'C0'),
          ('All-UAV', 'routes__A4__edge__physical__dl0__n{n}__', 'C1'),
          ('All-cloud', 'routes__A4__cloud__physical__dl0__n{n}__', 'C2')]
DELAYS = [('Nominal LEO', 'nominal__A4__allocated__physical__dl0__n{n}__', 'C0'),
          ('+150 ms downlink', 'delay__A4__allocated__physical__dl150__n{n}__', 'C3'),
          ('+300 ms downlink', 'delay__A4__allocated__physical__dl300__n{n}__', 'C4')]


def planned_counts():
    """Episodes planned per job-id prefix (from matrix.json)."""
    c = Counter()
    for j in json.load(open(RUNS / 'matrix.json')):
        c[re.sub(r'c\d+$', '', j['job_id'])] += 1
    return c


def load_cell(prefix):
    """Stream result.json files for one cell; keep only latencies + placement."""
    lat, loc, undeliv, eps = [], {}, 0, 0
    for d in sorted(RUNS.glob('main/' + prefix + 'c*')):
        f = d / 'result.json'
        if not f.exists():
            continue
        try:
            p = json.load(open(f))
        except (json.JSONDecodeError, OSError):  # mid-write; skip, next refresh picks it up
            continue
        eps += 1
        for t in p['tasks']:
            v = t.get('cycle_latency_seconds')
            if v is None:
                undeliv += 1
            else:
                lat.append(v); loc.setdefault(t.get('location'), []).append(v)
        del p
    return np.sort(np.asarray(lat, float)), undeliv, eps, loc


def stats(lat, undeliv):
    total = len(lat) + undeliv
    if total == 0:
        return {}
    # quantiles over all commands with undelivered = +inf
    full = np.concatenate([lat, np.full(undeliv, np.inf)])
    q = lambda p: float(np.quantile(full, p, method='inverted_cdf')) * 1000
    return dict(n_commands=total, delivered=len(lat), undelivered=undeliv,
                p50_ms=q(.5), p95_ms=q(.95), p99_ms=q(.99),
                pr_le_250ms=float(np.sum(lat <= CUTOFF)) / total,
                pr_le_350ms=float(np.sum(lat <= INTERVAL_REPO)) / total,
                pr_le_1s=float(np.sum(lat <= INTERVAL_PAPER)) / total)


def make_figure(curves, name, title, recs, planned, figure_tag):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    xmax = 0; cells_note = []
    data = {}
    for n in NS:
        for label, tmpl, color in curves:
            pre = tmpl.format(n=n)
            lat, und, eps, loc = load_cell(pre)
            data[(n, label)] = (lat, und, eps, color)
            if len(lat): xmax = max(xmax, float(np.quantile(lat, .999)))
            s = stats(lat, und)
            recs.append(dict(figure=figure_tag, num_agents=n, curve=label, n_episodes=eps,
                             planned_episodes=planned.get(pre, 0),
                             **({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} if s else dict(n_commands=0)),
                             frac_drone_placed=round(len(loc.get('drone', [])) / max(1, len(lat)), 4),
                             p50_drone_ms=round(float(np.median(loc['drone'])) * 1000, 2) if loc.get('drone') else None,
                             p50_cloud_ms=round(float(np.median(loc['cloud'])) * 1000, 2) if loc.get('cloud') else None))
    xmax_ms = max(1100, min(xmax * 1000 * 1.05, 5000))
    totals = {label: (sum(len(data[(n, label)][0]) + data[(n, label)][1] for n in NS),
                      sum(data[(n, label)][1] for n in NS)) for label, _, _ in curves}
    for ci, n in enumerate(NS):
        top, bot = axes[0, ci], axes[1, ci]
        minfloor = 1
        for label, tmpl, color in curves:
            lat, und, eps, _ = data[(n, label)]
            total = len(lat) + und
            if total == 0:
                continue
            cdf = np.arange(1, len(lat) + 1) / total
            x = np.concatenate([[0], lat * 1000, [xmax_ms]]); y = np.concatenate([[0], cdf, [cdf[-1] if len(cdf) else 0]])
            top.step(x, y, where='post', color=color, lw=1.4)
            ccdf = 1 - y
            if und == 0:   # end at the last sample instead of drawing a fake plateau
                keep = ccdf[:-1] > 0
                xs, ys = x[:-1][keep], ccdf[:-1][keep]
                if len(lat): xs = np.append(xs, lat[-1] * 1000); ys = np.append(ys, ys[-1] if len(ys) else 1)
                bot.step(xs, ys, where='post', color=color, lw=1.4)
                minfloor = min(minfloor, 1 / total)
            else:          # undelivered = T beyond any t: real plateau at undelivered share
                bot.step(x, ccdf, where='post', color=color, lw=1.4)
                minfloor = min(minfloor, und / total)
        for ax in (top, bot):
            ax.axvline(CUTOFF * 1000, color='k', ls='--', lw=.9)
            ax.axvline(INTERVAL_REPO * 1000, color='0.35', ls=':', lw=1.1)
            ax.axvline(INTERVAL_PAPER * 1000, color='0.4', ls='-.', lw=1.0)
            ax.set_xlim(0, xmax_ms)
        top.set_title(f'N = {n}', fontsize=9, fontweight='bold')
        top.set_ylim(0, 1.02)
        bot.set_yscale('log'); bot.set_ylim(max(minfloor / 2, 1e-5), 1.5)
        if ci:
            top.set_yticklabels([]); bot.tick_params(labelleft=True)
    axes[0, 0].set_ylabel('Pr(T ≤ t)')
    axes[1, 0].set_ylabel('Pr(T > t)  (log scale)')
    
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.6, label=f'{lab} (n={totals[lab][0]:,})') for lab, _, c in curves if totals[lab][0]]
    handles += [Line2D([], [], color='k', ls='--', lw=.9, label='250 ms cutoff'),
                Line2D([], [], color='0.35', ls=':', lw=1.1, label='350 ms interval'),
                Line2D([], [], color='0.4', ls='-.', lw=1.0, label='1 s interval')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=7.5, bbox_to_anchor=(.5, .025),
               handlelength=2.2, columnspacing=1.4)
    notes = []
    mine = [r for r in recs if r['figure'] == figure_tag]
    done = sum(r['n_episodes'] for r in mine); plan = sum(r['planned_episodes'] for r in mine)
    if done < plan:
        per = ', '.join(f"{lab} " + '/'.join(str(r['n_episodes']) for r in mine if r['curve'] == lab) for lab, _, _ in curves)
        notes.append(f'Partial run: {done}/{plan} episodes ({per} at N = 8/16/32).')
    und_tot = sum(v[1] for v in totals.values())
    if und_tot:
        notes.append(f'{und_tot} undelivered commands counted as T > t.')
    notes.append('n = commands over all N.')
    fig.text(.5, .005, '  '.join(notes), ha='center', va='bottom', fontsize=6.5, color='0.35')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .97))
    fig.text(.5, .085, 'Command latency t (ms)', ha='center', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    planned = planned_counts(); recs = []
    make_figure(ROUTES, 'fig5b_latency_cdf_ccdf_by_n',
                'Command latency by route (A4, nominal LEO)',
                recs, planned, 'fig5b')
    recs_c = []
    has_delay = any(any((RUNS / 'main').glob(t.format(n=n) + 'c*/result.json')) for _, t, _ in DELAYS[1:] for n in NS)
    if has_delay:
        make_figure(DELAYS, 'fig5c_latency_cdf_ccdf_by_delay',
                    'Command latency vs added delay (allocator, A4)', recs_c, planned, 'fig5c')
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(FIG / 'fig5b_latency_by_n.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)
    if recs_c:
        with open(FIG / 'fig5c_latency_by_delay.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs_c)
    for r in recs + recs_c:
        print({k: r.get(k) for k in ('figure', 'num_agents', 'curve', 'n_episodes', 'planned_episodes', 'n_commands', 'undelivered',
                                     'p50_ms', 'p95_ms', 'p99_ms', 'pr_le_250ms', 'frac_drone_placed', 'p50_drone_ms', 'p50_cloud_ms')})


if __name__ == '__main__':
    main()
