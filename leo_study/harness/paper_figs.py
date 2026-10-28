#!/workspace/SANGIN-dev/.venv/bin/python
"""Paper-ready replots of fig2, fig4, fig5b, fig6, fig7, fig8 (readability pass, 2026-10-27).

Read-only over runs/main/*/result.json; keeps only the fields it needs per episode (low memory).
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/paper_figs.py
Style: Okabe-Ito colorblind-safe palette, fixed colors per route / delay / policy,
same speed = open (white-filled) marks, mixed speed = solid; partial cells (n<10) faded + hatched.
"""
import json, glob, math, re, sys, csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

STUDY = Path('/workspace/leo_study'); FIG = STUDY / 'figures'; RUNS = STUDY / 'runs'
sys.path.insert(0, str(STUDY / 'harness'))
from analyze import wilson, boot_mean, boot_ratio  # noqa: E402

EXPECTED = 10; CUTOFF = .25
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 12, 'figure.titlesize': 15,
                     'axes.grid': True, 'grid.alpha': .25, 'axes.grid.axis': 'y',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
# Okabe-Ito
ROUTE_C = {'allocator': '#0072B2', 'all-cloud': '#009E73', 'all-UAV': '#E69F00'}
ROUTE_KEY = {'allocator': ('nominal', 'allocated'), 'all-cloud': (None, 'cloud'), 'all-UAV': (None, 'edge')}
DELAY_C = {'ideal network': '#000000', 'nominal LEO': '#56B4E9', '100 ms deadline': '#CC79A7', 'real time': '#D55E00'}
POLICY_C = {'WAIT': '#999999', 'cached fallback': '#0072B2'}
NO_SUCCESS = '--no-success' in sys.argv  # variant without success rates / completion times
OUT = FIG / 'no_success' if NO_SUCCESS else FIG; SUF = '_ns' if NO_SUCCESS else ''
FOOT = 'Hatched/faded = partial cell (fewer than 10 episodes; n shown).'


# ---------------- data ----------------
def load():
    rows = []
    for f in glob.glob(str(RUNS / 'main/*/result.json')):
        try: p = json.load(open(f))
        except (json.JSONDecodeError, OSError): continue
        j = p['job']; x = dict(job=j, row=p['row'], pair=p['pair_stats'])
        if j['treatment'] == 'A4' and j['network'] == 'physical' and j.get('downlink_seconds', 0) == 0:
            x['lat'] = [t.get('cycle_latency_seconds') for t in p['tasks']]
        rows.append(x); del p
    return rows

def sel(rows, **kw):
    return [x for x in rows if all(x['job'].get(k) == v for k, v in kw.items())]

def route_rows(rows, route, treatment, n):
    if route == 'allocator': return sel(rows, group='nominal', treatment=treatment, route='allocated', num_agents=n)
    grp = 'routes' if treatment == 'A4' else 'routes_speed'
    return sel(rows, group=grp, treatment=treatment, route=ROUTE_KEY[route][1], num_agents=n)

def delay_rows(rows, cond, treatment, n):
    if cond == 'ideal network': return sel(rows, group='baseline', treatment=treatment, num_agents=n)
    if cond == 'nominal LEO': return sel(rows, group='nominal', treatment=treatment, num_agents=n)
    d = .15 if cond == '100 ms deadline' else .3
    return sel(rows, group='delay', treatment=treatment, downlink_seconds=d, num_agents=n)

def trimmed_reason(job):
    rules = json.loads((RUNS / 'trim.json').read_text())['rules']
    for r in rules:
        if all(job.get(k) == v if not isinstance(v, list) else job.get(k) in v for k, v in r['match'].items()): return True
    return False

def missing_label(group, treatment, n, dl=0.0):
    """'trimmed' if the cell is in trim.json, else 'running' (cell not finished yet)."""
    job = dict(group=group, treatment=treatment, num_agents=n, downlink_seconds=dl)
    return 'trimmed' if trimmed_reason(job) else 'pending'

def write_csv(name, recs):
    keys = list(dict.fromkeys(k for r in recs for k in r))
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name.split('_')[0] if NO_SUCCESS else name}{SUF}.csv", 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(recs)

def save(fig, name, foot=None):
    titled = [ax for ax in fig.axes if ax.get_title()]  # panel letters (a), (b), (c) on multi-panel figures
    if len(titled) > 1:
        for i, ax in enumerate(titled): ax.set_title(f'({chr(97 + i)}) {ax.get_title()}')
    if False and foot: fig.text(.01, -.01, foot, ha='left', va='top', fontsize=10.5, color='0.35', wrap=True)
    OUT.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f"{name.split('_')[0] if NO_SUCCESS else name}{SUF}.{ext}", bbox_inches='tight', dpi=300)
    plt.close(fig)

def bar(ax, x, k, n, color, width, open_=False, label=None, ylim_top=1.0, fmt='{k}/{n}', note=None):
    """Success bar with Wilson CI and k/n label; partial = faded + hatched; empty = text note."""
    if n == 0:
        return  # no data: cell omitted
    p, lo, hi = wilson(k, n); partial = n < EXPECTED
    face = 'white' if open_ else color
    ax.bar(x, p, width, color=face, edgecolor=color, linewidth=1.8, alpha=1,
           hatch=None, label=label)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color='0.2', capsize=3, lw=1.1)
    ax.text(x, min(hi, ylim_top) + .03, fmt.format(k=k, n=n), ha='center', va='bottom', rotation=90, fontsize=10.5)

def dot(ax, x, est, lo, hi, color, open_=False, n=EXPECTED, marker='o', ms=8, label=None, top=None, rot=0):
    if est is None or (isinstance(est, float) and math.isnan(est)): return
    partial = n < EXPECTED
    if top is not None and est > top:  # off-scale: arrow at the top edge with the value written out
        ax.plot(x, top * .97, marker='^', ms=ms + 2, color=color, alpha=1, ls='none', clip_on=False)
        lab = f'{est/1000:.1f} s' if est >= 1000 else f'{est:.0f}'
        ax.annotate(lab + (f' (n={n})' if partial else ''), (x, top * .97), textcoords='offset points', xytext=(0, -16),
                    ha='center', va='top', fontsize=10, color='0.25', rotation=90)
        return
    ax.errorbar(x, est, yerr=[[max(est - lo, 0)], [max(hi - est, 0)]], color=color, capsize=4, lw=1.6,
                marker=marker, ms=ms, mfc='white' if open_ else color, mec=color, mew=1.8,
                alpha=1, ls='none', label=label)
    if partial: ax.annotate(f'n={n}', (x, hi), textcoords='offset points', xytext=(0, 4), ha='center', va='bottom', fontsize=10, color='0.3', rotation=rot)


def top(fig, handles, ncol, title=None):
    """Panel layout with one shared legend above the panels and an optional figure title above it."""
    fig.tight_layout(rect=(0, 0, 1, .84 if title else .9))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .84 if title else .9), ncol=ncol)
    if title: fig.suptitle(title, y=.995)


# ---------------- Fig 8: same vs mixed speed per route ----------------
ROUTES = ('allocator', 'all-cloud', 'all-UAV')
SPEEDS = (('same speed', 'A3', True), ('mixed speed', 'A4', False))

def route_speed_legend(fig, ncol=3, y=1.0):
    h = [Patch(facecolor=ROUTE_C[r], edgecolor=ROUTE_C[r], label=r) for r in ROUTES]
    h += [Patch(facecolor='white', edgecolor='0.2', lw=1.5, label='same speed (open)'),
          Patch(facecolor='0.2', edgecolor='0.2', label='mixed speed (solid)')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, y), ncol=5, fontsize=12)

def fig8(rows):
    NS = (8, 16, 32); recs = []
    # 8a success
    fig, ax = plt.subplots(figsize=(11, 4.8)); w = .13
    for gi, n in enumerate(NS):
        for si, (sname, tr, op) in enumerate(SPEEDS):
            for ri, r in enumerate(ROUTES):
                sub = route_rows(rows, r, tr, n); k = sum(x['row']['success'] for x in sub)
                x = gi + (si * 3 + ri - 2.5) * w
                grp = 'nominal' if r == 'allocator' else ('routes' if tr == 'A4' else 'routes_speed')
                bar(ax, x, k, len(sub), ROUTE_C[r], w * .92, open_=op)
                p, lo, hi = wilson(k, len(sub)) if sub else (math.nan,) * 3
                recs.append(dict(panel='success', route=r, speed=sname, num_agents=n, episodes=len(sub), successes=k, value=p, ci_low=lo, ci_high=hi))
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f'N = {n}' for n in NS])
    ax.set_ylim(0, 1.32); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_ylabel('Mission success (fraction)')
    ax.set_title('Mission success by route')
    h = [Patch(facecolor=ROUTE_C[r], edgecolor=ROUTE_C[r], label=r) for r in ROUTES]
    h += [Patch(facecolor='white', edgecolor='0.2', lw=1.5, label='same speed (open)'),
          Patch(facecolor='0.2', edgecolor='0.2', label='mixed speed (solid)')]
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    save(fig, 'fig8a_routes_success_by_speed', FOOT + '  N = 2, 4: 10/10 in every cell (not shown). Error bars: Wilson 95% CI. Cached fallback, nominal LEO.')
    # 8b completion time + p95 latency
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, ylab, fn) in zip(axes, (
            ('Completion time', 'Completion time, successful runs (s)', lambda s: boot_mean([x['row']['makespan_seconds'] for x in s if x['row']['success']])),
            ('p95 command latency', 'p95 command latency (ms)', lambda s: boot_mean([x['row']['task_latency_p95_seconds'] * 1000 for x in s if x['row'].get('task_latency_p95_seconds') is not None])))):
        for gi, n in enumerate(NS):
            for si, (sname, tr, op) in enumerate(SPEEDS):
                for ri, r in enumerate(ROUTES):
                    sub = route_rows(rows, r, tr, n)
                    if not sub: continue
                    est, lo, hi = fn(sub); x = gi + (si * 3 + ri - 2.5) * .13
                    dot(ax, x, est, lo, hi, ROUTE_C[r], open_=op, n=len(sub), marker='o' if op else 's', top=300 if 'latency' in name else None)
                    recs.append(dict(panel=name, route=r, speed=sname, num_agents=n, episodes=len(sub), value=est, ci_low=lo, ci_high=hi))
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([f'N = {n}' for n in NS]); ax.set_xlim(-.6, len(NS) - .4)
        ax.set_ylabel(ylab); ax.set_title(name)
    axes[1].axhline(250, color='0.3', ls='--', lw=1.2); axes[1].text(-.55, 255, '250 ms cutoff', fontsize=11, color='0.3', va='bottom')
    axes[0].set_ylim(bottom=0); axes[1].set_ylim(0, 300)
    h = [Line2D([], [], color=ROUTE_C[r], marker='s', ls='none', ms=9, label=r) for r in ROUTES]
    h += [Line2D([], [], color='0.2', marker='o', mfc='white', ls='none', ms=9, mew=1.8, label='same speed (open)'),
          Line2D([], [], color='0.2', marker='s', ls='none', ms=9, label='mixed speed (solid)')]
    top(fig, h, 5)
    save(fig, 'fig8b_routes_time_latency_by_speed', FOOT + '  Mean with 95% case-bootstrap CI. Faded = partial cell.')
    write_csv('fig8_routes_by_speed', recs)
    # legacy filename: the success figure is the headline
    import shutil
    for ext in ('png', 'pdf'): shutil.copyfile(FIG / f'fig8a_routes_success_by_speed.{ext}', FIG / f'fig8_routes_by_speed.{ext}')


# ---------------- Fig 6: cached fallback vs WAIT ----------------
def fig6(rows):
    recs = []; NS = (8, 16, 32); DL = (('nominal LEO', 0.0), ('100 ms deadline', .15), ('real time', .3))
    for tag, speed, (tw, tc) in (('a', 'mixed speed', ('A2', 'A4')), ('b', 'same speed', ('A1', 'A3'))):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True); w = .36
        for ax, n in zip(axes, NS):
            for di, (dlab, dl) in enumerate(DL):
                for pi, (pol, tr) in enumerate((('WAIT', tw), ('cached fallback', tc))):
                    sub = delay_rows(rows, dlab, tr, n); k = sum(x['row']['success'] for x in sub)
                    grp = 'nominal' if dl == 0 else 'delay'
                    bar(ax, di + (pi - .5) * w, k, len(sub), POLICY_C[pol], w * .92)
                    p, lo, hi = wilson(k, len(sub)) if sub else (math.nan,) * 3
                    recs.append(dict(speed=speed, policy=pol, treatment=tr, condition=dlab, num_agents=n,
                                     episodes=len(sub), successes=k, success=p, ci_low=lo, ci_high=hi,
                                     cutoff_waits_mean=np.mean([x['row']['scheduled_cutoff_waits'] for x in sub]) if sub else None,
                                     fallback_moves_mean=np.mean([x['row']['fallback_movements'] for x in sub]) if sub else None))
            ax.set_xticks(range(3)); ax.set_xticklabels([d.replace('nominal LEO', 'nominal\nLEO').replace('100 ms deadline', '100 ms\ndeadline') for d, _ in DL]); ax.set_title(f'N = {n}')
            ax.set_ylim(0, 1.32); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_xlim(-.6, 2.6); ax.set_xlabel('Network condition')
        axes[0].set_ylabel('Mission success (fraction)')
        top(fig, [Patch(color=POLICY_C['WAIT'], label='WAIT on missed command'),
                  Patch(color=POLICY_C['cached fallback'], label='cached-path fallback')], 2,
            'Cached fallback vs WAIT')
        save(fig, f'fig6{tag}_fallback_vs_wait_{speed.replace(" ", "_")}',
             f'{speed.capitalize()}. ' + FOOT + '  Wilson 95% CI.')
    write_csv('fig6_fallback_effectiveness', recs)
    import shutil
    for ext in ('png', 'pdf'): shutil.copyfile(FIG / f'fig6a_fallback_vs_wait_mixed_speed.{ext}', FIG / f'fig6_fallback_effectiveness.{ext}')


# ---------------- Fig 2: ordering errors by network condition ----------------
CONDS = ('ideal network', 'nominal LEO', '100 ms deadline', 'real time')

def fig2(rows):
    recs = []; NS = (8, 16, 32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8)); w = .19
    for ax, (title, num, den) in zip(axes, (('All robot pairs', 'ordering_errors', 'directed_pairs'),
                                             ('Field-of-view pairs', 'fov_errors', 'relevant_pairs'))):
        for gi, n in enumerate(NS):
            for ci, c in enumerate(CONDS):
                sub = delay_rows(rows, c, 'A4', n)
                if not sub: continue
                est, lo, hi = boot_ratio([x['row'][num] for x in sub], [x['row'][den] for x in sub])
                est, lo, hi = est * 100, lo * 100, hi * 100
                dot(ax, gi + (ci - 1.5) * w, est, lo, hi, DELAY_C[c], n=len(sub))
                recs.append(dict(panel=title, condition=c, num_agents=n, episodes=len(sub), errors=sum(x['row'][num] for x in sub),
                                 pairs=sum(x['row'][den] for x in sub), rate_pct=est, ci_low_pct=lo, ci_high_pct=hi))
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([f'N = {n}' for n in NS]); ax.set_xlim(-.6, len(NS) - .4)
        ax.set_title(title); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Priority-ordering errors\n(% of pair decisions)')
    fig.legend(handles=[Line2D([], [], color=DELAY_C[c], marker='o', ls='none', ms=9, label=c) for c in CONDS],
               loc='lower center', bbox_to_anchor=(.5, .9), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'fig2_ul_dl_e2e_errors', 'End-to-end errors (cached order vs true physical order), mixed speed + cached fallback. Mean, 95% case-bootstrap CI. '
         'Faded = partial (n shown).')
    write_csv('fig2_ul_dl_e2e_errors', recs)
    # 2b: where errors arise (UL vs DL vs E2E), N = 16
    recs2 = []; fig, ax = plt.subplots(figsize=(8, 4.6)); w = .19
    for gi, (lab, num) in enumerate((('Uplink\n(twin vs physical)', 'uplink_errors'), ('Downlink\n(cache vs twin)', 'downlink_errors'),
                                     ('End-to-end\n(cache vs physical)', 'ordering_errors'))):
        for ci, c in enumerate(CONDS):
            sub = delay_rows(rows, c, 'A4', 16)
            if not sub: continue
            est, lo, hi = boot_ratio([x['row'][num] for x in sub], [x['row']['directed_pairs'] for x in sub])
            dot(ax, gi + (ci - 1.5) * w, est * 100, lo * 100, hi * 100, DELAY_C[c], n=len(sub))
            recs2.append(dict(stage=lab.replace('\n', ' '), condition=c, num_agents=16, episodes=len(sub), rate_pct=est * 100, ci_low_pct=lo * 100, ci_high_pct=hi * 100))
    ax.set_xticks(range(3)); ax.set_xticklabels(['Uplink', 'Downlink', 'End-to-end'])
    ax.set_ylabel('Ordering errors\n(% of pair decisions)'); ax.set_title('Error source, N = 16'); ax.set_ylim(bottom=0)
    fig.legend(handles=[Line2D([], [], color=DELAY_C[c], marker='o', ls='none', ms=9, label=c) for c in CONDS],
               loc='lower center', bbox_to_anchor=(.5, .9), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'fig2b_error_stage_n16', 'All robot pairs, mixed speed + cached fallback. Mean, 95% case-bootstrap CI; faded = partial (n shown).')
    write_csv('fig2b_error_stage_n16', recs2)


# ---------------- Fig 4: conflict detection (corrected definition) ----------------
def conflict_counts(x):
    F = x['pair']['conflict_fields']; c = Counter()
    for rec in x['pair']['conflict']:
        d = dict(zip(F, rec)); k = d['count'] * d['active']
        if d['physical_relevant'] and not d['cached_relevant']: c['missed'] += k
        if d['physical_relevant']: c['phys'] += k
        if d['physical_relevant'] and d['cached_relevant']: c['hit'] += k
    c['active_pairs'] = sum(rec[-1] for rec in x['pair']['aoi'] if rec[1])
    return c

def fig4(rows):
    recs = []; NS = (8, 16, 32); w = .19
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cache = {}
    for ax, (title, ylab, fn) in zip(axes, (
            ('Missed conflicts', 'Missed conflicts\nper 1,000 active pair decisions',
             lambda cs: tuple(v for v in boot_ratio([1000 * c['missed'] for c in cs], [c['active_pairs'] for c in cs]))),
            ('Detection recall', 'Conflicts detected from cache\n(fraction of true conflicts)',
             lambda cs: boot_ratio([c['hit'] for c in cs], [c['phys'] for c in cs])))):
        for gi, n in enumerate(NS):
            for ci, c in enumerate(CONDS):
                sub = delay_rows(rows, c, 'A4', n)
                if not sub: continue
                cs = cache.setdefault((c, n), [conflict_counts(x) for x in sub])
                est, lo, hi = fn(cs); dot(ax, gi + (ci - 1.5) * w, est, lo, hi, DELAY_C[c], n=len(sub))
                recs.append(dict(metric=title, condition=c, num_agents=n, episodes=len(sub), value=est, ci_low=lo, ci_high=hi,
                                 missed_active=sum(x['missed'] for x in cs), true_conflicts_active=sum(x['phys'] for x in cs),
                                 active_pair_decisions=sum(x['active_pairs'] for x in cs)))
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([f'N = {n}' for n in NS]); ax.set_xlim(-.6, len(NS) - .4)
        ax.set_title(title); ax.set_ylabel(ylab)
    axes[0].set_ylim(bottom=0); axes[1].set_ylim(.25, 1.02)
    fig.legend(handles=[Line2D([], [], color=DELAY_C[c], marker='o', ls='none', ms=9, label=c) for c in CONDS],
               loc='lower center', bbox_to_anchor=(.5, .9), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'fig4_conflict_detection', 'Active (not-yet-finished) robot pairs only (corrected definition). Mixed speed + cached fallback. '
         'Mean, 95% case-bootstrap CI; faded = partial (n shown).')
    write_csv('fig4_conflict_detection', recs)


# ---------------- Fig 5b: latency tail by route ----------------
def fig5b(rows):
    recs = []; NS = (8, 16, 32)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, n in zip(axes, NS):
        for r in ROUTES:
            sub = route_rows(rows, r, 'A4', n)
            if not sub: continue
            vals = [v for x in sub for v in x.get('lat', [])]
            und = sum(v is None for v in vals); lat = np.sort(np.asarray([v for v in vals if v is not None])) * 1000
            tot = len(lat) + und; t = np.linspace(0, 400, 801)
            ccdf = 1 - np.searchsorted(lat, t, side='right') / tot
            ax.plot(t, np.maximum(ccdf, 1e-5), color=ROUTE_C[r], lw=2.4, alpha=1,
                    ls='-')
            p250 = float(np.sum(lat > 250) + und) / tot
            recs.append(dict(route=r, num_agents=n, episodes=len(sub), commands=tot, p50_ms=float(np.quantile(lat, .5)),
                             p95_ms=float(np.quantile(lat, .95)), p99_ms=float(np.quantile(lat, .99)), frac_over_250ms=p250))
        ax.axvline(250, color='0.3', ls='--', lw=1.2); ax.text(254, .3, '250 ms\ncutoff', fontsize=11, color='0.3')
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1.3); ax.set_xlim(0, 400); ax.set_title(f'N = {n}'); ax.set_xlabel('Command latency (ms)')
        ax.grid(True, which='major', axis='both', alpha=.25)
    axes[0].set_ylabel('Fraction of commands slower\nthan x (log scale)')
    fig.legend(handles=[Line2D([], [], color=ROUTE_C[r], lw=2.6, label=r) for r in ROUTES], loc='lower center', bbox_to_anchor=(.5, .9), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'fig5b_latency_cdf_ccdf_by_n', 'Complementary CDF of command latency (mixed speed, cached fallback, nominal LEO). '
         'Dashed/faded = partial.')
    write_csv('fig5b_latency_by_n', recs)


# ---------------- Fig 7: latency / energy / success by route ----------------
def fig7(rows):
    recs = []; NS = (2, 4, 8, 16, 32); w = .26
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, ylab, fn, scale) in zip(axes, (
            ('p95 command latency', 'p95 command latency (ms)', lambda s: boot_mean([x['row']['task_latency_p95_seconds'] for x in s if x['row'].get('task_latency_p95_seconds') is not None]), 1000),
            ('Energy per mission', 'Modeled comms + UAV energy\nper mission (J)', lambda s: boot_mean([x['row']['modeled_energy_lower_j'] for x in s]), 1))):
        for gi, n in enumerate(NS):
            for ri, r in enumerate(ROUTES):
                sub = route_rows(rows, r, 'A4', n)
                if not sub:
                    continue
                est, lo, hi = fn(sub); est, lo, hi = est * scale, lo * scale, hi * scale
                dot(ax, gi + (ri - 1) * w, est, lo, hi, ROUTE_C[r], n=len(sub), marker='s', top=300 if scale == 1000 else None)
                recs.append(dict(metric=name, route=r, num_agents=n, episodes=len(sub), value=est, ci_low=lo, ci_high=hi,
                                 **({} if NO_SUCCESS else dict(success=f"{sum(x['row']['success'] for x in sub)}/{len(sub)}"))))
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlabel('Number of robots N')
        ax.set_xlim(-.6, len(NS) - .4); ax.set_title(name); ax.set_ylabel(ylab); ax.set_ylim(bottom=0)
    axes[0].axhline(250, color='0.3', ls='--', lw=1.2); axes[0].text(-.5, 253, '250 ms cutoff', fontsize=11, color='0.3', va='bottom'); axes[0].set_ylim(0, 300)
    fig.legend(handles=[Line2D([], [], color=ROUTE_C[r], marker='s', ls='none', ms=9, label=r) for r in ROUTES], loc='lower center', bbox_to_anchor=(.5, .9), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'fig7_latency_energy_success', 'Mixed speed + cached fallback, nominal LEO. Mean, 95% case-bootstrap CI; faded = partial (n shown). '
         + ('Energy and latency over all episodes, failed ones included.' if NO_SUCCESS else 'Mission success by route: see Fig. 8a.'))
    write_csv('fig7_latency_energy_success', recs)


# ---------------- no-success variants (--no-success) ----------------
def fig8_ns(rows):
    """Fig 8b without completion time: p95 command latency only, same vs mixed speed per route (all episodes)."""
    NS_ = (8, 16, 32); recs = []
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for gi, n in enumerate(NS_):
        for si, (sname, tr, op) in enumerate(SPEEDS):
            for ri, r in enumerate(ROUTES):
                sub = route_rows(rows, r, tr, n); x = gi + (si * 3 + ri - 2.5) * .13
                if not sub:
                    continue
                est, lo, hi = boot_mean([x_['row']['task_latency_p95_seconds'] * 1000 for x_ in sub if x_['row'].get('task_latency_p95_seconds') is not None])
                dot(ax, x, est, lo, hi, ROUTE_C[r], open_=op, n=len(sub), marker='o' if op else 's', top=300)
                recs.append(dict(route=r, speed=sname, num_agents=n, episodes=len(sub), p95_latency_ms=est, ci_low=lo, ci_high=hi))
    ax.set_xticks(range(len(NS_))); ax.set_xticklabels([f'N = {n}' for n in NS_]); ax.set_xlim(-.6, len(NS_) - .4)
    ax.set_ylim(0, 300); ax.set_ylabel('p95 command latency (ms)'); ax.set_title('p95 command latency by route')
    ax.axhline(250, color='0.3', ls='--', lw=1.2); ax.text(-.55, 253, '250 ms cutoff', fontsize=11, color='0.3', va='bottom')
    h = [Line2D([], [], color=ROUTE_C[r], marker='s', ls='none', ms=9, label=r) for r in ROUTES]
    h += [Line2D([], [], color='0.2', marker='o', mfc='white', ls='none', ms=9, mew=1.8, label='same speed (open)'),
          Line2D([], [], color='0.2', marker='s', ls='none', ms=9, label='mixed speed (solid)')]
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0); fig.tight_layout()
    save(fig, 'fig8b', 'Per-episode p95, mean over all episodes (failed included), 95% case-bootstrap CI. Cached fallback, nominal LEO. Faded = partial (n shown).')
    write_csv('fig8b', recs)

def fig6_ns(rows):
    """Fallback vs WAIT without success: stalled decisions and collision attempts per move, all episodes."""
    recs = []; NS_ = (8, 16, 32); DL = (('nominal LEO', 0.0), ('100 ms deadline', .15), ('real time', .3)); MK = {8: 'o', 16: 's', 32: 'D'}
    metrics = (('Stalled decisions', 'Stalled decisions\n(% of robot decisions)',
                lambda s: boot_ratio([100 * x['row']['scheduled_cutoff_waits'] for x in s], [x['row']['eligible_decisions'] for x in s]),
                lambda s: sum(x['row']['eligible_decisions'] for x in s) > 0),
               ('Collision attempts', 'Attempted collisions\nper 1,000 robot moves',
                lambda s: boot_ratio([1000 * (x['row']['attempted_vertex_conflicts'] + x['row']['attempted_edge_swaps']) for x in s], [x['row']['physical_distance_cells'] for x in s]),
                lambda s: sum(x['row']['physical_distance_cells'] for x in s) > 0))
    for tag, speed, (tw, tc) in (('a', 'mixed speed', ('A2', 'A4')), ('b', 'same speed', ('A1', 'A3'))):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9)); w = .13
        for ax, (title, ylab, fn, defined) in zip(axes, metrics):
            for di, (dlab, dl) in enumerate(DL):
                for ni, n in enumerate(NS_):
                    for pi, (pol, tr) in enumerate((('WAIT', tw), ('cached fallback', tc))):
                        x = di + (ni * 2 + pi - 2.5) * w; sub = delay_rows(rows, dlab, tr, n)
                        grp = 'nominal' if dl == 0 else 'delay'
                        if not sub or not defined(sub): continue  # no data or undefined: omitted
                        est, lo, hi = fn(sub)
                        dot(ax, x, est, lo, hi, POLICY_C[pol], n=len(sub), marker=MK[n], ms=8, rot=90)
                        recs.append(dict(metric=title, speed=speed, policy=pol, treatment=tr, condition=dlab, num_agents=n, episodes=len(sub), value=est, ci_low=lo, ci_high=hi))
            ax.set_xticks(range(3)); ax.set_xticklabels([d for d, _ in DL]); ax.set_xlim(-.6, 2.6); ax.set_xlabel('Network condition')
            ax.set_title(title); ax.set_ylabel(ylab); ax.set_ylim(bottom=0)
        axes[0].set_ylim(0, 112); axes[1].set_ylim(0, axes[1].dataLim.y1 * 1.22)
        h = [Line2D([], [], color=POLICY_C['WAIT'], marker='o', ls='none', ms=9, label='WAIT on missed command'),
             Line2D([], [], color=POLICY_C['cached fallback'], marker='o', ls='none', ms=9, label='cached-path fallback')]
        h += [Line2D([], [], color='0.3', marker=MK[n], ls='none', ms=8, label=f'N = {n}') for n in NS_]
        top(fig, h, 5, 'Cached fallback vs WAIT')
        save(fig, f'fig6{tag}', f'{speed.capitalize()}. All episodes (failed included); ratios pooled over episodes, 95% case-bootstrap CI. Within each condition: N = 8, 16, 32 left to right. '
             'Faded = partial (n shown).')
    write_csv('fig6', recs)


if __name__ == '__main__':
    rows = load(); print('episodes', len(rows))
    if NO_SUCCESS:
        # base names + _ns: fig8b, fig6a/b, fig2, fig2b, fig4, fig5b, fig7 (fig8a dropped: success only)
        for f in (fig8_ns, fig6_ns, fig2, fig4, fig5b, fig7):
            f(rows); print('ok', f.__name__)
    else:
        for f in (fig8, fig6, fig2, fig4, fig5b, fig7):
            f(rows); print('ok', f.__name__)
