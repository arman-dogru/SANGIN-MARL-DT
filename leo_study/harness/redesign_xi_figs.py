#!/workspace/SANGIN-dev/.venv/bin/python
"""Run-level semantic ordering-error (Xi) rates, combined same speed + mixed speed (mirrors the omega AoI figures).
    nice -n 19 /workspace/SANGIN-dev/.venv/bin/python harness/redesign_xi_figs.py
Same speed (Delta = 1 s, WAIT) = omega-sensitivity run:
   N = 8/16: archive replays of every archived episode (failures included) -> figures/aoi_semantic/omega/{uav6_n8,uav6_n16,omega_w25}_phys.jsonl
             (harness/omega_phys_extract.py; W1 from the uav6 pilot + extension, W2-W5 allocator from omega-20261025-01)
   N = 32:   SANGIN-dev/results/omega_sensitivity/episodes.csv (completed missions only); no R~ (no archive).
Mixed speed (Delta = 0.35 s, cached fallback) = redesign checkpoint 20261027-004052, treatment A4 (episodes.csv; R~ verified against archive replays).
Per run:  R_E2E = ordering_errors / ordering_pairs_verified;  R_FoV = fov_errors / relevant pairs (zeta);
          R~ = physical_relevant_ordering_errors / physical_relevant_pairs (DT epoch-snapshot pairs whose TRUE 5-step paths conflict in FoV).
Figures: redesign_xi_run_level (by route, omega 1:1 vs mixed) and redesign_xi_run_level_by_omega (allocator, 5 omegas + mixed)."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RED = Path('/workspace/SANGIN-dev/results/redesign/20261027-004052/episodes.csv')
OMG = Path('/workspace/SANGIN-dev/results/omega_sensitivity/episodes.csv')
PHYS = Path('/workspace/leo_study/figures/aoi_semantic/omega')
STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'figures/aoi_semantic/redesign'; OUT.mkdir(parents=True, exist_ok=True)
B = 2000; RNG = np.random.default_rng(20261027); NS = (8, 16, 32); FLOOR = 3e-5
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 11.5, 'axes.grid': True, 'grid.alpha': .25, 'axes.grid.axis': 'y',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
R_LAB = {'allocated': 'allocator', 'cloud': 'all-cloud', 'edge': 'all-UAV'}
R_C = {'allocated': '#0072B2', 'cloud': '#009E73', 'edge': '#E69F00'}
COND = {'allocated-6ghz': 'allocated', 'cloud-6ghz': 'cloud', 'uav-6ghz': 'edge'}
W_LAB = {'W1': 'ω₁:ω₂ = 1:1', 'W2': 'ω₁:ω₂ = 1:2', 'W3': 'ω₁:ω₂ = 1:4', 'W4': 'ω₁:ω₂ = 2:1', 'W5': 'ω₁:ω₂ = 4:1'}
W_C = {'W1': '#3E5B86', 'W2': '#4F9B8F', 'W3': '#CDA84F', 'W4': '#8865AB', 'W5': '#C96B6E'}; MIX_C = '#222222'
YL = {'R_E2E': 'All directed pairs, $R_{\\mathrm{E2E}}$\n(% of pair decisions, log)',
      'R_FoV': 'Conflict-relevant pairs, $R_{\\mathrm{FoV}}$\n(% of relevant pairs, log)',
      'R_phys': 'Physically conflicting pairs, $\\tilde{R}$\n(% of physical-relevant pairs, log)'}
NUMDEN = {'R_E2E': ('ordering_errors', 'ordering_pairs_verified'), 'R_FoV': ('fov_errors', 'relevant_pairs'),
          'R_phys': ('physical_relevant_ordering_errors', 'physical_relevant_pairs')}
KEYS = ('R_E2E', 'R_FoV', 'R_phys')

def I(v): return int(float(v)) if v not in ('', None) else None
def rat(a, b): return a / b if (a is not None and b) else None
def finish(d):
    for k, (nu, de) in NUMDEN.items(): d[k] = rat(d[nu], d[de])
    return d

def load():
    rows = []
    for r in csv.DictReader(open(RED)):          # mixed speed
        if r['treatment'] != 'A4': continue
        rows.append(finish(dict(series='mixed', weight_id='W1', source='redesign_20261027-004052 episodes.csv', route=r['route'], num_agents=int(r['num_agents']),
            case_id=r['case_id'], success=r['success'] == 'True', steps=I(r['steps']), delta_s=.35,
            **{k: I(r[k]) for k in ('ordering_errors', 'ordering_pairs_verified', 'relevant_pairs', 'fov_errors', 'physical_relevant_pairs', 'physical_relevant_ordering_errors')})))
    omg = {(r['weight_id'], r['route'], r['case_id']): r for r in csv.DictReader(open(OMG))}
    xchk = []
    for name in ('uav6_n8', 'uav6_n16', 'omega_w25'):   # same speed N = 8/16, archive replays
        for q in map(json.loads, open(PHYS / f'{name}_phys.jsonl')):
            assert q['validated'], q['case_id']
            c = q['condition'].split('/'); w = c[0] if len(c) == 2 else 'W1'; route = COND[c[-1]]; t = q['totals']
            d = finish(dict(series='same', weight_id=w, source=f'omega archive replay ({name})', route=route, num_agents=q['num_agents'], case_id=q['case_id'],
                success=bool(q['success']), steps=q['steps'], delta_s=1., ordering_errors=t['ordering_errors'], ordering_pairs_verified=t['ordering_pairs_verified'],
                relevant_pairs=t['fov_relevant_pairs'], fov_errors=t['fov_errors'], physical_relevant_pairs=t['physical_relevant_pairs'],
                physical_relevant_ordering_errors=t['physical_relevant_ordering_errors']))
            rows.append(d); o = omg.get((w, route, q['case_id']))
            xchk.append(dict(weight_id=w, route=route, case_id=q['case_id'], success=d['success'], in_episodes_csv=o is not None,
                **({f'{k}_match': I(o[k2]) == d[k] for k, k2 in (('ordering_errors', 'ordering_errors'), ('ordering_pairs_verified', 'ordering_pairs_verified'),
                    ('relevant_pairs', 'fov_relevant_pairs'), ('fov_errors', 'fov_errors'), ('steps', 'steps'))} if o else {})))
    for (w, route, cid), r in omg.items():             # same speed N = 32 (episodes.csv only)
        if int(r['num_agents']) != 32: continue
        rows.append(finish(dict(series='same', weight_id=w, source='omega_sensitivity episodes.csv', route=route, num_agents=32, case_id=cid,
            success=r['success'] == 'True', steps=I(r['steps']), delta_s=1., ordering_errors=I(r['ordering_errors']), ordering_pairs_verified=I(r['ordering_pairs_verified']),
            relevant_pairs=I(r['fov_relevant_pairs']), fov_errors=I(r['fov_errors']), physical_relevant_pairs=None, physical_relevant_ordering_errors=None)))
    return rows, xchk

def boot_mean(v):
    v = np.asarray(v, float); m = RNG.choice(v, (B, len(v))).mean(1); return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def panel(ax, groups, key):
    """groups: (label, colour, marker, linestyle, rows)."""
    w = .8 / len(groups); recs = []
    for ci, (lab, col, mk, ls, rows) in enumerate(groups):
        xs, ms = [], []
        for gi, n in enumerate(NS):
            cell = [r for r in rows if r['num_agents'] == n and r[key] is not None]
            if not cell: continue
            va = np.array([100 * r[key] for r in cell]); x0 = gi + (ci - (len(groups) - 1) / 2) * w
            ax.scatter(x0 + RNG.uniform(-.025, .025, len(va)), np.where(va > 0, va, FLOOR), s=14, color=col, alpha=.4, lw=0, zorder=2, marker=mk)
            m, lo, hi = boot_mean(va); xs.append(x0); ms.append(m)
            nu, de = NUMDEN[key]; E, D = sum(r[nu] or 0 for r in cell), sum(r[de] or 0 for r in cell)
            recs.append(dict(metric=key, series=lab, num_agents=n, runs=len(va), successful_runs=sum(r['success'] for r in cell), zero_runs=int((va == 0).sum()),
                             errors=E, denominator=D, pooled_ratio_pct=100 * E / D if D else None, mean_pct=m, ci_low_pct=lo, ci_high_pct=hi,
                             median_pct=float(np.median(va)), sources='; '.join(sorted({r['source'] for r in cell}))))
            if m > 0: ax.errorbar(x0, m, yerr=[[m - max(lo, FLOOR)], [hi - m]], color=col, marker=mk, ms=7, mec='white', mew=.8, capsize=2.5, lw=1.5, zorder=4)
            else: ax.plot(x0, FLOOR, marker=mk, ms=7, mfc='white', mec=col, mew=1.6, zorder=4, ls='none')
        ms_ = np.array(ms, float); ax.plot(xs, np.where(ms_ > 0, ms_, np.nan), color=col, lw=1.4, ls=ls, alpha=.85, zorder=3)
    ax.set_yscale('log'); ax.set_ylim(FLOOR / 4.5, 30); ax.axhline(FLOOR, color='0.55', lw=.9, ls=':', zorder=1)
    ax.text(-.62, FLOOR * 1.25, '0 errors', fontsize=10, color='0.4', va='bottom')
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]); ax.set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1', '10'])
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlim(-.65, len(NS) - .35)
    ax.set_ylabel(YL[key]); ax.set_xlabel('Number of robots N'); return recs

def figure(groups, name, ncol):
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.2)); recs = []
    for ax, key in zip(axes, KEYS):
        rr = panel(ax, groups, key); [x.update(figure=name) for x in rr]; recs += rr
    fig.tight_layout(rect=(0, 0, 1, .86))
    h = [Line2D([], [], color=c, marker=m, ls=ls, lw=1.4, ms=7.5, label=l) for l, c, m, ls, _ in groups]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, .86), ncol=ncol)
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); return recs

def write(name, rr):
    keys = list(dict.fromkeys(k for r in rr for k in r))
    with open(OUT / f'{name}.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rr)

if __name__ == '__main__':
    rows, xchk = load(); write('redesign_xi_per_run', rows); write('redesign_xi_omega_archive_vs_csv', xchk)
    m = [x for x in xchk if x['in_episodes_csv']]
    print('same-speed archive episodes', len(xchk), 'in omega episodes.csv', len(m), 'all match', all(all(v for k, v in x.items() if k.endswith('_match')) for x in m),
          'not in csv (success flags):', [(x['weight_id'], x['route'], x['case_id'], x['success']) for x in xchk if not x['in_episodes_csv']])
    same = lambda r, k, w='W1': r['series'] == 'same' and r['route'] == k and r['weight_id'] == w and not (k == 'edge' and r['num_agents'] == 32)
    g_route = [(f'{R_LAB[k]}, same speed', R_C[k], 'o', '--', [r for r in rows if same(r, k)]) for k in R_LAB] + \
              [(f'{R_LAB[k]}, mixed speed', R_C[k], 's', '-', [r for r in rows if r['series'] == 'mixed' and r['route'] == k]) for k in R_LAB]
    g_omega = [(W_LAB[w], W_C[w], 'o', '-', [r for r in rows if same(r, 'allocated', w)]) for w in W_LAB] + \
              [('mixed speed', MIX_C, 's', '-', [r for r in rows if r['series'] == 'mixed' and r['route'] == 'allocated'])]
    recs = figure(g_route, 'redesign_xi_run_level', 2) + figure(g_omega, 'redesign_xi_run_level_by_omega', 3)
    write('redesign_xi_summary', recs)
    for r in recs: print(r['figure'][18:], r['metric'], r['series'], r['num_agents'], r['runs'], r['zero_runs'], f"{r['mean_pct']:.4f} [{r['ci_low_pct']:.4f},{r['ci_high_pct']:.4f}] err {r['errors']}/{r['denominator']}")
