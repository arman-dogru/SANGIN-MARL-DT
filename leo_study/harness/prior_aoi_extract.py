#!/workspace/SANGIN-dev/.venv/bin/python
"""AoI vs semantic ordering error from PRIOR SANGIN-dev remote-DT episode archives (communication_effects, omega,
staleness, priority, ... any run whose episodes/**/<case>.json.gz is a run_episode trace with remote_events).

Pairs are not stored in archives; they are rebuilt with the run's own frozen remote_audit.replay_trace
(<run>/code/panama_uav if present, else /workspace/SANGIN-dev/src), which checks every rebuilt consistency snapshot
against the archived consistency_sha256.  A wrapper around remote_audit.consistency_snapshot aggregates each
directed pair (receiver i, peer j) per decision exactly like the LEO harness pair_stats.aoi:
    aoi_bin = min(int(max(self_age, peer_age) / 0.05), 400)
    active  = receiver and peer not physically completed (truth at the decision epoch)
    relevant, e2e_error, ul_error, dl_error = pair['relevant'], pair['*_ordering_error']
and drops pair lists after their digest check (memory lean).  Archives are processed one at a time.

    nice -n 19 harness/prior_aoi_extract.py RUN_DIR [--name NAME] [--conditions GLOB ...] [--extract-only|--plot-only]

Outputs (figures/aoi_semantic/prior/):  <name>_episodes_aoi.jsonl  (one line per episode; resumable cache)
    <name>_fig9.png/.pdf/.csv (fine bins, merged to >=200), <name>_fig9b_coarse.*, <name>_fig9_by_condition.csv
Delta (decision interval) = trace remote_settings.seconds_per_step per episode (fallback: manifest
configuration.communication.seconds_per_step); x = (aoi_bin + 0.5) * 0.05 / Delta.
"""
import argparse, csv, fnmatch, gc, gzip, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

AOI_W, AOI_MAX = .05, 400
FIELDS = ['aoi_bin', 'active', 'relevant', 'e2e_error', 'ul_error', 'dl_error', 'count']
OUTD = Path('/workspace/leo_study/figures/aoi_semantic/prior')

def import_code(run, override=None):
    code = run / 'code'
    src = Path(override) if override else code if (code / 'panama_uav' / 'remote_audit.py').exists() else Path('/workspace/SANGIN-dev/src')
    sys.path.insert(0, str(src))
    sys.path.append('/workspace/SANGIN-dev/scripts')     # compact_telemetry (redesign storage-v1 pickles)
    from panama_uav import remote_audit as RA
    return RA, src

def install_hook(RA):
    orig = RA.consistency_snapshot; state = dict(counter=None, prev=[], snaps=0, pairs=0)
    def hooked(positions, truth, *a, **k):
        snap = orig(positions, truth, *a, **k)
        for s in state['prev']: s['pairs'] = []          # previous snapshots already digest-checked
        state['prev'] = [snap]
        if k.get('schema_version', RA.CONSISTENCY_SCHEMA) == RA.CONSISTENCY_SCHEMA and state['counter'] is not None:
            c = state['counter']; state['snaps'] += 1
            for p in snap.get('pairs', ()):
                age = max(p['self_age_seconds'], p['peer_age_seconds'])
                act = int(not truth[p['receiver']]['completed'] and not truth[p['peer']]['completed'])
                c[(min(int(age / AOI_W), AOI_MAX), act, int(p['relevant']), int(p['e2e_ordering_error']),
                   int(p.get('uplink_ordering_error', 0)), int(p.get('downlink_ordering_error', 0)))] += 1
                state['pairs'] += 1
        return snap
    RA.consistency_snapshot = hooked
    return state

def archives(run, pats):
    base = run / 'episodes' if (run / 'episodes').is_dir() else run
    files = sorted(list(base.rglob('*.json.gz')) + [f for f in base.rglob('*.pkl.gz') if not f.name.startswith('failed_')])
    for f in files:
        if f.name in ('trace.pkl.gz', 'trace.json.gz'):      # redesign layout .../<cond...>/<case>/trace.pkl.gz
            cond = f.parent.parent.relative_to(base).as_posix(); case = f.parent.name
        else:
            cond = f.parent.relative_to(base).as_posix(); case = f.name.split('.')[0]
        if pats and not any(fnmatch.fnmatch(cond, p) for p in pats): continue
        yield cond, case, f

def _load(f):
    if f.name.endswith('.pkl.gz'):
        import pickle      # only open verified, locally generated archives (see OMEGA_SENSITIVITY.md)
        with gzip.open(f, 'rb') as s: return pickle.load(s)
    with gzip.open(f, 'rt', encoding='utf-8') as s: return json.load(s)

def _expected_sha(f):
    for cand in (f.with_name(f.name.split('.')[0] + '.result.json'), f.with_name('result.json'), f.with_name(f.name.split('.')[0] + '.json')):
        if cand.exists() and cand != f:
            try: a = json.loads(cand.read_text()).get('audit', {})
            except Exception: continue
            if a.get('archive_sha256'): return a['archive_sha256']
    return None

def _sha(f):
    import hashlib; h = hashlib.sha256()
    with open(f, 'rb') as s:
        for b in iter(lambda: s.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def extract(run, name, pats, cfg_delta, code=None, verify=True):
    RA, src = import_code(run, code); st = install_hook(RA)
    cache = OUTD / f'{name}_episodes_aoi.jsonl'; done = set()
    if cache.exists():
        for l in cache.open(): r = json.loads(l); done.add((r['condition'], r['case_id']))
    todo = [(c, k, f) for c, k, f in archives(run, pats) if (c, k) not in done]
    print(f'code={src}  archives to process={len(todo)}  cached={len(done)}', flush=True)
    for i, (cond, case, f) in enumerate(todo):
        t0 = time.monotonic(); st.update(counter=Counter(), prev=[], snaps=0, pairs=0)
        exp = _expected_sha(f) if verify else None
        if exp and _sha(f) != exp:
            rec = dict(condition=cond, case_id=case, status='error', error='archive sha256 differs from result audit')
            with cache.open('a') as o: o.write(json.dumps(rec) + '\n')
            print('SKIP sha mismatch', f, flush=True); continue
        trace = _load(f)
        rs = trace.get('remote_settings', {}); delta = rs.get('seconds_per_step', cfg_delta)
        n = len(trace['frames'][0]['positions']); out = trace.get('outcome', {})
        rec = dict(condition=cond, case_id=case, archive=str(f), sha_checked=bool(exp), num_agents=n, delta_seconds=delta,
                   success=out.get('success'), steps=out.get('steps'))
        res = None
        try:
            res = RA.replay_trace(trace, None)       # frozen-DQN re-inference skipped; all checksums still verified
            rec.update(status='ok', consistency_events=len(res.get('pair_consistency', [])), hooked_snapshots=st['snaps'],
                       pair_decisions=st['pairs'], **{k: res[k] for k in ('ordering_pairs_verified', 'consistency_decisions', 'relevant_pairs', 'vertex_relevant_pairs', 'vertex_e2e_errors', 'fov_errors') if k in res})
        except Exception as e:
            rec.update(status='error', error=f'{type(e).__name__}: {e}')
        rec['aoi_fields'] = FIELDS; rec['aoi_bin_seconds'] = AOI_W
        rec['aoi'] = [list(k) + [v] for k, v in sorted(st['counter'].items())]
        rec['wall_seconds'] = round(time.monotonic() - t0, 2)
        with cache.open('a') as o: o.write(json.dumps(rec) + '\n')
        print(f"[{i+1}/{len(todo)}] {cond}/{case} N={n} {rec['status']} pairs={st['pairs']} {rec['wall_seconds']}s", flush=True)
        del trace, res; st['prev'] = []; gc.collect()

def load_cache(name):
    sys.path.insert(0, '/workspace/leo_study/harness'); eps = []
    for l in (OUTD / f'{name}_episodes_aoi.jsonl').open():
        r = json.loads(l)
        if r['status'] != 'ok' or not r['aoi']: continue
        a = np.array(r['aoi'], float); d = r['delta_seconds']
        eps.append(dict(n=r['num_agents'], cond=r['condition'], x=(a[:, 0] + .5) * r['aoi_bin_seconds'] / d,
                        act=a[:, 1].astype(bool), rel=a[:, 2].astype(bool), err=a[:, 3], cnt=a[:, 6], delta=d))
    return eps

def plot(name, eps):
    import fig9_aoi as F
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    F.OUT = OUTD
    Ns = sorted({e['n'] for e in eps}); deltas = sorted({e['delta'] for e in eps})
    pal = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2', '#D55E00', '#000000']
    col = {n: pal[i % len(pal)] for i, n in enumerate(Ns)}
    xs = np.concatenate([e['x'][e['act']] for e in eps]); hi = float(np.ceil(min(np.percentile(np.repeat(xs, 1), 99.5), 10) * 4) / 4) if len(xs) else 3.5
    fine = list(np.round(np.arange(0, max(hi, 1) + 1e-9, .25), 2))
    def draw(edges, fname, merge_bins, recs):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9)); tot = {}
        for ax, (rel, title) in zip(axes, ((False, 'All active pairs'), (True, 'Coordination-relevant pairs'))):
            for n in Ns:
                sub = [e for e in eps if e['n'] == n]
                c, e_ = F.per_episode(sub, np.array(edges, float), rel); sx = F.per_episode.sx; t0 = c.sum(0); E = np.array(edges[:-1])
                if merge_bins:
                    bins, c, e_ = F.merge(edges, c, e_)
                    xc = [sx[(E >= a - 1e-9) & (E < b - 1e-9)].sum() / max(t0[(E >= a - 1e-9) & (E < b - 1e-9)].sum(), 1e-12) for a, b in bins]
                else:
                    keep = c.sum(0) > 0; bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if keep[i]]; c, e_ = c[:, keep], e_[:, keep]
                    xc = [F.COARSE.index(a) for a, b in bins]
                if not bins: continue
                est, lo, hi_ = F.boot(c, e_)
                ax.plot(xc, est, color=col[n], lw=2.2, marker='o', ms=6); ax.fill_between(xc, lo, hi_, color=col[n], alpha=.15, lw=0)
                tot[(rel, n)] = int(c.sum())
                for j, ((a, b), v, l, h, cc, ee) in enumerate(zip(bins, est, lo, hi_, c.sum(0), e_.sum(0))):
                    recs.append(dict(panel=title, num_agents=n, aoi_lo_delta=a, aoi_hi_delta=b, aoi_mean_delta=xc[j] if merge_bins else None,
                                     pair_decisions=int(cc), errors=int(ee), pr_error=v, ci_low=l, ci_high=h, episodes=len(sub)))
            ax.set_title(title); ax.set_xlabel('AoI (decision intervals)'); ax.set_ylim(bottom=0); ax.grid(True, alpha=.25)
            if merge_bins: ax.set_xlim(0, edges[-1])
            if not merge_bins:
                labs = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,10)', '[10,∞)']; ax.set_xticks(range(6)); ax.set_xticklabels(labs); ax.set_xlim(-.5, 5.5)
        axes[0].set_ylabel('Pr(semantic ordering error)')
        fig.legend(handles=[Line2D([], [], color=col[n], lw=2.4, marker='o', label=f'N = {n}') for n in Ns],
                   loc='lower center', bbox_to_anchor=(.5, .9), ncol=len(Ns)); fig.tight_layout(rect=(0, 0, 1, .9))
        F.PF_save(fig, fname, f'Run {name}; Δ = ' + ', '.join(f'{d:g} s' for d in deltas) + ' (archived decision interval). Pair decisions (all / relevant): '
                  + '; '.join(f'N = {n}: {tot.get((False, n), 0):,} / {tot.get((True, n), 0):,}' for n in Ns) + '. 95% bootstrap over episodes.')
    r1, r2, r3 = [], [], []
    draw(fine, f'{name}_fig9', True, r1); F.write_csv(f'{name}_fig9', r1)
    draw(F.COARSE, f'{name}_fig9b_coarse', False, r2); F.write_csv(f'{name}_fig9b_coarse', r2)
    for cond in sorted({e['cond'] for e in eps}):
        for n in Ns:
            sub = [e for e in eps if e['n'] == n and e['cond'] == cond]
            if not sub: continue
            for rel, panel in ((False, 'All active pairs'), (True, 'Coordination-relevant pairs')):
                c, e_ = F.per_episode(sub, np.array(F.COARSE, float), rel)
                for i in range(len(F.COARSE) - 1):
                    if c[:, i].sum(): r3.append(dict(condition=cond, num_agents=n, panel=panel, aoi_lo_delta=F.COARSE[i], aoi_hi_delta=F.COARSE[i + 1],
                        episodes=len(sub), pair_decisions=int(c[:, i].sum()), errors=int(e_[:, i].sum()), pr_error=e_[:, i].sum() / c[:, i].sum()))
    F.write_csv(f'{name}_fig9_by_condition', r3)
    print('figures ->', OUTD, 'episodes', len(eps), 'N', Ns, 'delta', deltas)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--name')
    ap.add_argument('--conditions', nargs='*', default=[]); ap.add_argument('--extract-only', action='store_true'); ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--code', help='override source root containing panama_uav (default <run>/code, else SANGIN-dev/src)')
    ap.add_argument('--no-verify', action='store_true', help='skip archive sha256 check against sibling result json')
    a = ap.parse_args(); run = a.run.resolve(); name = a.name or run.name; OUTD.mkdir(parents=True, exist_ok=True)
    m = json.loads((run / 'manifest.json').read_text()) if (run / 'manifest.json').exists() else {}
    cfg_delta = m.get('configuration', {}).get('communication', {}).get('seconds_per_step')
    print('manifest protocol', m.get('protocol'), 'config seconds_per_step', cfg_delta)
    if not a.plot_only: extract(run, name, a.conditions, cfg_delta, a.code, not a.no_verify)
    if not a.extract_only:
        eps = load_cache(name)
        if a.conditions: eps = [e for e in eps if any(fnmatch.fnmatch(e['cond'], p) for p in a.conditions)]
        plot(name, eps)
