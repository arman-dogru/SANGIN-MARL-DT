#!/workspace/SANGIN-dev/.venv/bin/python
"""Run-level Xi counters (incl. R~ = physical_relevant_ordering_errors / physical_relevant_pairs on DT epoch-snapshot pairs,
the redesign run's definition) for omega-archive episodes, via the run's frozen remote_audit.replay_trace (digest-checked).
    nice -n 19 harness/omega_phys_extract.py RUN_DIR --name NAME  -> figures/aoi_semantic/omega/NAME_phys.jsonl (resumable)
R_E2E / R_FoV totals are checked against the episode's result.json row."""
import argparse, gc, json, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/workspace/leo_study/harness'); import prior_aoi_extract as P
ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--name', required=True); a = ap.parse_args()
run = a.run.resolve(); RA, src = P.import_code(run)
out = Path('/workspace/leo_study/figures/aoi_semantic/omega') / f'{a.name}_phys.jsonl'
done = {(r['condition'], r['case_id']) for r in map(json.loads, out.open())} if out.exists() else set()
T = Counter(); prev = []; orig = RA.consistency_snapshot
def hooked(positions, truth, *aa, **k):
    snap = orig(positions, truth, *aa, **k)
    for s in prev: s['pairs'] = []
    prev[:] = [snap]
    if k.get('schema_version', RA.CONSISTENCY_SCHEMA) == RA.CONSISTENCY_SCHEMA:
        for p in snap.get('pairs', ()):
            T['ordering_pairs_verified'] += 1; T['ordering_errors'] += p['e2e_ordering_error']
            T['fov_relevant_pairs'] += p['relevant']; T['fov_errors'] += p['relevant'] * p['e2e_ordering_error']
            T['physical_unknown_pairs'] += p['physical_relevant'] is None
            T['physical_relevant_pairs'] += p['physical_relevant'] or 0
            T['physical_relevant_ordering_errors'] += (p['physical_relevant'] or 0) * p['ordering_error']
    return snap
RA.consistency_snapshot = hooked
print('code', src, flush=True)
for cond, case, f in P.archives(run, []):
    if (cond, case) in done: continue
    t0 = time.monotonic(); T.clear(); prev.clear()
    R = json.loads(f.with_name(case + '.result.json').read_text()); row = R['row']
    tr = P._load(f)
    try: RA.replay_trace(tr, None); rp = 'ok'
    except Exception as ex: rp = f'error {type(ex).__name__}: {ex}'
    chk = {k: (T[k], row.get(k)) for k in ('ordering_pairs_verified', 'ordering_errors', 'fov_relevant_pairs', 'fov_errors')}
    rec = dict(run=a.name, condition=cond, case_id=case, num_agents=len(tr['frames'][0]['positions']), success=tr.get('outcome', {}).get('success'),
               steps=tr.get('outcome', {}).get('steps'), replay=rp, totals=dict(T), row_check=chk, validated=rp == 'ok' and all(x == y for x, y in chk.values()),
               wall_seconds=round(time.monotonic() - t0, 1))
    with out.open('a') as o: o.write(json.dumps(rec) + '\n')
    print(cond, case, rec['num_agents'], rp[:50], 'valid', rec['validated'], dict(T), rec['wall_seconds'], flush=True)
    del tr; prev.clear(); gc.collect()
