#!/workspace/SANGIN-dev/.venv/bin/python
"""Shadow-only pass: physical-reference pairs keyed also by the CACHED peer 'completed' flag (pair['peer_completed']).
    nice -n 19 harness/omega_shadow_pass.py RUN_DIR --name NAME   -> figures/aoi_semantic/omega/NAME_shadow2.jsonl"""
import argparse, gc, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/workspace/leo_study/harness'); import prior_aoi_extract as P
ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--name', required=True); a = ap.parse_args()
P.import_code(a.run.resolve())
out = Path('/workspace/leo_study/figures/aoi_semantic/omega') / f'{a.name}_shadow2.jsonl'
done = {(r['condition'], r['case_id']) for r in map(json.loads, out.open())} if out.exists() else set()
for cond, case, f in P.archives(a.run.resolve(), []):
    if (cond, case) in done: continue
    tr = P._load(f); S = Counter()
    for e in tr.get('oracle_events', ()):
        if e['event'] != 'controlled_peer_shadow': continue
        fin = {int(k): bool(v.get('completed')) for k, v in e['fresh_states'].items()}
        for p in e['pairs']:
            S[(min(int(max(p['self_age_seconds'], p['peer_age_seconds']) / P.AOI_W), P.AOI_MAX), int(bool(p['physical_relevant'])),
               int(p['ordering_error']), int(fin.get(p['peer'], False)), int(bool(p['peer_completed'])), int(bool(p['self_completed'])))] += 1
    with out.open('a') as o: o.write(json.dumps(dict(condition=cond, case_id=case, num_agents=len(tr['frames'][0]['positions']),
        fields=['aoi_bin', 'physical_relevant', 'ordering_error', 'peer_finished_true', 'peer_finished_cached', 'self_finished_cached', 'count'],
        shadow=[list(k) + [v] for k, v in sorted(S.items())])) + '\n')
    print(cond, case, sum(S.values()), flush=True); del tr; gc.collect()
