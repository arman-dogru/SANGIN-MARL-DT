#!/workspace/SANGIN-dev/.venv/bin/python
"""Omega-sensitivity AoI x semantic-error extraction (one episode at a time; resumable).
    nice -n 19 harness/omega_aoi_extract.py RUN_DIR --name NAME
Per episode, with the run's frozen code/ (via prior_aoi_extract.import_code / install_hook):
  (a) DT consistency pairs: frozen remote_audit.replay_trace rebuilds every decision-epoch consistency snapshot
      (digest-checked against archived consistency_sha256); hook bins (max(self,peer) age, active, relevant, e2e, ul, dl).
  (b) physical-reference shadow pairs: archived oracle_events 'controlled_peer_shadow' (evaluation-only; one per
      inference, receiver = inferring robot, peers = all others).  pair['ordering_error'] = |physical order (fresh
      sensed priorities at the inference instant) - cached order the robot actually used|; pair['physical_relevant'] =
      the robots' TRUE planned 5-step paths intersect (vertex or edge swap) inside the receiver's 7-cell FoV.
      Each diagnostic's sha256 is recomputed (canonical json) and must equal its stored checksum.
Totals are checked against the sibling <case>.result.json row.
"""
import argparse, gc, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/workspace/leo_study/harness')
import prior_aoi_extract as P

OUTD = Path('/workspace/leo_study/figures/aoi_semantic/omega')
W = P.AOI_W; MAXB = P.AOI_MAX

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--name', required=True)
    a = ap.parse_args(); run = a.run.resolve()
    RA, src = P.import_code(run); st = P.install_hook(RA)
    cache = OUTD / f'{a.name}_episodes.jsonl'; done = set()
    if cache.exists(): done = {(r['condition'], r['case_id']) for r in map(json.loads, cache.open())}
    todo = [t for t in P.archives(run, []) if t[:2] not in done]
    print('code', src, 'todo', len(todo), 'cached', len(done), flush=True)
    for i, (cond, case, f) in enumerate(todo):
        t0 = time.monotonic(); st.update(counter=Counter(), prev=[], snaps=0, pairs=0)
        resf = f.with_name(case + '.result.json'); R = json.loads(resf.read_text())
        row = R['row']; exp = R.get('audit', {}).get('archive_sha256')
        rec = dict(run=a.name, condition=cond, case_id=case, archive=str(f), weight_id=R.get('weight') if isinstance(R.get('weight'), str) else R.get('audit', {}).get('weight_id'))
        rec['sha_ok'] = (P._sha(f) == exp) if exp else None
        tr = P._load(f); rs = tr.get('remote_settings', {})
        rec.update(num_agents=len(tr['frames'][0]['positions']), delta_seconds=rs.get('seconds_per_step'),
                   success=tr.get('outcome', {}).get('success'), steps=tr.get('outcome', {}).get('steps'))
        # (b) shadows first (from archive, independent of replay)
        S = Counter(); sh_ok = sh_n = 0; prev = None; chain_ok = 0
        for e in tr.get('oracle_events', ()):
            if e['event'] != 'controlled_peer_shadow': continue
            sh_n += 1; d = {k: v for k, v in e.items() if k != 'checksum'}
            h = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            sh_ok += h == e.get('checksum'); chain_ok += prev is None or e.get('previous_checksum') == prev; prev = e.get('checksum')
            done_ = {int(k): bool(v.get('completed')) for k, v in e['fresh_states'].items()}
            for p in e['pairs']:
                age = max(p['self_age_seconds'], p['peer_age_seconds'])
                S[(min(int(age / W), MAXB), min(int(p['peer_age_seconds'] / W), MAXB), int(bool(p['physical_relevant'])),
                   int(p['ordering_error']), int(p['relevant']), int(p['fov_error']), int(done_.get(p['peer'], False)))] += 1
        rec.update(shadow_events=sh_n, shadow_sha_ok=sh_ok, shadow_chain_ok=chain_ok)
        rec['shadow_fields'] = ['aoi_bin', 'peer_aoi_bin', 'physical_relevant', 'ordering_error', 'relevant', 'fov_error', 'peer_completed', 'count']
        rec['shadow'] = [list(k) + [v] for k, v in sorted(S.items())]
        T = Counter()
        for k, v in S.items():
            T['inference_pairs'] += v; T['physical_relevant_pairs'] += v * k[2]; T['physical_fov_order_errors'] += v * k[2] * k[3]
            T['inference_relevant_pairs'] += v * k[4]; T['inference_fov_errors'] += v * k[5]
        # (a) DT consistency replay
        try:
            res = RA.replay_trace(tr, None); rec['replay'] = 'ok'
        except Exception as ex:
            res = None; rec['replay'] = f'error {type(ex).__name__}: {ex}'
        rec['aoi_fields'] = P.FIELDS; rec['aoi'] = [list(k) + [v] for k, v in sorted(st['counter'].items())]
        for k, v in st['counter'].items():
            T['ordering_pairs_verified'] += v; T['ordering_errors'] += v * k[3]; T['fov_relevant_pairs'] += v * k[2]; T['fov_errors'] += v * k[2] * k[3]
        rec['totals'] = dict(T); rec['row'] = {k: row.get(k) for k in T}
        rec['row_match'] = {k: T[k] == row.get(k) for k in T if row.get(k) is not None}
        rec['wall_seconds'] = round(time.monotonic() - t0, 1)
        with cache.open('a') as o: o.write(json.dumps(rec) + '\n')
        bad = [k for k, v in rec['row_match'].items() if not v]
        print(f"[{i+1}/{len(todo)}] {cond}/{case} N={rec['num_agents']} d={rec['delta_seconds']} succ={rec['success']} sha={rec['sha_ok']} "
              f"replay={rec['replay'][:40]} shadows={sh_n} shaok={sh_ok} chain={chain_ok} mismatch={bad} {rec['wall_seconds']}s", flush=True)
        del tr, res; st['prev'] = []; gc.collect()

if __name__ == '__main__': main()
