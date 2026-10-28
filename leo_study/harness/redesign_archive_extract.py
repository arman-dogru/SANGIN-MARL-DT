#!/workspace/SANGIN-dev/.venv/bin/python
"""Mixed-speed (redesign A4/W1) AoI x semantic-error extraction from Arman's archived traces (one episode per call; resumable).
    nice -n 19 harness/redesign_archive_extract.py EPISODE_DIR      (…/episodes/A4/W1/<route>/<N>/<case>)
Same logic as omega_aoi_extract.py (+ omega_shadow_pass.py):
  (a) frozen remote_audit.replay_trace (SANGIN-dev/src @03c9587, protocol dt_bundle_slow_pipeline_v2 supported) rebuilds every
      decision-epoch consistency snapshot (digest-checked against archived consistency_sha256); hook bins pairs by max(self,peer) age.
  (b) archived oracle_events 'controlled_peer_shadow' -> physical-reference pairs (+ cached/true finished flags), checksums verified.
Traces need the archive's compact_telemetry.py on sys.path (README.txt). Output: redesign_aoi/A4_archive_episodes.jsonl
(same record layout as redesign_aoi_batch.py). Totals checked against episodes.csv and the episode's result.json row."""
import sys, json, gzip, pickle, hashlib, time, resource, csv, gc
from collections import Counter
from pathlib import Path
ARC = Path('/workspace/leo_study/redesign_archive/20261027-004052')
sys.path[:0] = [str(ARC), '/workspace/SANGIN-dev/src']
from panama_uav import remote_audit as RA
OUT = Path('/workspace/leo_study/redesign_aoi/A4_archive_episodes.jsonl'); AOI_W, AOI_MAX = .05, 400
ep = Path(sys.argv[1]); route, case_id = ep.parent.parent.name, ep.name
t0 = time.monotonic()
R = json.loads((ep / 'result.json').read_text()); row = R['row']
f = ep / 'trace.pkl.gz'; sha = hashlib.sha256(f.read_bytes()).hexdigest()
with gzip.open(f, 'rb') as s: tr = pickle.load(s)
C = Counter(); prev = []; orig = RA.consistency_snapshot
def hooked(positions, truth, *a, **k):
    snap = orig(positions, truth, *a, **k)
    for s_ in prev: s_['pairs'] = []
    prev[:] = [snap]
    if k.get('schema_version', RA.CONSISTENCY_SCHEMA) == RA.CONSISTENCY_SCHEMA:
        for p in snap.get('pairs', ()):
            age = max(p['self_age_seconds'], p['peer_age_seconds'])
            act = int(not truth[p['receiver']]['completed'] and not truth[p['peer']]['completed'])
            C[(min(int(age / AOI_W), AOI_MAX), act, int(p['relevant']), int(p['e2e_ordering_error']),
               int(p.get('uplink_ordering_error', 0)), int(p.get('downlink_ordering_error', 0)),
               int(bool(p.get('physical_relevant'))), int(bool(p.get('ordering_error'))))] += 1
    return snap
RA.consistency_snapshot = hooked
S = Counter(); sh_n = sh_ok = chain = 0; pc = None
for e in tr.get('oracle_events', ()):
    if e['event'] != 'controlled_peer_shadow': continue
    e = dict(e.items()); sh_n += 1; d = {k: v for k, v in e.items() if k != 'checksum'}
    sh_ok += hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == e.get('checksum')
    chain += pc is None or e.get('previous_checksum') == pc; pc = e.get('checksum')
    fin = {int(k): bool(v.get('completed')) for k, v in e['fresh_states'].items()}
    for p in e['pairs']:
        S[(min(int(max(p['self_age_seconds'], p['peer_age_seconds']) / AOI_W), AOI_MAX), int(bool(p['physical_relevant'])),
           int(p['ordering_error']), int(fin.get(p['peer'], False)), int(bool(p['peer_completed'])), int(bool(p['self_completed'])))] += 1
try: RA.replay_trace(tr, None); replay = 'ok'
except Exception as ex: replay = f'error {type(ex).__name__}: {ex}'
F = Counter()
for k, v in C.items(): F[k[:6]] += v
tot = dict(ordering_pairs_verified=sum(F.values()), ordering_errors=sum(v * k[3] for k, v in F.items()),
           relevant_pairs=sum(v * k[2] for k, v in F.items()), fov_errors=sum(v * k[2] * k[3] for k, v in F.items()),
           snap_physical_relevant_pairs=sum(v * k[6] for k, v in C.items()), snap_physical_relevant_ordering_errors=sum(v * k[6] * k[7] for k, v in C.items()),
           physical_relevant_pairs=sum(v * k[1] for k, v in S.items()), physical_fov_order_errors=sum(v * k[1] * k[2] for k, v in S.items()))
ref = next((r for r in csv.DictReader(open('/workspace/SANGIN-dev/results/redesign/20261027-004052/episodes.csv'))
            if r['treatment'] == 'A4' and r['route'] == route and r['case_id'] == case_id), {})
I = lambda v: int(float(v)) if v not in (None, '') else None
chk = {k: (tot[k], I(ref.get(k)), I(row.get(k))) for k in ('ordering_pairs_verified', 'ordering_errors', 'relevant_pairs', 'fov_errors')}
chk['physical_relevant_pairs'] = (tot['snap_physical_relevant_pairs'], I(ref.get('physical_relevant_pairs')), I(row.get('physical_relevant_pairs')))
chk['physical_relevant_ordering_errors'] = (tot['snap_physical_relevant_ordering_errors'], I(ref.get('physical_relevant_ordering_errors')), I(row.get('physical_relevant_ordering_errors')))
chk['steps'] = (tr['outcome']['steps'], I(ref.get('steps')), I(row.get('steps'))); chk['success'] = (str(tr['outcome']['success']), ref.get('success'), str(row.get('success')))
ok = replay == 'ok' and all(len(set(v)) == 1 for v in chk.values()) and sh_ok == sh_n
rec = dict(run='redesign_A4_archive', condition=f'A4/{route}', route=route, case_id=case_id, num_agents=len(tr['frames'][0]['positions']),
           delta_seconds=tr['remote_settings']['seconds_per_step'], success=tr['outcome']['success'], steps=tr['outcome']['steps'],
           archive=str(f), archive_sha256=sha, archive_sha_ok=sha == R.get('audit', {}).get('archive_sha256'), replay=replay,
           shadow_events=sh_n, shadow_sha_ok=sh_ok, shadow_chain_ok=chain, validated=ok, check=chk, totals=tot,
           wall_seconds=round(time.monotonic() - t0, 1), peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
           aoi_fields=['aoi_bin', 'active', 'relevant', 'e2e_error', 'ul_error', 'dl_error', 'count'], aoi=[list(k) + [v] for k, v in sorted(F.items())],
           shadow_fields=['aoi_bin', 'physical_relevant', 'ordering_error', 'peer_finished_true', 'peer_finished_cached', 'self_finished_cached', 'count'],
           shadow=[list(k) + [v] for k, v in sorted(S.items())])
with open(OUT, 'a') as o: o.write(json.dumps(rec, default=str) + '\n')
print(route, case_id, 'N', rec['num_agents'], 'steps', rec['steps'], 'replay', replay[:60], 'sha', rec['archive_sha_ok'], f'shadow {sh_ok}/{sh_n}',
      'validated', ok, {k: v for k, v in chk.items() if len(set(v)) > 1}, rec['wall_seconds'], 's', rec['peak_rss_gib'], 'GiB', flush=True)
