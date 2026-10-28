#!/workspace/SANGIN-dev/.venv/bin/python
"""BATCH (one episode per call; appends to redesign_aoi/A4_episodes.jsonl). Based on SCOPING SMOKE (Task 2): rerun ONE redesign A4/W1 episode locally with a live consistency_snapshot AoI hook.
Code = /workspace/SANGIN-dev/src @03c9587 (read-only import); assets/cases = omega_archive omega-20261025-01 (the redesign run's source).
Mirrors scripts/run_redesign.py episode() for treatment != A0. No repo edits, nothing pushed.
    nice -n 19 harness/redesign_aoi_smoke.py ROUTE CASE_ID [--save-trace]"""
import sys, json, time, gzip, pickle, resource, argparse, csv
from collections import Counter
from dataclasses import replace
from pathlib import Path
sys.path[:0] = ['/workspace/SANGIN-dev/src', '/workspace/SANGIN-dev/scripts']
import torch, numpy as np
from panama_uav.config import FIXED_MAP, ProjectConfig
from panama_uav import redesign as RD
from panama_uav.redesign import settings_for, configured, run_frequency_episode
from panama_uav.inference_controls import InferenceCondition
from panama_uav.scenarios import Scenario
from panama_uav.training import load_model
from panama_uav.redesign_metrics import measure
import run_omega_sensitivity as core
SRC = Path('/workspace/leo_study/omega_archive/omega_sensitivity/omega-20261025-01'); OUT = Path('/workspace/leo_study/redesign_aoi')
AOI_W, AOI_MAX = .05, 400
ap = argparse.ArgumentParser(); ap.add_argument('route'); ap.add_argument('case_id'); a = ap.parse_args()
M = core.read(SRC / 'manifest.json'); cfg = ProjectConfig.from_dict(M['configuration'])
cfg = replace(cfg, project_root=SRC / 'assets', communication=replace(cfg.communication, leo_timeline=str(SRC / 'assets/orbit.npz')))
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
model, payload = load_model(SRC / 'assets/checkpoint.pt', cfg, torch.device('cpu'), require_qualified=True)
case = {c['case_id']: c for c in core.read(SRC / 'cases.json')}[a.case_id]
cond = next(c for c in core.CONDITIONS if c['route'] == a.route); condition = InferenceCondition(cond['name'], a.route)
base = replace(cfg, communication=replace(cfg.communication, mission_route='cloud' if a.route in ('allocated', 'cloud') else 'uav',
    backhaul_time_offset_seconds=cfg.communication.backhaul_time_offset_seconds + case['orbital_offset_seconds']),
    resource_allocation=replace(cfg.resource_allocation, failure_directory=str(OUT / 'solver_failures')))
settings = settings_for('A4', 250.); config = configured(base, settings, 1., 1.)
scenario = Scenario(Path(case['case_id']), SRC / 'assets' / FIXED_MAP, tuple(map(tuple, case['starts'])), tuple(map(tuple, case['goals'])))
C = Counter(); orig = RD.consistency_snapshot
def hooked(positions, truth, *aa, **k):
    snap = orig(positions, truth, *aa, **k)
    for p in snap.get('pairs', ()):
        age = max(p['self_age_seconds'], p['peer_age_seconds'])
        act = int(not truth[p['receiver']]['completed'] and not truth[p['peer']]['completed'])
        C[(min(int(age / AOI_W), AOI_MAX), act, int(p['relevant']), int(p['e2e_ordering_error']),
           int(p.get('uplink_ordering_error', 0)), int(p.get('downlink_ordering_error', 0)),
           int(bool(p.get('physical_relevant'))), int(bool(p.get('ordering_error'))))] += 1
    return snap
RD.consistency_snapshot = hooked
t0 = time.monotonic()
row, trace = run_frequency_episode(config, model, case, torch.device('cpu'), payload.get('learner_updates', 0), scenario, condition, settings)
sim = time.monotonic() - t0
row, tasks = measure(row, trace, 'A4', dict(id='W1', latency=1., energy=1.), condition)
S = Counter()
for e in trace['oracle_events']:
    if e['event'] != 'controlled_peer_shadow': continue
    fin = {int(k): bool(v.get('completed')) for k, v in e['fresh_states'].items()}
    for p in e['pairs']:
        S[(min(int(max(p['self_age_seconds'], p['peer_age_seconds']) / AOI_W), AOI_MAX), int(bool(p['physical_relevant'])),
           int(p['ordering_error']), int(fin.get(p['peer'], False)), int(bool(p['peer_completed'])), int(bool(p['self_completed'])))] += 1
F = Counter()  # omega_aoi_extract-compatible aoi (6 fields) and snapshot physical counts
for k, v in C.items(): F[k[:6]] += v
row_ref = lambda k: int(float(ref[k])) if ref.get(k) not in (None, '') else None
ref = next((r for r in csv.DictReader(open('/workspace/SANGIN-dev/results/redesign/20261027-004052/episodes.csv'))
            if r['treatment'] == 'A4' and r['route'] == a.route and r['case_id'] == a.case_id), {})
tot = dict(ordering_pairs_verified=sum(F.values()), ordering_errors=sum(v * k[3] for k, v in F.items()),
           relevant_pairs=sum(v * k[2] for k, v in F.items()), fov_errors=sum(v * k[2] * k[3] for k, v in F.items()),
           snap_physical_relevant_pairs=sum(v * k[6] for k, v in C.items()), snap_physical_relevant_ordering_errors=sum(v * k[6] * k[7] for k, v in C.items()),
           physical_relevant_pairs=sum(v * k[1] for k, v in S.items()), physical_fov_order_errors=sum(v * k[1] * k[2] for k, v in S.items()))
chk = {k: (tot[k], row_ref(k)) for k in ('ordering_pairs_verified', 'ordering_errors', 'relevant_pairs', 'fov_errors')}
chk['physical_relevant_pairs'] = (tot['snap_physical_relevant_pairs'], row_ref('physical_relevant_pairs'))
chk['physical_relevant_ordering_errors'] = (tot['snap_physical_relevant_ordering_errors'], row_ref('physical_relevant_ordering_errors'))
chk['steps'] = (row['steps'], row_ref('steps')); chk['success'] = (str(row['success']), ref.get('success'))
ok = all(x == y for x, y in chk.values())
rec = dict(run='redesign_A4', condition=f'A4/{a.route}', route=a.route, case_id=a.case_id, num_agents=case['num_agents'], delta_seconds=settings.control_interval_seconds,
           success=row['success'], steps=row['steps'], sim_seconds=round(sim, 1), peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
           validated=ok, check=chk, totals=tot,
           aoi_fields=['aoi_bin', 'active', 'relevant', 'e2e_error', 'ul_error', 'dl_error', 'count'], aoi=[list(k) + [v] for k, v in sorted(F.items())],
           shadow_fields=['aoi_bin', 'physical_relevant', 'ordering_error', 'peer_finished_true', 'peer_finished_cached', 'self_finished_cached', 'count'],
           shadow=[list(k) + [v] for k, v in sorted(S.items())])
with open(OUT / 'A4_episodes.jsonl', 'a') as o: o.write(json.dumps(rec, default=str) + '\n')
print(a.route, a.case_id, 'steps', row['steps'], 'validated', ok, {k: v for k, v in chk.items() if v[0] != v[1]}, f"{sim:.0f}s rss {rec['peak_rss_gib']}GiB", flush=True)
