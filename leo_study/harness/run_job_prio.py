"""Priority-instrumented copy of run_job.py (leo_study). Output: runs/prio/<prio__job_id>/result.json.

Runtime monkeypatch only (no repo source edits): wraps RemoteDTCommunication.update to snapshot, once per
control cycle, every robot's TRUE environment priority (environment._refresh_priorities: distance to goal
+ id*1e-6; settled robots = width*height), completed flag, rank among ACTIVE robots (0 = highest priority =
lowest value), normalised rank in [0,1] (rank/(n_active-1); 0 if one active), and the physical allocator
weight allocation.priority_weights(truth) in [1,4]. Each task record (agent, cycle) is joined to its
cycle snapshot; the DT-side weight actually used by the allocator for that job is added when available.
Behaviour is unchanged: the wrapper only reads state after the original update() returns.
"""
import sys, json, time, os, argparse, resource
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['LEO_STUDY_LEAN'] = '1'
sys.path.insert(0, '/workspace/leo_study/harness')
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
import run_job as RJ                                   # same config/model/condition helpers as the main study
from panama_uav import remote as REMOTE
from panama_uav.allocation import priority_weights
from panama_uav.config import FIXED_MAP
from panama_uav.training import load_model
from panama_uav.resource_study import study_cases
from panama_uav.scenarios import Scenario
from panama_uav.redesign import run_frequency_episode, settings_for, configured
from panama_uav.redesign_metrics import measure

SNAP = {}      # cycle -> {agent: dict(priority, completed, rank, norm_rank, phys_weight)}
JOBW = {}      # job_id -> allocator weight used (DT view)
_orig_update = REMOTE.RemoteDTCommunication.update
_orig_admit = REMOTE.RemoteDTCommunication._admit

def _update(self, agents, now, *a, **k):
    out = _orig_update(self, agents, now, *a, **k)
    cyc = int(self.current_cycle)
    if cyc not in SNAP:
        truth = self.truth; ids = sorted(truth)
        pr = {i: float(truth[i]['priority']) for i in ids}; done = {i: bool(truth[i]['completed']) for i in ids}
        active = sorted((i for i in ids if not done[i]), key=lambda i: pr[i])
        rank = {i: r for r, i in enumerate(active)}; na = len(active)
        w = priority_weights([pr[i] for i in ids], [done[i] for i in ids], self.parameters.weight_min, self.parameters.weight_max)
        SNAP[cyc] = {i: dict(priority=pr[i], completed=done[i], active_rank=rank.get(i), n_active=na,
                             norm_rank=(None if i not in rank else (rank[i] / (na - 1) if na > 1 else 0.0)),
                             phys_weight=float(w[k_])) for k_, i in enumerate(ids)}
    return out

def _admit(self, payload):
    out = _orig_admit(self, payload)
    job = self.jobs.get(payload.get('job_id'))
    if job is not None and 'weight' in job: JOBW[str(job['job_id'])] = float(job['weight'])
    return out

REMOTE.RemoteDTCommunication.update = _update
REMOTE.RemoteDTCommunication._admit = _admit

def run(job, out):
    lock = None
    if job['num_agents'] >= 32:
        import fcntl
        lock = open(RJ.HEAVY_LOCK, 'w'); fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(42)
    folder = Path(out) / job['job_id']; folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'result.json'
    if final.exists(): return json.loads(final.read_text())
    cfg = RJ.base_config()
    model, payload = load_model(RJ.MODEL, cfg, torch.device('cpu'), require_qualified=False)
    case = next(c for c in study_cases(cfg, job['case'] + 1, (job['num_agents'],), 250) if c['case'] == job['case'])
    cond = RJ.condition_for(job)
    base = replace(cfg, communication=replace(cfg.communication,
        mission_route='cloud' if job['route'] in ('allocated', 'cloud') else 'uav',
        backhaul_time_offset_seconds=cfg.communication.backhaul_time_offset_seconds + case['orbital_offset_seconds']),
        resource_allocation=replace(cfg.resource_allocation, failure_directory=str(folder / 'solver_failures')))
    settings = settings_for(job['treatment'], 250.)
    config = configured(base, settings, 1., 1.)
    scen = Scenario(Path(case['case_id']), cfg.resolve(FIXED_MAP), tuple(map(tuple, case['starts'])), tuple(map(tuple, case['goals'])))
    t0 = time.monotonic()
    row, trace = run_frequency_episode(config, model, case, torch.device('cpu'), payload.get('learner_updates', 0), scen, cond, settings)
    sim = time.monotonic() - t0
    row, tasks = measure(row, trace, job['treatment'], dict(id='W1', latency=1., energy=1.), cond)
    row.update(job=job['job_id'], network=job['network'], downlink_added_seconds=float(job.get('downlink_seconds', 0.)),
        sim_wall_seconds=sim, measure_wall_seconds=time.monotonic() - t0 - sim,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        model_sha256='0e999c445d0855c1d0ba92d667f045910139ea01ce82cff18b847aa8c1c0e612',
        speed_assignments=json.dumps(trace['remote_settings']['speed_assignments']))
    keep = ('job_id', 'agent', 'cycle', 'speed_class', 'e2e_latency_seconds', 'cycle_latency_seconds', 'transport_latency_seconds',
            'action_source', 'command_used', 'executed_x', 'location', 'mission_success')
    out_tasks, unmatched = [], 0
    for t in tasks:
        r = {k: t.get(k) for k in keep}
        s = SNAP.get(int(t['cycle']), {}).get(int(t['agent']))
        if s is None: unmatched += 1; s = {}
        r.update(priority=s.get('priority'), completed=s.get('completed'), active_rank=s.get('active_rank'),
                 n_active=s.get('n_active'), norm_rank=s.get('norm_rank'), phys_weight=s.get('phys_weight'),
                 alloc_weight=JOBW.get(str(t.get('job_id'))))
        out_tasks.append(r)
    packet = dict(job=job, row=row, tasks=out_tasks, pair_stats=trace['pair_stats'],
                  prio_meta=dict(cycles_snapshotted=len(SNAP), tasks=len(out_tasks), unmatched_tasks=unmatched,
                                 jobs_with_alloc_weight=len(JOBW)))
    tmp = folder / 'result.json.tmp'
    tmp.write_text(json.dumps(packet, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    os.replace(tmp, final)
    return packet

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('job'); p.add_argument('--out', default=RJ.STUDY + '/runs/prio')
    a = p.parse_args()
    job = json.loads(Path(a.job).read_text()) if a.job.endswith('.json') else json.loads(a.job)
    packet = run(job, a.out); r = packet['row']
    print(json.dumps(dict(job=r.get('job'), success=r.get('success'), sim_wall_seconds=r.get('sim_wall_seconds'), **packet['prio_meta'])))
