"""Run one study job (box-local, leo_study patched copy). Writes <out>/<job_id>/result.json atomically."""
import sys, json, time, os, argparse, gzip, resource
os.environ.setdefault('OMP_NUM_THREADS','1')
if '--keep-trace' not in sys.argv:os.environ['LEO_STUDY_LEAN']='1'  # audit-only payloads dropped; metrics unchanged
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np, torch
from panama_uav.config import load_config, FIXED_MAP
from panama_uav.training import load_model
from panama_uav.resource_study import study_cases
from panama_uav.inference_controls import InferenceCondition
from panama_uav.scenarios import Scenario
from panama_uav.redesign import run_frequency_episode, settings_for, configured
from panama_uav.redesign_metrics import measure

STUDY='/workspace/leo_study'
CONFIG=STUDY+'/repo/configs/paper_remote_dt_250ms_uav6_cached.toml'
TIMELINE=STUDY+'/inputs/tremblant_full.npz'
MODEL='/workspace/panama_models/model8.pt'

def condition_for(job):
    route=job['route']
    if job['network']=='ideal':
        return InferenceCondition(f'{route}__ideal_network',route,axis='network_reference',ideal_communication=True)
    d=float(job.get('downlink_seconds',0.))
    if d:
        return InferenceCondition(f'{route}__downlink-{d:g}',route,axis='downlink',level=d,downlink_seconds=d)
    return InferenceCondition(f'{route}__nominal',route)

def base_config():
    cfg=load_config(CONFIG)
    return replace(cfg,communication=replace(cfg.communication,leo_timeline=TIMELINE))

TRIM_FILE=Path(STUDY)/'runs/trim.json'
HEAVY_LOCK=Path(STUDY)/'runs/heavy.lock'

def trimmed(job):
    """6-hour budget rule: trimmed 32-agent cells are listed in runs/trim.json (documented in README)."""
    if not TRIM_FILE.exists():return None
    for rule in json.loads(TRIM_FILE.read_text())['rules']:
        if all(job.get(k)==v if not isinstance(v,list) else job.get(k) in v for k,v in rule['match'].items()):return rule['reason']
    return None

def run(job,out,keep_trace=False):
    reason=trimmed(job)
    if reason:
        folder=Path(out)/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
        (folder/'trimmed.txt').write_text(reason);print('TRIMMED',reason);return None
    lock=None
    if (job.get('downlink_seconds',0)>=.3 and job['num_agents']>=16) or job['num_agents']>=32:
        import fcntl
        lock=open(HEAVY_LOCK,'w');fcntl.flock(lock,fcntl.LOCK_EX)  # memory: one +300 ms N>=16 (or any N=32: failing 32-agent episodes grow to 8-9 GB) episode at a time
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    folder=Path(out)/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
    final=folder/'result.json'
    if final.exists():return json.loads(final.read_text())
    cfg=base_config()
    model,payload=load_model(MODEL,cfg,torch.device('cpu'),require_qualified=False)
    case=next(c for c in study_cases(cfg,job['case']+1,(job['num_agents'],),250) if c['case']==job['case'])
    cond=condition_for(job)
    base=replace(cfg,communication=replace(cfg.communication,
        mission_route='cloud' if job['route'] in ('allocated','cloud') else 'uav',
        backhaul_time_offset_seconds=cfg.communication.backhaul_time_offset_seconds+case['orbital_offset_seconds']),
        resource_allocation=replace(cfg.resource_allocation,failure_directory=str(folder/'solver_failures')))
    settings=settings_for(job['treatment'],250.)
    config=configured(base,settings,1.,1.)
    scen=Scenario(Path(case['case_id']),cfg.resolve(FIXED_MAP),tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
    t0=time.monotonic()
    row,trace=run_frequency_episode(config,model,case,torch.device('cpu'),payload.get('learner_updates',0),scen,cond,settings)
    sim=time.monotonic()-t0
    row,tasks=measure(row,trace,job['treatment'],dict(id='W1',latency=1.,energy=1.),cond)
    row.update(job=job['job_id'],network=job['network'],downlink_added_seconds=float(job.get('downlink_seconds',0.)),
        sim_wall_seconds=sim,measure_wall_seconds=time.monotonic()-t0-sim,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_sha256='0e999c445d0855c1d0ba92d667f045910139ea01ce82cff18b847aa8c1c0e612',
        speed_assignments=json.dumps(trace['remote_settings']['speed_assignments']))
    keep=('job_id','agent','cycle','speed_class','e2e_latency_seconds','cycle_latency_seconds','transport_latency_seconds',
          'action_source','command_used','executed_x','location','mission_success')
    tasks=[{k:t.get(k) for k in keep} for t in tasks]
    packet=dict(job=job,row=row,tasks=tasks,pair_stats=trace['pair_stats'])
    if keep_trace:
        import pickle;pickle.dump(trace,gzip.open(folder/'trace.pkl.gz','wb'))
    tmp=folder/'result.json.tmp'
    tmp.write_text(json.dumps(packet,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
    os.replace(tmp,final)
    return packet

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--out',default=STUDY+'/runs/main');p.add_argument('--keep-trace',action='store_true')
    a=p.parse_args()
    job=json.loads(Path(a.job).read_text()) if a.job.endswith('.json') else json.loads(a.job)
    packet=run(job,a.out,a.keep_trace)
    if packet is None:sys.exit(0)
    r=packet['row']
    print(json.dumps({k:r.get(k) for k in ('job','success','makespan_seconds','sim_wall_seconds','speed_limited_nonmovement','cached_fallback_decisions',
        'command_cutoff_misses','ordering_errors','missed_physical_conflicts','peak_rss_bytes')}))
