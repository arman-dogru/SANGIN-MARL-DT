"""Local smoke harness (box only): one redesign-style frozen-inference episode per job."""
import sys, json, time, os, argparse
from dataclasses import replace
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np, torch
sys.path.insert(0,'/workspace/SANGIN-dev/scripts')
from panama_uav.config import load_config, FIXED_MAP
from panama_uav.training import load_model
from panama_uav.resource_study import study_cases
from panama_uav.inference_controls import InferenceCondition
from panama_uav.scenarios import Scenario
from panama_uav.redesign import run_frequency_episode, settings_for, configured
from panama_uav.redesign_metrics import measure

ROUTES={'allocated':'allocated-6ghz','edge':'uav-6ghz','cloud':'cloud-6ghz'}
p=argparse.ArgumentParser();p.add_argument('--agents',type=int,nargs='+',default=[2,4]);p.add_argument('--cases',type=int,default=1)
p.add_argument('--treatment',default='A4');p.add_argument('--route',default='allocated');p.add_argument('--model',default='/workspace/panama_models/01M5XQ25TE32MCTXN3GT8K20CK.pt')
p.add_argument('--horizon',type=float,default=250.);p.add_argument('--out',default='/workspace/leo_study/smoke');p.add_argument('--diagnostic',action='store_true')
a=p.parse_args()
torch.set_num_threads(1)
cfg=load_config('/workspace/SANGIN-dev/configs/paper_remote_dt_250ms_uav6_cached.toml')
cfg=replace(cfg,communication=replace(cfg.communication,leo_timeline='/workspace/leo_study/inputs/tremblant_full.npz'))
model,payload=load_model(a.model,cfg,torch.device('cpu'),require_qualified=not a.diagnostic)
cases=study_cases(cfg,a.cases,tuple(a.agents),250)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
for case in cases:
    cond=InferenceCondition(ROUTES[a.route],a.route)
    base=replace(cfg,communication=replace(cfg.communication,mission_route='cloud' if a.route in ('allocated','cloud') else 'uav',
        backhaul_time_offset_seconds=cfg.communication.backhaul_time_offset_seconds+case['orbital_offset_seconds']))
    settings=settings_for(a.treatment,a.horizon);config=configured(base,settings,1.,1.)
    scen=Scenario(Path(case['case_id']),cfg.resolve(FIXED_MAP),tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
    t0=time.monotonic()
    row,trace=run_frequency_episode(config,model,case,torch.device('cpu'),payload.get('learner_updates',0),scen,cond,settings)
    sim=time.monotonic()-t0
    row,tasks=measure(row,trace,a.treatment,dict(id='W1',latency=1.,energy=1.),cond)
    row.update(treatment=a.treatment,route=a.route,sim_wall_seconds=sim,measure_wall_seconds=time.monotonic()-t0-sim)
    name=f"{a.treatment}_{a.route}_{case['case_id']}"
    json.dump(row,open(out/f'{name}.row.json','w'),default=str)
    import pickle,gzip;pickle.dump(trace,gzip.open(out/f'{name}.trace.pkl.gz','wb'))
    keys=['num_agents','success','makespan_seconds','sim_wall_seconds','uplink_errors','downlink_errors','ordering_errors','xi_fov','xi_aoi','mean_peer_aoi',
          'movement_priority_reversals','missed_physical_conflicts','false_positive_conflicts','cached_fallback_decisions','command_cutoff_misses',
          'command_transport_p95_seconds','modeled_energy_upper_j','waiting_steps','collision_attempts','peak_rss_bytes']
    print(name,{k:row.get(k) for k in keys},flush=True)
