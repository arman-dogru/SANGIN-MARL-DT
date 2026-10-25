"""Local population extension of the existing parameter-only UAV CPU set."""
from collections import Counter,defaultdict,deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,gc,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from panama_uav.config import ProjectConfig,FIXED_MAP
from panama_uav.records import sha256,write_json,append_jsonl
from panama_uav.resource_study import study_cases
from panama_uav.communication_report import csv_table,paired_changes

LABEL='Parameter-only pilot results'
BASE=Path(__file__).resolve().parents[1]/'output/uav6_pilot/uav6-20261024-01'

def module(path):
 spec=importlib.util.spec_from_file_location('base_pilot',path)
 result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
 return result

def verify(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text())
 for name,key in (('assets/checkpoint.pt','model_sha256'),('assets/orbit.npz','timeline_sha256'),
   ('assets/'+FIXED_MAP,'map_sha256'),('cases.json','cases_sha256'),
   ('code/base_pilot.py','base_harness_sha256'),('code/extend_uav6.py','harness_sha256')):
  if sha256(root/name)!=m[key]:raise ValueError('Frozen input changed: '+name)
 for name,digest in m['code_sha256'].items():
  if sha256(root/'code/panama_uav'/name)!=digest:raise ValueError('Frozen implementation changed')
 if sha256(Path(m['base_experiment'])/'manifest.json')!=m['base_manifest_sha256']:
  raise ValueError('Original experiment manifest changed')
 return m

def prepare(root,base,cases,counts,cap,workers):
 import torch
 from panama_uav.training import load_model
 root,base=Path(root).resolve(),Path(base).resolve()
 if root.exists():raise ValueError('Use a fresh extension directory')
 if not 1<=cases<=5 or tuple(counts) not in ((16,32),(32,)) or not 1<=cap<=250:
  raise ValueError('Use the bounded 16/32-agent cohort')
 m=json.loads((base/'manifest.json').read_text())
 assert json.loads((base/'status.json').read_text())['status']=='complete'
 torch.set_num_threads(1)
 original=ProjectConfig.from_dict(m['configuration'])
 _,payload=load_model(base/'assets/checkpoint.pt',original,torch.device('cpu'),require_qualified=True)
 assets=root/'assets';assets.mkdir(parents=True)
 for name in ('checkpoint.pt','orbit.npz',FIXED_MAP):
  target=assets/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/'assets'/name,target)
 cfg=replace(original,project_root=assets,communication=replace(original.communication,leo_timeline=str(assets/'orbit.npz')))
 bank=study_cases(cfg,cases,counts,250)
 for case in bank:case.update(map_path=str(assets/FIXED_MAP),map_sha256=sha256(assets/FIXED_MAP),scenario='random_mapf')
 write_json(root/'cases.json',bank)
 shutil.copytree(base/'code/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copy2(base/'code/run_uav6_pilot.py',root/'code/base_pilot.py')
 # Buffer JSON encoding without retaining a second multi-gigabyte trace copy.
 target=root/'code/base_pilot.py'
 text=target.read_text()
 old='            stream.write(json.dumps(trace, allow_nan=False, separators=(",", ":")).encode())'
 new='''            parts = []
            size = 0
            for part in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(trace):
                parts.append(part)
                size += len(part)
                if size >= 1048576:
                    stream.write("".join(parts).encode())
                    parts, size = [], 0
            if parts:
                stream.write("".join(parts).encode())'''
 if text.count(old)!=1:raise ValueError('Archive buffering adapter did not match')
 target.write_text(text.replace(old,new),encoding='utf-8')
 shutil.copy2(__file__,root/'code/extend_uav6.py')
 for name in ('base.toml','pilot.toml'):shutil.copy2(base/name,root/name)
 h=module(root/'code/base_pilot.py');h.parameter_change(root/'base.toml',root/'pilot.toml')
 manifest=dict(protocol='uav6_population_extension_v1',label=LABEL,base_experiment=str(base),
  base_manifest_sha256=sha256(base/'manifest.json'),configuration=cfg.to_dict(),conditions=m['conditions'],
  populations=list(counts),cases_per_population=cases,episode_cap=cap,workers=workers,budget_seconds=14400,
  expected_episodes=len(bank)*len(m['conditions']),learner_updates=payload.get('learner_updates',0),
  model_sha256=m['model_sha256'],timeline_sha256=m['timeline_sha256'],map_sha256=m['map_sha256'],
  cases_sha256=sha256(root/'cases.json'),policy_updates=0,source_commit=m['source_commit'],
  code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
  base_harness_sha256=sha256(root/'code/base_pilot.py'),harness_sha256=sha256(root/'code/extend_uav6.py'),
  protected_sha256=m['protected_sha256'],project_root=m['project_root'],
  combined_populations=[2,4,8,*counts] if cap==250 else list(counts),
  orbital_offsets='unchanged study_cases formula with 250-step geometry horizon',
  inference_device='cpu',threads_per_worker=1,archive_encoding='buffered standard JSON; same payload bytes')
 write_json(root/'manifest.json',manifest)
 write_json(root/'status.json',dict(status='prepared',completed_episodes=0,expected_episodes=manifest['expected_episodes']))
 print(json.dumps(dict(populations=counts,cases_per_population=cases,conditions=len(m['conditions']),episodes=manifest['expected_episodes'])),flush=True)
 return manifest

def initialize(root):
 import torch
 from panama_uav.training import load_model
 global H
 root=Path(root);m=verify(root)
 H=module(root/'code/base_pilot.py')
 H.ROOT=root;H.MANIFEST=m;H.CONFIG=ProjectConfig.from_dict(m['configuration'])
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
 H.MODEL,_=load_model(root/'assets/checkpoint.pt',H.CONFIG,torch.device('cpu'),require_qualified=True)

def episode(argument):
 result=H.episode(argument)
 gc.collect()
 return result

def run(root):
 root=Path(root).resolve();m=verify(root)
 if (root/'episodes').exists():raise ValueError('Existing episodes cannot be overwritten')
 bank=json.loads((root/'cases.json').read_text());started=time.monotonic();completed=0
 def status(state,**extra):
  write_json(root/'status.json',dict(status=state,completed_episodes=completed,expected_episodes=m['expected_episodes'],
   wall_seconds=time.monotonic()-started,**extra))
 status('running')
 try:
  arguments=iter((condition,case) for case in bank for condition in m['conditions'])
  with ProcessPoolExecutor(max_workers=m['workers'],initializer=initialize,initargs=(str(root),)) as executor:
   pending=deque()
   for _ in range(m['workers']):
    item=next(arguments,None)
    if item is not None:pending.append(executor.submit(episode,item))
   while pending:
    packet=pending.popleft().result(timeout=max(1.,m['budget_seconds']-(time.monotonic()-started)))
    for name,key in (('episodes','row'),('tasks','tasks'),('rounds','rounds'),('audits','audit')):
     values=packet[key] if isinstance(packet[key],list) else [packet[key]]
     for value in values:append_jsonl(root/(name+'.jsonl'),value)
    completed+=1
    status('running',latest_case=packet['row']['case_id'],latest_condition=packet['row']['condition'])
    if time.monotonic()-started>m['budget_seconds']:raise TimeoutError('Bounded extension runtime reached')
    item=next(arguments,None)
    if item is not None:pending.append(executor.submit(episode,item))
  report(root)
  status('complete')
 except Exception as error:
  status('error',error=str(error));raise

def aggregate(group,h):
 rows=[p['row'] for p in group];tasks=[t for p in group for t in p['tasks']]
 rounds=[r for p in group for r in p['rounds'] if r['cycle']==0]
 horizon=sum(r['observed_seconds'] for r in rows)
 result=dict(label=LABEL,condition=rows[0]['condition'],num_agents=rows[0]['num_agents'],cases=len(rows),
  completed_missions=sum(r['success'] for r in rows),**h.describe(tasks))
 result.update(first_round_uav_mean=float(np.mean([r['uav_tasks'] for r in rounds])),
  first_round_cloud_mean=float(np.mean([r['cloud_tasks'] for r in rounds])),
  makespan_steps=h._mean(r['makespan_seconds'] for r in rows),waiting_steps=h._mean(r['waiting_completed_steps'] for r in rows),
  cutoff_waits=sum(r['command_cutoff_misses'] for r in rows),
  modeled_energy_total_j=sum(r['modeled_energy_lower_j'] for r in rows),
  modeled_energy_mean_episode_j=h._mean(r['modeled_energy_lower_j'] for r in rows),
  robot_tx_energy_j=sum(r['robot_tx_energy_j'] for r in rows),uav_tx_energy_j=sum(r['drone_tx_energy_j'] for r in rows),
  uav_compute_energy_j=sum(r['drone_compute_energy_j'] for r in rows),
  unavailable_admission_tasks=sum(t['backhaul_available_at_admission'] is False for t in tasks),
  handover_windows=sum(r['handover_service_windows'] for r in rows),satellite_attachment_events=sum(r['satellite_attachment_events'] for r in rows))
 for pool in ('drone','cloud'):
  result[pool+'_utilization']=sum(r[pool+'_utilization_seconds'] for r in rows)/horizon
  result[pool+'_concurrent_mean']=sum(r[pool+'_concurrent_job_seconds'] for r in rows)/horizon
  occupied=sum(r[pool+'_occupied_seconds'] for r in rows)
  result[pool+'_concurrent_when_busy']=sum(r[pool+'_concurrent_job_seconds'] for r in rows)/occupied if occupied else None
  result[pool+'_concurrent_max']=max(r[pool+'_max_concurrent_jobs'] for r in rows)
 return result

def report(root):
 root=Path(root).resolve();m=verify(root);h=module(root/'code/base_pilot.py')
 cases={c['case_id']:c for c in json.loads((root/'cases.json').read_text())}
 packets=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*/*.result.json'))]
 expected={(c['name'],case) for c in m['conditions'] for case in cases}
 identities=[(p['row']['condition'],p['row']['case_id']) for p in packets]
 if len(identities)!=len(set(identities)) or set(identities)!=expected:raise ValueError('Incomplete paired cohort')
 grouped=defaultdict(list)
 for p in packets:
  row=p['row'];a=p['audit'];tasks=p['tasks']
  assert p['manifest_sha256']==sha256(root/'manifest.json') and a['status']=='passed' and sha256(a['archive'])==a['archive_sha256']
  assert all(row[k]==cases[row['case_id']][k] for k in h.IDENTITY)
  assert len(tasks)==row['tasks_generated'] and len({t['job_id'] for t in tasks})==len(tasks)
  statuses=Counter(t['deadline_status'] for t in tasks)
  assert all(statuses[s]==row['deadline_commands_'+s] for s in ('on_time','late','undelivered'))
  assert row['command_cutoff_misses']==sum(t['cutoff_induced_wait'] for t in tasks)
  grouped[row['condition'],row['num_agents']].append(p)
 summary=[aggregate(group,h) for _,group in sorted(grouped.items())]
 contrasts=[];exposures=[];strata=defaultdict(list)
 for (condition,n),group in sorted(grouped.items()):
  rows=[p['row'] for p in group];tasks=[t for p in group for t in p['tasks']]
  if condition!='allocated-1ghz':
   reference=[p['row'] for p in grouped['allocated-1ghz',n]]
   for metric in ('mission_success_rate','command_on_time_rate','makespan_seconds','waiting_steps','modeled_energy_j'):
    contrasts.append(dict(condition=condition,reference='allocated-1ghz',num_agents=n,**paired_changes(rows,reference,metric)))
  for name,subset in (('handover_at_sensing',[t for t in tasks if t['handover_at_sensing']]),
   ('outside_handover_at_sensing',[t for t in tasks if not t['handover_at_sensing']]),
   ('unavailable_at_admission',[t for t in tasks if t['backhaul_available_at_admission'] is False]),
   ('available_at_admission',[t for t in tasks if t['backhaul_available_at_admission'] is True])):
   exposures.append(dict(label=LABEL,condition=condition,num_agents=n,subset=name,
    cutoff_waits=sum(t['cutoff_induced_wait'] for t in subset),**h.describe(subset)))
  for p in group:
   active={r['cycle']:r['tasks'] for r in p['rounds']}
   for t in p['tasks']:strata[condition,n,active[t['cycle']]].append(t)
 load=[dict(label=LABEL,condition=c,num_agents=n,active_robots=a,**h.describe(tasks)) for (c,n,a),tasks in sorted(strata.items())]
 for name,records in (('summary',summary),('comparisons',contrasts),('handover',exposures),('placement_by_active_count',load)):
  csv_table(root/(name+'.csv'),records)
 write_json(root/'summary.json',summary)
 for name,key in (('episodes','row'),('tasks','tasks'),('rounds','rounds'),('audits','audit')):
  records=[item for p in packets for item in (p[key] if isinstance(p[key],list) else [p[key]])]
  csv_table(root/(name+'.csv'),records)
 combined=summary
 if m['episode_cap']==250:
  base=Path(m['base_experiment']);old=json.loads((base/'summary.json').read_text())
  old_packets=[json.loads(p.read_text()) for p in sorted((base/'episodes').glob('*/*.result.json'))]
  old_groups=defaultdict(list)
  for p in old_packets:old_groups[p['row']['condition'],p['row']['num_agents']].append(p)
  for target in old:
   check=aggregate(old_groups[target['condition'],target['num_agents']],h)
   for key,value in target.items():
    if isinstance(value,(float,int)) and value is not None:assert np.isclose(value,check[key],rtol=1e-11,atol=1e-11)
    else:assert value==check[key]
  combined=old+summary
  csv_table(root/'combined_episodes.csv',[p['row'] for p in old_packets+packets])
  old_comparisons=[]
  import csv
  with (base/'comparisons.csv').open() as stream:old_comparisons=list(csv.DictReader(stream))
  csv_table(root/'combined_comparisons.csv',old_comparisons+contrasts)
 csv_table(root/'combined_summary.csv',combined);write_json(root/'combined_summary.json',combined)
 changed=[name for name,digest in m['protected_sha256'].items() if sha256(Path(m['project_root'])/name)!=digest]
 assert not changed,changed
 write_json(root/'validation.json',dict(status='passed',label=LABEL,episodes=len(packets),
  replay_audits=len(packets),tasks=sum(len(p['tasks']) for p in packets),rounds=sum(len(p['rounds']) for p in packets),
  paired_case_identity_verified=True,deadline_partition_verified=True,original_summary_reproduced=m['episode_cap']==250,
  preserved_files=len(m['protected_sha256']),policy_updates=0))
 plot(root,combined)
 lines=['# Extended UAV CPU experiment set','',LABEL+'.','',
  '| Robots | Condition | Missions | UAV/cloud tasks | Cloud % | E2E mean/p50/p95 ms | On-time % | Makespan | Waiting | Energy J/episode |',
  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
 for r in sorted(combined,key=lambda r:(r['num_agents'],[c['name'] for c in m['conditions']].index(r['condition']))):
  fmt=lambda v:'â€”' if v is None else f'{v:.4g}'
  lines.append('| '+' | '.join(map(str,(r['num_agents'],r['condition'],f"{r['completed_missions']}/{r['cases']}",
   f"{r['uav_tasks']}/{r['cloud_tasks']}",fmt(100*r['cloud_fraction']),
   '/'.join(fmt(1000*r[k]) for k in ('observed_mean_s','observed_p50_s','observed_p95_s')),
   fmt(100*r['on_time_fraction']),fmt(r['makespan_steps']),fmt(r['waiting_steps']),fmt(r['modeled_energy_mean_episode_j']))))+' |')
 lines.extend(['','Parameters: UAV 6e9 cycles/s (1e9 for original allocator), cloud 1e10 cycles/s; 100000 bits at 1000 cycles/bit; 10/50 Mbps LEO; kappa 1e-28; energy/latency coefficients 1/1; 250 ms command cutoffs, one-second control, 250-step cap. Original frozen model, timeline and production kernels are retained.','',
  'Existing 2/4/8-agent records are reused unchanged. New populations use five paired cases and the same seed/geometry generator. Predicted solver-snapshot latency is separate from observed sensing-to-command delivery. CPU utilization/concurrent load, active-peer AoI, handover exposure, fixed prior assignments and energy are retained in the tables.','',
  'Files: combined_summary.csv/json, combined_episodes.csv, combined_comparisons.csv; summary.csv, handover.csv, placement_by_active_count.csv, episodes.csv, tasks.csv, rounds.csv and validation.json for the added populations. Raw audited per-case archives remain under episodes/. No learning updates occur.'])
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return summary

def plot(root,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(11,6.3),layout='constrained')
 colors=('#666666','#009E73','#0072B2','#D55E00');names=('allocated-1ghz','allocated-6ghz','uav-6ghz','cloud-6ghz')
 labels=('Allocator 1 GHz','Allocator 6 GHz','All UAV 6 GHz','All cloud')
 fields=(('cloud_fraction','Cloud task fraction'),('observed_p95_s','Observed E2E command p95 (s)'),
  ('on_time_fraction','Commands within 250 ms'),('drone_utilization','UAV CPU utilization'),
  ('cloud_utilization','Cloud CPU utilization'),('completed_missions','Mission success fraction'))
 for condition,color,label in zip(names,colors,labels):
  rows=sorted([r for r in summary if r['condition']==condition],key=lambda r:r['num_agents'])
  for ax,(field,ylabel) in zip(axes.flat,fields):
   values=[r[field]/r['cases'] if field=='completed_missions' else r[field] for r in rows]
   ax.plot([r['num_agents'] for r in rows],values,color=color,marker='o',label=label)
 for ax,(field,label) in zip(axes.flat,fields):
  ax.set(xlabel='Robots',ylabel=label,xticks=sorted({r['num_agents'] for r in summary}));ax.grid(alpha=.2)
  if field in ('cloud_fraction','on_time_fraction','completed_missions'):ax.set_ylim(-.03,1.03)
 axes[0,0].legend(frameon=False,fontsize=8);axes[0,1].axhline(.25,color='black',ls=':',lw=1)
 fig.suptitle('Parameter-only pilot results â€” extended experiment set',fontsize=12)
 (root/'figures').mkdir(exist_ok=True)
 for ext in ('pdf','svg','png'):fig.savefig(root/'figures'/('combined_diagnostics.'+ext),dpi=300,metadata={'CreationDate':None} if ext=='pdf' else None)
 plt.close(fig)

def launch(root):
 import psutil
 root=Path(root).resolve();verify(root)
 if (root/'launch.json').exists():raise ValueError('Launch already recorded')
 env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 cmd=[sys.executable,str(root/'code/extend_uav6.py'),'--worker',str(root)]
 with (root/'stdout.log').open('a') as out,(root/'stderr.log').open('a') as err:
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 info=dict(pid=p.pid,creation_time=psutil.Process(p.pid).create_time(),command=cmd);write_json(root/'launch.json',info)
 print(json.dumps(info),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--base',type=Path,default=BASE)
 p.add_argument('--cases',type=int,default=5);p.add_argument('--counts',type=int,nargs='+',default=(16,32))
 p.add_argument('--cap',type=int,default=250);p.add_argument('--workers',type=int,default=4)
 p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--worker',type=Path)
 a=p.parse_args()
 if a.worker:run(a.worker)
 else:
  if a.prepare:prepare(a.root,a.base,a.cases,tuple(a.counts),a.cap,a.workers)
  if a.launch:launch(a.root)
  if a.report:report(a.root)
