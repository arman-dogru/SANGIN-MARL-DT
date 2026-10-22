"""Clean, disjoint-map DQN curriculum with a frozen final held-out evaluation."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .config import ExperimentMode, LearningCondition, ProjectConfig, COMMUNICATION_SCHEMA
from .learning import _learn
from .models import choose_device, cpu_state_dict, select_actions
from .phase1_data import FAMILIES, case_for, scenario_for, verify_dataset
from .phase1_environment import CleanMAPFEnvironment, epoch_metrics
from .records import append_jsonl, write_json, sha256
from .scenarios import load_map
from .training import TrainingSession, atomic_save, fingerprint, new_replay, qualifies

PROTOCOL='clean_mapf_phase1_v1'


def clean_config(config):
    if config.reward != ProjectConfig(config.project_root).reward:
        raise ValueError('Phase 1 preserves the existing PANAMA rewards')
    return replace(config,communication=replace(config.communication,profile='perfect',stale_information=(),
        leo_timeline=None,uplink_trace=None,downlink_trace=None,handover_enabled=False,contention_enabled=False),
        resource_allocation=replace(config.resource_allocation,enabled=False,execution_mode='support'))


class CleanMAPFSession(TrainingSession):
    def __init__(self,config,dataset,seed,run_id,device):
        self.dataset=dataset;self.current_case=None;self.epoch_totals={}
        super().__init__(clean_config(config),LearningCondition.CLEAN,seed,run_id,device)

    def start_episode(self):
        if self.stage_index>=len(self.config.curriculum):return
        count=self.config.curriculum[self.stage_index].num_agents
        family=FAMILIES[self.stage_episodes%len(FAMILIES)]
        options=[m for m in self.dataset['maps'] if m['split']=='train' and m['family']==family]
        episode_seed=self.seed+10_000_000+self.stage_index*1_000_000+self.stage_episodes
        rng=np.random.default_rng(episode_seed)
        record=options[int(rng.integers(len(options)))];grid=load_map(record['path'])
        case=case_for(grid,record,count,episode_seed,coordinated=self.stage_episodes%2==0)
        self.current_case=case
        self.environment=CleanMAPFEnvironment(self.config,ExperimentMode.UAV,record['path'],count,250,
            episode_seed,scenario=scenario_for(case))
        self.observations,_=self.environment.reset()
        self.action_rng=np.random.default_rng(episode_seed)
        if not self.actor_models:
            self.actor_models[0]=self._actor(self.latest_published_weights)
        version=max(self.actor_models)
        self.deployed_versions={a:version for a in self.observations}
        self.episode_reward=0.;self.episode_collisions=0
        self.epoch_totals=dict(waiting_steps=0,vertex_conflicts=0,edge_conflicts=0,relevant_pairs=0,directed_pairs=0)

    def step(self):
        t=self.config.training
        epsilon=t.epsilon_end+(t.epsilon_start-t.epsilon_end)*math.exp(-self.generated_steps/t.epsilon_decay_steps)
        actions=select_actions(self.actor_models,self.deployed_versions,self.observations,epsilon,self.action_rng,self.device)
        measures=epoch_metrics(self.environment,actions)
        result=self.environment.step(actions);self.generated_steps+=1
        done=result.terminated or result.truncated
        for robot in self.observations:
            self.replay.add((self.observations[robot],actions[robot],result.rewards[robot],result.observations[robot],done),
                float(self.generated_steps))
            self.delivered_experiences+=1
        self.last_loss=_learn(self.policy,self.target,self.optimizer,self.replay,t.batch_size,t.gamma,t.tau,self.device)
        self.learning_opportunities+=1
        if self.last_loss is not None:self.learner_updates+=1
        if self.generated_steps%t.model_broadcast_steps==0:
            self.published_version+=1;self.model_publications+=1
            self.latest_published_weights={k:v.numpy() for k,v in cpu_state_dict(self.policy).items()}
            self.actor_models={self.published_version:self._actor(self.latest_published_weights)}
            self.deployed_versions={a:self.published_version for a in actions}
        self.observations=result.observations
        self.episode_reward+=sum(result.rewards.values())
        for key,value in measures.items():self.epoch_totals[key]+=value
        # No modeled transport is involved in this clean learner. Clear ordinary
        # environment bookkeeping rather than retaining irrelevant radio logs.
        self.environment.communication.transport.log.clear()
        row=dict(generated_joint_steps=self.generated_steps,learner_updates=self.learner_updates,
            stage_index=self.stage_index,num_agents=self.environment.num_agents,case_id=self.current_case['case_id'],
            family=self.current_case['family'],loss=self.last_loss,epsilon=epsilon,replay_size=len(self.replay),
            communication_delay_seconds=0.,**measures)
        episode=None
        if done:
            self.stage_episodes+=1;self.episodes+=1
            episode=dict(self.current_case,steps=self.environment.step_count,success=result.terminated,
                stage_index=self.stage_index,stage_episodes=self.stage_episodes,episodes=self.episodes,
                generated_joint_steps=self.generated_steps,reward=self.episode_reward,epsilon=epsilon,**self.epoch_totals)
        return row,episode


def wilson(successes,n):
    if not n:return [None,None]
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def summarize_cases(rows,expected=200):
    groups=defaultdict(list)
    for row in rows:groups[row['family']].append(row)
    def describe(items):
        n=len(items);s=sum(r['success'] for r in items)
        result=dict(episodes=n,successes=s,success_rate=s/n if n else None,success_ci95=wilson(s,n),
            mean_capped_makespan=float(np.mean([r['steps'] for r in items])) if n else None,
            mean_waiting_steps=float(np.mean([r['waiting_steps'] for r in items])) if n else None,
            vertex_conflicts=sum(r['vertex_conflicts'] for r in items),edge_conflicts=sum(r['edge_conflicts'] for r in items),
            relevant_pairs=sum(r['relevant_pairs'] for r in items),directed_pairs=sum(r['directed_pairs'] for r in items),
            exposed_cases=sum(r['relevant_pairs']>0 for r in items))
        maps=defaultdict(list)
        for item in items:maps[item['map_id']].append(item)
        result['map_clusters']=len(maps);result['map_cluster_ci95']={}
        if len(maps)>1:
            groups=list(maps.values());sizes=np.array([len(g) for g in groups])
            indices=np.random.default_rng(42).integers(0,len(groups),(2000,len(groups)))
            for metric in ('success','steps','waiting_steps'):
                sums=np.array([sum(float(r[metric]) for r in g) for g in groups])
                values=sums[indices].sum(1)/sizes[indices].sum(1)
                result['map_cluster_ci95'][metric]=np.quantile(values,[.025,.975]).tolist()
        return result
    result=describe(rows);result['per_family']={f:describe(groups[f]) for f in FAMILIES}
    result['complete']=len(rows)==expected
    result['exposure_passed']=all(r['episodes']==40 and r['exposed_cases']>=20 for r in result['per_family'].values()) and result['relevant_pairs']>=200
    result['passed']=result['complete'] and result['success_rate']>=.95 and result['exposure_passed'] and all(
        r['success_rate'] is not None and r['success_rate']>=.95 for r in result['per_family'].values())
    return result


def weights_hash(model):
    d=hashlib.sha256()
    for name,value in cpu_state_dict(model).items():d.update(name.encode());d.update(value.numpy().tobytes())
    return d.hexdigest()


def evaluate(config,model,bank,device,path,tag,deadline,stop=lambda:False,cap=250):
    identity=weights_hash(model)
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    existing=[r for r in existing if r['tag']==tag]
    if any(r['model_weights_sha256']!=identity for r in existing):raise ValueError('Evaluation model changed during resumable cohort')
    seen={r['case_id'] for r in existing}
    for case in bank:
        if case['case_id'] in seen:continue
        if time.monotonic()>=deadline or stop():break
        env=CleanMAPFEnvironment(config,ExperimentMode.UAV,case['map_path'],case['num_agents'],cap,
            case['seed'],scenario=scenario_for(case));obs,_=env.reset()
        totals=dict(waiting_steps=0,vertex_conflicts=0,edge_conflicts=0,relevant_pairs=0,directed_pairs=0)
        while time.monotonic()<deadline and not stop():
            actions=select_actions({0:model},{a:0 for a in obs},obs,0.,np.random.default_rng(0),device)
            for key,value in epoch_metrics(env,actions).items():totals[key]+=value
            result=env.step(actions);obs=result.observations
            env.communication.transport.log.clear()
            if result.terminated or result.truncated:
                row=dict(case,tag=tag,model_weights_sha256=identity,steps=env.step_count,success=result.terminated,
                    final_positions=[list(a.position) for a in env.agents],**totals)
                append_jsonl(path,row);existing.append(row);break
        else:break  # Incomplete case is never counted as a completed failure/success.
    return existing


def checkpoint_payload(session,dataset_sha,validation=None,test=None,smoke=False,compatibility=None):
    validation=validation or [];test=test or []
    qualified=(not smoke and session.stage_index==4 and len(validation)==len(test)==4 and
        all(r['passed'] for r in validation+test) and compatibility is not None and compatibility.get('status')=='passed')
    return dict(communication_schema=COMMUNICATION_SCHEMA,policy_family='panama_0.1_task',
        training_protocol=PROTOCOL,condition='clean',task_reward=session.config.to_dict()['reward'],
        model_state_dict=cpu_state_dict(session.policy),spatial_shape=(3,15,15),vector_length=session.config.vector_length,
        dataset_sha256=dataset_sha,run_id=session.run_id,seed=session.seed,learner_updates=session.learner_updates,
        generated_joint_steps=session.generated_steps,curriculum_completed=session.stage_index==4,
        qualification=validation,held_out=test,phase1_qualified=qualified,smoke=smoke,
        compatibility=compatibility,model_weights_sha256=weights_hash(session.policy),
        observation_contract='unchanged remote_observation with instantaneous fresh state',
        training_map_scope='disjoint procedural maps; no legacy single-map qualification claim')


def verify_payload(payload):
    if (payload.get('training_protocol')!=PROTOCOL or payload.get('smoke') or not payload.get('phase1_qualified')
        or not payload.get('curriculum_completed')):return False
    compatibility=payload.get('compatibility') or {}
    if compatibility.get('status')!='passed' or [r.get('num_agents') for r in compatibility.get('checks',[])]!=[2,4,6,8]:return False
    for key in ('qualification','held_out'):
        rows=payload.get(key,[])
        if [r.get('num_agents') for r in rows]!=[2,4,6,8]:return False
        if any(r.get('episodes')!=200 or not r.get('complete') or r.get('success_rate',0)<.95 or
            not r.get('exposure_passed') or not r.get('passed') or
            set(r.get('per_family',{}))!=set(FAMILIES) or
            any(v.get('episodes')!=40 or v.get('success_rate',0)<.95 or v.get('exposed_cases',0)<20
                for v in r.get('per_family',{}).values()) for r in rows):return False
    return True


def hardware(device):
    import psutil
    process=psutil.Process()
    return dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        process_rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2]),
        gpu_allocated_bytes=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)


def run(root,device_name='auto',smoke=False,resume=False):
    start=time.monotonic();root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if bool(smoke)!=manifest['smoke']:raise ValueError('Smoke/production resume identity mismatch')
    config=clean_config(ProjectConfig.from_dict(manifest['configuration']))
    dataset_path=root/'data/dataset.json';dataset=json.loads(dataset_path.read_text());verify_dataset(dataset)
    if sha256(dataset_path)!=manifest['dataset_sha256']:raise ValueError('Dataset identity changed')
    if {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}!=manifest['code_sha256']:
        raise ValueError('Frozen phase1 implementation changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=choose_device(device_name);checkpoint=root/'checkpoint.pt'
    state=dict(mode='train',elapsed_seconds=manifest['setup_seconds'],validation=[],held_out=[],last_validation_episode=0)
    if checkpoint.exists():
        if not resume:raise ValueError('Existing checkpoint requires --resume')
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        session=TrainingSession.restore(saved['session'],config,'clean',manifest['seed'],root.name,device)
        state=saved['state']
        if state.get('mode')=='complete':return json.loads((root/'status.json').read_text())
        for name,size in saved['log_offsets'].items():
            with (root/name).open('r+b') as stream:stream.truncate(size)
    else:
        session=CleanMAPFSession(config,dataset,manifest['seed'],root.name,device)
    previous_elapsed=state['elapsed_seconds'];budget=config.training.budget_seconds
    end=start+max(0,budget-previous_elapsed)
    train_end=end-(15 if smoke else 1800)-config.training.shutdown_reserve_seconds
    stop={'requested':False}
    old_handler=signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    def elapsed():return previous_elapsed+time.monotonic()-start
    def status(reason):
        data=dict(status=reason,qualified=False,mode=state['mode'],phase=session.stage_index,
            stage_episodes=session.stage_episodes,episodes=session.episodes,generated_joint_steps=session.generated_steps,
            learner_updates=session.learner_updates,wall_seconds=elapsed(),budget_seconds=budget,
            qualification=state['validation'],held_out=state['held_out'],hardware=hardware(device))
        data['curriculum_qualification']=session.qualification
        write_json(root/'status.json',data);return data
    def save():
        state['elapsed_seconds']=elapsed()
        session.elapsed_seconds=state['elapsed_seconds']
        offsets={p.name:p.stat().st_size for p in root.glob('*.jsonl')}
        atomic_save(dict(session=session.checkpoint(),state=state,log_offsets=offsets),checkpoint)
        atomic_save(checkpoint_payload(session,manifest['dataset_sha256'],state['validation'],state['held_out'],smoke,state.get('compatibility')),root/'latest_model.pt')
    status('training');save()
    def validate_stage():
        stage=config.curriculum[session.stage_index]
        tag=state.get('pending_validation') or f'curriculum-{session.stage_index}-{session.generated_steps}'
        state['pending_validation']=tag;save()
        cohort=[c for c in dataset['banks']['validation'] if c['num_agents']==stage.num_agents]
        values=evaluate(config,session.policy,cohort,device,root/'validation_cases.jsonl',tag,train_end,lambda:stop['requested'])
        result=summarize_cases(values);result.update(num_agents=stage.num_agents,
            stage_index=session.stage_index,generated_joint_steps=session.generated_steps,wall_seconds=elapsed())
        append_jsonl(root/'validation.jsonl',result)
        if result['complete']:state['pending_validation']=None
        if qualifies(stage,session.stage_episodes,result) and result['passed']:
            session.qualification.append(result)
            atomic_save(checkpoint_payload(session,manifest['dataset_sha256'],smoke=smoke),root/f'stage_{session.stage_index}_model.pt')
            session.stage_index+=1;session.stage_episodes=0
            session.replay=new_replay(config,session.seed+session.stage_index)
            if session.stage_index<4:session.start_episode()
        save()
    try:
        while state['mode']=='train' and time.monotonic()<train_end and not stop['requested']:
            if state.get('pending_validation'):
                validate_stage()
                if state.get('pending_validation'):break
                continue
            if session.stage_index==4 or session.generated_steps>=config.training.target_environment_steps:
                break
            if session.stage_episodes>=config.curriculum[session.stage_index].max_episodes:break
            if session.environment.all_agents_at_goals or session.environment.step_count>=250:session.start_episode()
            record,episode=session.step();record['wall_seconds']=elapsed()
            append_jsonl(root/'learning_steps.jsonl',record)
            if episode:
                episode['wall_seconds']=elapsed();append_jsonl(root/'training.jsonl',episode)
                interval=config.training.validation_interval_episodes
                stage=config.curriculum[session.stage_index]
                if session.stage_episodes>=stage.min_episodes and session.stage_episodes%interval==0:
                    validate_stage()
            if session.generated_steps%64==0:status('training')
            if session.generated_steps%config.training.checkpoint_interval_steps==0:save()
        if stop['requested']:
            status('interrupted_partial');return
        state['mode']='final';save();status('final_evaluation')
        from .phase1_compatibility import verify_compatibility
        if not state.get('compatibility') and time.monotonic()<end-config.training.shutdown_reserve_seconds:
            state['compatibility']=verify_compatibility(config,session.policy,dataset['banks']['validation'],device,root)
            save()
        # No learning or model selection after this point, including on resume.
        cohorts={(split,count):sorted((c for c in dataset['banks'][split] if c['num_agents']==count),
            key=lambda c:(c['evaluation_index'],FAMILIES.index(c['family'])))
            for split in ('validation','test') for count in (2,4,6,8)}
        # Interleave splits/populations/families so a budget-limited result does
        # not silently contain only the easiest validation populations.
        results={split:{} for split in ('validation','test')}
        for offset in range(0,max(map(len,cohorts.values())),10):
            for split,destination in (('validation','validation'),('test','held_out')):
                for count in (2,4,6,8):
                    values=evaluate(config,session.policy,cohorts[split,count][offset:offset+10],device,
                        root/f'final_{split}_cases.jsonl',f'final-{split}-n{count}',
                        end-config.training.shutdown_reserve_seconds,lambda:stop['requested'],cap=8 if smoke else 250)
                    result=summarize_cases(values);result['num_agents']=count;results[split][count]=result
                    state[destination]=[results[split][n] for n in sorted(results[split])]
                    status('final_evaluation')
                save()
            if time.monotonic()>=end-config.training.shutdown_reserve_seconds or stop['requested']:break
        complete=all(r['complete'] for r in state['validation']+state['held_out']) and len(state['held_out'])==4
        if smoke:complete=True  # Smoke is structurally complete but can never qualify.
        state['mode']='complete' if complete else 'final'
        payload=checkpoint_payload(session,manifest['dataset_sha256'],state['validation'],state['held_out'],smoke,state.get('compatibility'))
        atomic_save(payload,root/'final_model.pt');save()
        outcome=status('smoke_complete' if smoke else ('qualified' if payload['phase1_qualified'] else 'not_qualified' if complete else 'budget_exhausted_partial_evaluation'))
        outcome['qualified']=payload['phase1_qualified'];outcome['final_model_sha256']=sha256(root/'final_model.pt')
        write_json(root/'status.json',outcome)
        from .phase1_report import report
        report(root)
        return outcome
    except BaseException as error:
        status('error_partial');write_json(root/'error.json',dict(error=repr(error)))
        raise
    finally:
        save();signal.signal(signal.SIGINT,old_handler)
        path=root/'status.json'
        if path.exists():
            latest=json.loads(path.read_text());latest['wall_seconds']=elapsed()
            latest['budget_overrun_seconds']=max(0.,elapsed()-budget)
            write_json(path,latest)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='auto')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.root,a.device,a.smoke,a.resume)
