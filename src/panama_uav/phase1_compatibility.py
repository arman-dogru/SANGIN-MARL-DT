"""Small zero-delay remote-DT inference/replay check, never a network study."""
from dataclasses import replace
import numpy as np

from .config import load_config, ExperimentMode
from .evaluation import run_episode
from .models import select_action
from .phase1_data import scenario_for
from .phase1_environment import CleanMAPFEnvironment
from .remote_audit import replay_trace
from .records import write_json


def verify_compatibility(config,model,bank,device,root):
    remote=load_config(config.project_root/'configs/paper_remote_dt_250ms.toml')
    remote=replace(remote,communication=replace(remote.communication,profile='perfect',leo_timeline=None))
    def infer(location,robot,obs):return select_action(model,obs,0.,np.random.default_rng(0),device)
    checks=[]
    for count in (2,4,6,8):
        case=next(c for c in bank if c['num_agents']==count)
        env=CleanMAPFEnvironment(config,ExperimentMode.UAV,case['map_path'],count,250,case['seed'],scenario_for(case))
        obs,_=env.reset()
        _,trace=run_episode(remote,model,case,device,scenario=scenario_for(case),max_steps=3)
        initial=[e for e in trace['remote_events'] if e['event']=='remote_inference' and e['job_id'].startswith('0:')]
        if len(initial)!=count:raise ValueError('Missing zero-delay initial inference')
        for event in initial:
            for name in ('spatial','vector'):
                if not np.array_equal(obs[event['robot_id']][name],event['observation'][name]):
                    raise ValueError('Clean / remote observation differs')
            if event['action']!=infer(event['location'],event['robot_id'],obs[event['robot_id']]):
                raise ValueError('Clean / remote greedy action differs')
        audit=replay_trace(trace,infer)
        checks.append(dict(num_agents=count,initial_observations_equal=count,
            remote_actions_verified=audit['model_actions_verified'],replay_status=audit['status']))
    result=dict(status='passed',checks=checks,
        scope='active initial observation/action equivalence and 3-step ideal remote replay; post-completion controller remains distinct')
    write_json(root/'compatibility.json',result)
    return result
