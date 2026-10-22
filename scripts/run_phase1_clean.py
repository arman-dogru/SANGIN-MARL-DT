"""Prepare an immutable clean MAPF run and launch its isolated worker."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import platform

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from panama_uav.config import load_config
from panama_uav.phase1 import clean_config
from panama_uav.phase1_data import prepare_dataset
from panama_uav.records import sha256,write_json


def prepare(config,root,seed=42,smoke=False):
    start=time.monotonic();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    config=clean_config(config)
    if smoke:
        config=replace(config,training=replace(config.training,budget_seconds=120.,shutdown_reserve_seconds=5.,
            target_environment_steps=128,replay_capacity=512,batch_size=16,checkpoint_interval_steps=64))
    elif config.training.budget_seconds!=10800:
        raise ValueError('Reviewed Phase 1 budget is three hours; use an explicit revised protocol for other budgets')
    shutil.copytree(PROJECT/'src/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    dataset=prepare_dataset(root/'data',seed,train_maps=2 if smoke else 20,eval_maps=1 if smoke else 5,eval_cases=2 if smoke else 40)
    manifest=dict(protocol='clean_mapf_phase1_v1',seed=seed,smoke=smoke,run_id=root.name,
        created_utc=datetime.now(timezone.utc).isoformat(),configuration=config.to_dict(),
        code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        dataset_sha256=sha256(root/'data/dataset.json'),setup_seconds=time.monotonic()-start,
        initialization='new deterministic random weights; no pretrained checkpoint',
        budget_includes='dataset preparation, initialization, training, validation, final qualification/test and checkpoints',
        final_evaluation_reserve_seconds=15 if smoke else 1800,
        qualification='200 cases per population; >=95% overall and in each 40-case family; exposure; final held-out gate',
        physical_kernels='unchanged',reserved_stress_maps=dataset['reserved_stress_maps'])
    import torch
    manifest['software']=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        numpy=__import__('numpy').__version__,deterministic_algorithms=True,cublas_workspace_config=':4096:8')
    write_json(root/'manifest.json',manifest)
    return root


def launch(root,device,smoke=False,resume=False):
    import psutil
    identity_path=root/'launch.json'
    if identity_path.exists():
        identity=json.loads(identity_path.read_text())
        try:
            previous=psutil.Process(identity['pid'])
            if abs(previous.create_time()-identity['creation_time'])<1 and previous.is_running():
                raise ValueError('This training run already has a live worker; refusing duplicate launch')
        except psutil.NoSuchProcess:pass
    env=os.environ.copy();env['PYTHONPATH']=str(root/'code');env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    command=[sys.executable,'-m','panama_uav.phase1','--root',str(root),'--device',device]
    if smoke:command.append('--smoke')
    if resume:command.append('--resume')
    with (root/'worker.stdout.log').open('a') as out,(root/'worker.stderr.log').open('a') as err:
        flags=(subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name=='nt' else 0
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=out,stderr=err,creationflags=flags,
            start_new_session=os.name!='nt')
    write_json(root/'launch.json',dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),
        command=command,device=device,smoke=smoke,resume=resume))
    return process.pid


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--config',default='configs/phase1_clean.toml');parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--launch',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):raise ValueError('Use a fresh safe run ID')
    root=PROJECT/'output/phase1_clean'/args.run_id
    if not args.resume:prepare(load_config(PROJECT/args.config),root,args.seed,args.smoke)
    if args.launch:print(json.dumps(dict(root=str(root),pid=launch(root,args.device,args.smoke,args.resume))))
    else:print(root)
