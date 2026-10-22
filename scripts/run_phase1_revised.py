"""Prepare/freeze V2 source and disjoint data, then launch a bounded clean worker."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from panama_uav.config import load_config
from panama_uav.phase1_revised import PROTOCOL, load_protocol, prepare_data, read_json
from panama_uav.records import sha256, write_json


def prepare(root, config_path):
    start = time.monotonic()
    root = Path(root).resolve()
    spec = load_protocol(config_path)
    historical = PROJECT/spec['historical_dataset']
    if not historical.exists():
        raise ValueError('Historical exclusion bank required; do not substitute another test split')
    root.mkdir(parents=True, exist_ok=False)
    dataset = prepare_data(root/'data', spec, historical)
    shutil.copytree(PROJECT/'src/panama_uav', root/'code/panama_uav',
                    ignore=shutil.ignore_patterns('__pycache__'))
    base = load_config(PROJECT/spec['base_config'])
    manifest = dict(protocol=PROTOCOL, spec=spec, base_configuration=base.to_dict(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip()),
        code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
        dataset_path=str(root/'data/dataset.json'), dataset_sha256=sha256(root/'data/dataset.json'),
        config_sha256=sha256(config_path), historical_dataset_sha256=sha256(historical),
        shared_preparation_seconds=time.monotonic()-start,
        created_utc=datetime.now(timezone.utc).isoformat(),
        map_counts={s:sum(m['split']==s for m in dataset['maps']) for s in ('train','validation','test')},
        test_use='No test evaluation during ablations. New untouched test only after final full-run freeze.',
        budget='Each run includes initialization, learning, validation/checkpoints and final evaluation; shared data/source preparation is reported separately.')
    write_json(root/'study.json',manifest)
    return manifest


def launch(root, mode, device, resume=False):
    import psutil
    root = Path(root).resolve()
    if mode == 'full' and not read_json(root/'ablation_comparison.json')['full_run_justified']:
        raise ValueError('Paired validation does not justify full training')
    # Match the Windows venv launcher OR its Python child; never trust PID alone.
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            cmd = process.info['cmdline'] or []
            if 'panama_uav.phase1_revised' in cmd and str(root) in cmd:
                raise ValueError(f"Existing revised worker {process.pid}; refusing duplicate launch")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'code')
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    command = [sys.executable,'-m','panama_uav.phase1_revised','--study',str(root),'--mode',mode,'--device',device]
    if resume:
        command.append('--resume')
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name=='nt' else 0
    with (root/f'{mode}.stdout.log').open('a') as out, (root/f'{mode}.stderr.log').open('a') as err:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=out, stderr=err,
            creationflags=flags, start_new_session=os.name!='nt')
    info = dict(pid=process.pid, creation_time=psutil.Process(process.pid).create_time(),
        command=command, device=device, mode=mode, resume=resume,
        launched_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/f'launch-{mode}.json',info)
    return info


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-id',required=True)
    parser.add_argument('--config',default='configs/phase1_revised.toml')
    parser.add_argument('--mode',choices=['ablations','full'],default='ablations')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.study_id):
        raise ValueError('Use a fresh safe study ID')
    root=PROJECT/'output/phase1_revised'/args.study_id
    if args.prepare:
        if args.resume or args.mode!='ablations':
            raise ValueError('Prepare a new study before any training; do not overwrite previous runs')
        prepare(root,PROJECT/args.config)
    if args.launch:
        print(json.dumps(launch(root,args.mode,args.device,args.resume)))
    else:
        print(root)
