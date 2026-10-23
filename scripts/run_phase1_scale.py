"""Prepare, calibrate, freeze and monitor the scale-16 episode-coverage protocol."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from panama_uav.phase1_scale import (benchmark,launch,load_protocol,prepare,prepare_data,reference_audit)
from panama_uav.records import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--config',default='configs/phase1_scale.toml')
    p.add_argument('--data-root',default='output/phase1_scale/data-16-01')
    p.add_argument('--prepare-data',action='store_true');p.add_argument('--reference-audit',action='store_true')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--benchmark-json')
    p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--monitor',action='store_true');p.add_argument('--monitor-registered',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):raise ValueError('Safe versioned run ID required')
    root=PROJECT/'output/phase1_scale'/args.run_id;config=PROJECT/args.config
    data=PROJECT/args.data_root;dataset=data/'dataset.json'
    if args.prepare_data:
        result=prepare_data(data,load_protocol(config),PROJECT)
        print(json.dumps(dict(dataset=str(dataset),maps=len(result['maps']),
            validation_cases=len(result['banks']['validation']),test_cases=len(result['banks']['test']))))
    if args.reference_audit:
        result=reference_audit(dataset,data/'reference_audit.json');print(json.dumps(result['counts']))
    if args.benchmark:
        result=benchmark(root,dataset,config,args.device);print(json.dumps(result['projection']))
    if args.prepare:
        if not args.benchmark_json:raise ValueError('Measured benchmark JSON required before preparing the full run')
        benchmark_path=PROJECT/args.benchmark_json
        measured=json.loads(benchmark_path.read_text());plan=measured['projection']
        manifest=prepare(root,config,episode_authorization=2500,
            wall_seconds=plan['proposed_budget_seconds'],evaluation_reserve_seconds=plan['final_evaluation_reserve_seconds'],
            dataset_path=dataset,benchmark_path=benchmark_path)
        import platform
        import torch
        manifest['software']=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
            numpy=__import__('numpy').__version__,cuda_runtime=torch.version.cuda,
            device=args.device,deterministic_algorithms=True)
        write_json(root/'manifest.json',manifest)
        print(json.dumps(dict(root=str(root),source_commit=manifest['source_commit'],wall_seconds=plan['proposed_budget_seconds'])))
    if args.launch:print(json.dumps(launch(root,args.device,args.resume,monitor_registered=args.monitor_registered)))
    if args.monitor:
        from panama_uav.research_monitor import inspect
        result=inspect(root,PROJECT/'plan.md')
        print(json.dumps({k:result[k] for k in ('decision','joint_steps','episodes','population_episodes','remaining_budget_seconds','next_check_local')}))
