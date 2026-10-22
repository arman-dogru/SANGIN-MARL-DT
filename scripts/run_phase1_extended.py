"""Primary evidence -> explicit budget plan -> new bounded clean full run."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from panama_uav.phase1_extended import analyse_comparison,benchmark_and_plan,prepare_extended,launch,diagnose_models

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--comparison',default='output/phase1_revised/phase1-comparison-20261022-01')
    parser.add_argument('--run-id')
    parser.add_argument('--analyse',action='store_true');parser.add_argument('--estimate',action='store_true')
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    study=(PROJECT/args.comparison).resolve()
    if args.analyse:
        result=analyse_comparison(study)
        print(json.dumps(dict(selected=result['selection']['selected'],full_run_justified=result['selection']['full_run_justified'])))
    if args.estimate:
        result=benchmark_and_plan(study,args.device)
        print(json.dumps(result['projection']))
    if args.diagnose:
        result=diagnose_models(study,args.device)
        print(json.dumps(dict(diagnostic_replays=len(result),training_updates=0)))
    if args.prepare or args.launch:
        if not args.run_id or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):
            raise ValueError('Use a fresh versioned run ID')
        root=PROJECT/'output/phase1_extended'/args.run_id
        if args.prepare:
            if args.resume:raise ValueError('Never prepare over an existing run')
            prepare_extended(root,study)
        if args.launch:print(json.dumps(launch(root,args.device,args.resume)))
