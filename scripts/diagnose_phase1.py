"""Replay a frozen failed Phase 1 checkpoint and independently test feasibility."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from panama_uav.phase1_diagnosis import diagnose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='output/phase1_clean/phase1-clean-20261021-01')
    parser.add_argument('--output',required=True)
    parser.add_argument('--failures-per-cell',type=int,default=4)
    parser.add_argument('--successes-per-cell',type=int,default=2)
    parser.add_argument('--planner-cases',type=int,default=24)
    parser.add_argument('--planner-timeout',type=float,default=2.)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    summary = diagnose(PROJECT/args.source,PROJECT/args.output,
        failures_per_cell=args.failures_per_cell,successes_per_cell=args.successes_per_cell,
        planner_cases=args.planner_cases,planner_timeout=args.planner_timeout,device_name=args.device)
    print(json.dumps(summary,indent=2))
