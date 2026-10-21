"""Verify paired weight-ratio controls against the nominal equal-weight policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from panama_uav.records import sha256, write_json


def verify(study, output):
    study=Path(study)
    manifest=json.loads((study/'manifest.json').read_text())
    assert not manifest['diagnostic']
    bank=json.loads((study/'held_out_cases.json').read_text())
    ids={r['case_id'] for r in bank}
    conditions={r['name']:r for r in manifest['conditions']}
    reference='up-1-cpu-1__equal'
    targets=[f'weight-ratio-1__{policy}' for policy in ('equal','static','aoi','priority')]
    targets+=['weight-ratio-16__equal']
    def read(name):
        rows=[json.loads(s) for s in (study/name/'episodes.jsonl').read_text().splitlines()]
        assert len(rows)==len(ids) and {r['case_id'] for r in rows}==ids
        return {r['case_id']:r for r in rows}
    base=read(reference)
    reports=[]
    for target in targets:
        ignored={'condition','solver_seconds','mean_relaxation_gap'}
        ignored.update(conditions[reference]['tags'])
        ignored.update(conditions[target]['tags'])
        actual=read(target)
        maxima={}
        for case in sorted(ids):
            a,b=base[case],actual[case]
            assert set(a)-ignored==set(b)-ignored
            for key in set(a)-ignored:
                value,other=a[key],b[key]
                if isinstance(value,(int,float)) and isinstance(other,(int,float)):
                    difference=abs(float(value)-float(other))
                    maxima[key]=max(maxima.get(key,0),difference)
                    assert difference<=1e-10*max(1,abs(value),abs(other)),(target,case,key,value,other)
                else:
                    assert value==other,(target,case,key)
        reports.append(dict(condition=target,episodes=len(ids),status='passed',maximum_differences=maxima))
    report=dict(status='passed',reference=reference,episodes_compared=sum(r['episodes'] for r in reports),
        interpretation='Ratio one removes every policy weight difference; the equal policy also ignores the upper ratio bound.',
        exclusions='Condition tags, relaxation bound and host-dependent solver wall time; physical execution must agree.',
        manifest_sha256=sha256(study/'manifest.json'),conditions=reports)
    write_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=verify(args.study,args.output)
    print(f"Weight identity controls agree across {result['episodes_compared']} paired episodes.")
