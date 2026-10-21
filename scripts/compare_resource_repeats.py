"""Compare independent executions of identical paired resource conditions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from panama_uav.records import write_json


def compare(left,right,output):
    left,right=Path(left),Path(right)
    lm=json.loads((left/"manifest.json").read_text()); rm=json.loads((right/"manifest.json").read_text())
    for key in ("map_sha256","timeline_sha256","case_bank_sha256"):
        assert lm[key]==rm[key]
    assert lm["model"]["sha256"]==rm["model"]["sha256"]
    common={c["name"] for c in lm["conditions"]}&{c["name"] for c in rm["conditions"]}
    differences={}; episodes=0; relaxed_max=0.0
    for name in sorted(common):
        def read(root):
            return {r["case_id"]:r for r in (json.loads(s) for s in (root/name/"episodes.jsonl").read_text().splitlines())}
        a,b=read(left),read(right)
        assert set(a)==set(b) and len(a)==len(json.loads((left/"held_out_cases.json").read_text()))
        for case in a:
            assert set(a[case])==set(b[case])
            for key,value in a[case].items():
                other=b[case][key]
                if key=="solver_seconds":
                    continue  # Host contention changes wall time, never simulated service.
                if isinstance(value,(int,float)) and isinstance(other,(int,float)):
                    delta=abs(float(value)-float(other))
                    if key=="mean_relaxation_gap":
                        relaxed_max=max(relaxed_max,delta); assert delta<=1e-6
                    else:
                        differences[key]=max(differences.get(key,0),delta)
                        assert delta<=1e-10*max(1,abs(value),abs(other)),(name,case,key,value,other)
                else:
                    assert value==other,(name,case,key)
            episodes+=1
    report=dict(status="passed",conditions=len(common),episodes_compared=episodes,
        physical_metric_maximum_differences=differences,maximum_relaxation_gap_difference=relaxed_max,
        excluded="solver wall time; it does not advance the virtual clock",
        checkpoint_sha256=lm["model"]["sha256"],timeline_sha256=lm["timeline_sha256"])
    write_json(output,report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left",required=True); p.add_argument("--right",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); r=compare(a.left,a.right,a.output)
    print(f"Independent repeats agree across {r['episodes_compared']} episodes.")
