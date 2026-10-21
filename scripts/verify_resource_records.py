"""Stream saved-case record counts; optionally restore interrupted prefix tails."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from panama_uav.records import write_json


def case_prefix(line):
    if not line.endswith("\n") or not line.rstrip().endswith("}"):
        raise ValueError("Incomplete JSONL record; preserve and inspect the interrupted tail")
    return json.loads(line.split(",",1)[0]+"}")["case_id"]


def verify(study, output, repair=False):
    study=Path(study)
    status=json.loads((study/"status.json").read_text())
    active=status.get("condition") if status["status"]=="running" else None
    reports=[]
    for episode_file in sorted(study.glob("*/episodes.jsonl")):
        directory=episode_file.parent
        if directory.name==active:
            continue  # Never edit an actively written condition.
        rows=[json.loads(s) for s in episode_file.read_text().splitlines()]
        assert len({r["case_id"] for r in rows})==len(rows)
        for filename,key,event in (("tasks.jsonl","tasks_generated",None),
                                   ("allocations.jsonl","allocation_solves","paper_allocation")):
            path=directory/filename
            counts=Counter()
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    for line in stream: counts[case_prefix(line)]+=1
            for row in rows:
                case=row["case_id"]; expected_count=row[key]
                if counts[case]==expected_count:
                    continue
                if not repair:
                    raise ValueError(f"Record count mismatch: {directory.name}/{filename}/{case}")
                with gzip.open(directory/f"{case}.json.gz","rt",encoding="utf-8") as stream:
                    trace=json.load(stream)
                source=trace["resource_tasks"] if event is None else [r for r in trace["transport_events"] if r["event"]==event]
                expected=[dict(case_id=case,**r) for r in source]
                actual=[]
                if path.exists():
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            if case_prefix(line)==case: actual.append(json.loads(line))
                if len(expected)!=expected_count or actual!=expected[:len(actual)]:
                    raise ValueError("Interrupted data are not a verified prefix; no repair applied")
                with path.open("a",encoding="utf-8") as stream:
                    for item in expected[len(actual):]:
                        stream.write(json.dumps(item,allow_nan=False,separators=(",",":"))+"\n")
                reports.append(dict(condition=directory.name,case_id=case,file=filename,
                                    restored_records=len(expected)-len(actual)))
            reports.append(dict(condition=directory.name,file=filename,
                committed_cases=len(rows),expected_records=sum(r[key] for r in rows),status="passed"))
    result=dict(status="passed",active_condition_skipped=active,records=reports)
    write_json(output,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True); p.add_argument("--output",required=True)
    p.add_argument("--repair",action="store_true",help="Restore only a verified missing suffix from a complete trace")
    a=p.parse_args(); verify(a.study,a.output,a.repair)
    print("Saved-case record counts verified.")
