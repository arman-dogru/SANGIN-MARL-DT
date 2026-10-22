"""Site-conditional paired inference from already audited task/episode CSVs."""
import csv
from collections import defaultdict
from pathlib import Path
import json
import subprocess

import numpy as np

from .priority_evaluation import BASELINE, PRIMARY, paired_interval
from .records import sha256, write_json


def report_locations(output, samples=2000):
    output = Path(output)
    validation = json.loads((output/"validation.json").read_text())
    if validation["status"] != "passed":
        raise ValueError("Location inference requires passed replay validation")
    with (output/"episodes.csv").open(newline="",encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    if len(episodes) != validation["episodes"]:
        raise ValueError("Audited episode count differs from location inputs")
    keys = {(r["condition"],r["case_id"]):r for r in episodes}
    if len(keys) != len(episodes):
        raise ValueError("Duplicate audited episodes")
    tasks = defaultdict(lambda: dict(tasks=0,on_time=0,late=0,undelivered=0,assessed=0,
        cutoff_waits=0,latencies=[],cpu_elapsed=[],cpu_wait=[]))
    seen = set()
    with (output/"tasks_by_priority.csv").open(newline="",encoding="utf-8") as stream:
        for task in csv.DictReader(stream):
            identity = (task["condition"],task["case_id"],task["job_id"])
            if identity in seen or identity[:2] not in keys:
                raise ValueError("Duplicate or unpaired location task")
            seen.add(identity)
            group = tasks[(*identity[:2],task["location"])]
            group["tasks"] += 1
            group[task["deadline_status"]] += 1
            group["assessed"] += task["deadline_met"] != ""
            group["cutoff_waits"] += task["cutoff_wait"] == "True"
            for source,destination in (("latency_seconds","latencies"),("cpu_elapsed_seconds","cpu_elapsed"),
                                       ("cpu_zero_rate_wait_seconds","cpu_wait")):
                if task[source] != "":
                    group[destination].append(float(task[source]))
    rows = []
    for episode in episodes:
        observed = sum(tasks[(episode["condition"],episode["case_id"],site)]["tasks"] for site in ("drone","cloud","unassigned"))
        if observed != int(episode["tasks_generated"]):
            raise ValueError("Location task denominator differs from audited episode")
        for site in ("drone","cloud","unassigned"):
            group = tasks[(episode["condition"],episode["case_id"],site)]
            row = dict(condition=episode["condition"],case_id=episode["case_id"],num_agents=int(episode["num_agents"]),
                location=site,**{k:v for k,v in group.items() if not isinstance(v,list)})
            row.update(mean_latency=float(np.mean(group["latencies"])) if group["latencies"] else None,
                latency_p95=float(np.quantile(group["latencies"],.95)) if group["latencies"] else None,
                mean_cpu_elapsed=float(np.mean(group["cpu_elapsed"])) if group["cpu_elapsed"] else None,
                mean_cpu_zero_rate_wait=float(np.mean(group["cpu_wait"])) if group["cpu_wait"] else None,
                pool_utilization=float(episode[f"mean_{site}_cpu_utilization"]) if site != "unassigned" else None)
            rows.append(row)
    differences = []
    for count in sorted({r["num_agents"] for r in rows}):
        for site in ("drone","cloud","unassigned"):
            reference = [r for r in rows if r["condition"] == BASELINE and r["num_agents"] == count and r["location"] == site]
            for condition in dict.fromkeys(r["condition"] for r in episodes):
                actual = [r for r in rows if r["condition"] == condition and r["num_agents"] == count and r["location"] == site]
                for metric,ratio in (("on_time_rate",("on_time","tasks")),("deadline_compliance",("on_time","assessed")),
                    ("mean_latency",None),("latency_p95",None),("mean_cpu_elapsed",None),
                    ("mean_cpu_zero_rate_wait",None),("pool_utilization",None)):
                    differences.append(dict(condition=condition,reference=BASELINE,num_agents=count,location=site,
                        primary_comparison=condition == PRIMARY,
                        **paired_interval(actual,reference,metric,ratio=ratio,samples=samples)))
    for filename,records in (("execution_location_case_metrics.csv",rows),("paired_execution_location_differences.csv",differences)):
        with (output/filename).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
            writer.writeheader()
            writer.writerows(records)
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit = None
    write_json(output/"location_methodology.json",dict(analysis_commit=commit,source_sha256=sha256(__file__),
        input_tasks_sha256=sha256(output/"tasks_by_priority.csv"),input_episodes_sha256=sha256(output/"episodes.csv"),
        bootstrap_samples=samples,bootstrap_unit="paired case",bootstrap_seed=42,
        note="Conditional site cohorts change with routing; site rates are not causal same-task service effects"))
    return differences
