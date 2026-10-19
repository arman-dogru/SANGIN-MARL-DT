"""Derive physical transport statistics from recorded events, leaving raw data intact.

The original runner counted any positive floating-point subtraction as a
handover, including ~1e-14 s access-link roundoff. This audit restricts holds
to satellite hops and tolerates 1 ns numerical noise. Orbital windows come
from actual orbital events, not the periodic-profile helper.
"""
import csv
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-suites",nargs="+",choices=["planned","orbital"])
    args = parser.parse_args()
    root = Path("output/inference/baseline-20261006-01")
    status = json.loads((root/"status.json").read_text())
    if args.completed_suites:
        assert set(args.completed_suites) <= set(status["completed_suites"])
    else:
        assert status["status"] == "complete"
    rows = []
    for suite in args.completed_suites or ("planned", "orbital"):
        for folder in sorted((root/suite).glob("physical_*")):
            for path in sorted(folder.glob("agents-*-case-*.json.gz")):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    trace = json.load(f)
                tx = [e for e in trace["transport_events"] if e["event"] == "transmit"]
                satellite = [e for e in tx if e["link"].startswith("satellite")]
                holds = [max(0.,e.get("handover_seconds",0.)) for e in satellite]
                windows = []
                for e in trace["transport_events"]:
                    if e["event"] == "orbit_handover":
                        offset = 1000.
                        windows.append([max(0.,e["source_window"]["start"]-offset),e["source_window"]["end"]-offset])
                    elif e["event"] == "handover_start":
                        windows.append([e["time"],e["time"]+.05])
                rows.append({"suite":suite,"condition":folder.name, "case_id":trace["case"]["case_id"],
                    "num_agents":trace["case"]["num_agents"],
                    "satellite_transmissions":len(satellite),
                    "transmissions_interrupted_by_handover":sum(h>1e-9 for h in holds),
                    "handover_wait_seconds":sum(h for h in holds if h>1e-9),
                    "handover_windows_elapsed":len(windows),"actual_handover_windows":json.dumps(windows),
                    "maximum_delivery_age_seconds":max(e["age_seconds"] for e in trace["transport_events"] if e["event"] == "delivered")})
            print(f"Audited {suite}/{folder.name}",flush=True)
    path = root/"transport_event_audit.csv"
    if args.completed_suites and path.exists():
        with path.open(newline="",encoding="utf-8") as f:
            rows.extend(r for r in csv.DictReader(f) if r["suite"] not in args.completed_suites)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    assert all(float(r["handover_wait_seconds"])==0 and int(r["satellite_transmissions"])==0 for r in rows if "_uav_" in r["condition"])
    print(json.dumps({"event_audit_rows":len(rows),"path":str(path.resolve()),"uav_local_handover_hold_is_zero":True}))


if __name__ == "__main__":
    main()
