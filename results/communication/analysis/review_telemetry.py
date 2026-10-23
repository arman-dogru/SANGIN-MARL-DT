"""Read-only cross-check of saved case transactions and aggregate telemetry."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
manifest=json.loads((root/'manifest.json').read_text())
cases=json.loads((root/'paired_cases.json').read_text())
expected={(c['name'],p['case_id']) for c in manifest['conditions'] for p in cases}
manifest_sha=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest(records):
    h=hashlib.sha256()
    for record in records:h.update(encoded(record))
    return len(records),h.hexdigest()
transactions={};totals=Counter();counts=Counter()
for path in sorted((root/'episodes').glob('*/*.result.json')):
    packet=json.loads(path.read_text());row=packet['row'];tasks=packet['tasks'];rounds=packet['rounds']
    key=(row['condition'],row['case_id'])
    assert key in expected and key not in transactions
    assert packet['manifest_sha256']==manifest_sha and packet['audit']['status']=='passed'
    assert len(tasks)==row['tasks_generated']==row['deadline_tasks_monitored']
    assert len({t['job_id'] for t in tasks})==len(tasks)
    statuses=Counter(t['deadline_status'] for t in tasks)
    for state,suffix in (('on_time','on_time'),('late','late'),('undelivered','undelivered')):
        assert statuses[state]==row['deadline_commands_'+suffix]
        totals[state]+=statuses[state]
    assert sum(statuses.values())==len(tasks)
    assert sum(t.get('cutoff_induced_wait') is True for t in tasks)==row['command_cutoff_misses']
    assert sum(t['deadline_censored'] for t in tasks)==row['deadline_censored_tasks']
    assert row['timely_comparisons']+row['command_cutoff_misses']==row['active_decisions']
    assert row['makespan_seconds'] is not None if row['success'] else row['makespan_seconds'] is None
    assert row['waiting_completed_steps'] is not None if row['success'] else row['waiting_completed_steps'] is None
    assert all(-1e-9<=row[k]<=1+1e-9 for k in ('mean_drone_cpu_utilization','mean_cloud_cpu_utilization'))
    by_cycle=defaultdict(list)
    for task in tasks:
        assert task['condition']==row['condition'] and task['case_id']==row['case_id']
        assert task['required_cycles']==100000000 and task['input_bits']==100000
        x=task.get('executed_x')
        assert x in (None,0,1)
        if x is not None and row['route']!='allocated':assert x==(0 if row['route']=='cloud' else 1)
        if task['command_delivery_seconds'] is not None:
            assert abs(task['command_delivery_seconds']-task['inference_ready_seconds']-task['egress_seconds'])<1e-9
        by_cycle[task['cycle']].append(task)
    assert len(rounds)==row['rounds']==len(by_cycle)
    for cycle in rounds:
        group=by_cycle[cycle['cycle']]
        assert cycle['tasks']==len(group)
        assert cycle['on_time_commands']==sum(t['deadline_status']=='on_time' for t in group)
        assert cycle['delivered_commands']==sum(t['command_delivery_seconds'] is not None for t in group)
    transactions[key]={name:digest(value if isinstance(value,list) else [value])
                       for name,value in (('episodes',row),('tasks',tasks),('rounds',rounds))}
    totals.update(tasks=len(tasks),rounds=len(rounds),cutoff_waits=row['command_cutoff_misses'])
    counts[row['num_agents']]+=1
assert set(transactions)==expected
for name in ('episodes','tasks','rounds'):
    hashes=defaultdict(hashlib.sha256);sizes=Counter()
    with (root/(name+'.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            value=json.loads(line);key=(value['condition'],value['case_id'])
            assert key in expected
            hashes[key].update(encoded(value));sizes[key]+=1
    assert set(hashes)==expected
    assert all((sizes[k],hashes[k].hexdigest())==transactions[k][name] for k in expected)
result=dict(status='passed',case_transactions=len(transactions),populations=dict(counts),
    totals=dict(totals),aggregate_transactions_match=True,deadline_partition_verified=True,
    round_denominators_verified=True,assignments_and_cpu_bounds_verified=True,
    source_manifest_sha256=manifest_sha)
(root/'denominator_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
