"""Verify published extension data with only the Python standard library."""
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import tomllib


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def table(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def verify(root):
    root=Path(root).resolve();original=root.parent
    entries=[r for r in table(original.parent/'files.csv') if r['study']=='uav6-extension']
    assert entries and len({r['file'] for r in entries})==len(entries)
    for entry in entries:
        path=original.parent/entry['file']
        assert digest(path)==entry['sha256'],path
        if entry['format'].startswith('gzip'):
            with gzip.open(path,'rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['source_sha256'],path
        else:
            assert entry['sha256']==entry['source_sha256']
    old=tomllib.loads((root/'base.toml').read_text())
    proposed=tomllib.loads((root/'pilot.toml').read_text())
    assert old['resource_allocation']['drone_cycles_per_second']==1e9
    assert proposed['resource_allocation']['drone_cycles_per_second']==6e9
    proposed['resource_allocation']['drone_cycles_per_second']=1e9
    assert old==proposed
    manifest=json.loads((root/'manifest.json').read_text())
    base_manifest=json.loads((original/'manifest.json').read_text())
    assert manifest['base_manifest_sha256']==digest(original/'manifest.json')
    assert manifest['model_sha256']==base_manifest['model_sha256']
    assert manifest['timeline_sha256']==base_manifest['timeline_sha256']
    assert manifest['map_sha256']==base_manifest['map_sha256']
    assert manifest['populations']==[16,32] and manifest['cases_per_population']==5
    assert manifest['episode_cap']==250 and manifest['expected_episodes']==40 and manifest['policy_updates']==0
    assert manifest['harness_sha256']==digest(root/'analysis/extend.py')
    assert manifest['base_harness_sha256']==digest(root/'analysis/base_pilot.py')
    assert manifest['cases_sha256']==digest(root/'cases.json')
    cases={c['case_id']:c for c in json.loads((root/'cases.json').read_text())}
    expected={(c['name'],case) for c in manifest['conditions'] for case in cases}
    episodes=table(root/'episodes.csv')
    observed=set();episode_tasks=Counter();episode_statuses=Counter();episode_waits=0
    for row in episodes:
        key=row['condition'],row['case_id'];assert key in expected and key not in observed
        observed.add(key);case=cases[row['case_id']]
        for field in ('num_agents','seed'):assert int(row[field])==case[field]
        for field in ('starts','goals'):assert json.loads(row[field])==case[field]
        assert float(row['orbital_offset_seconds'])==case['orbital_offset_seconds']
        assert row['map_sha256']==case['map_sha256']
        episode_tasks[key]=int(row['tasks_generated'])
        episode_waits+=int(row['command_cutoff_misses'])
        for status in ('on_time','late','undelivered'):
            episode_statuses[status]+=int(row['deadline_commands_'+status])
    assert observed==expected and len(observed)==40
    counts=Counter();states=Counter();unique=set();cycle_counts=Counter();waits=0
    groups=defaultdict(lambda:dict(latency=[],placement=Counter(),states=Counter()))
    population={case:cases[case]['num_agents'] for case in cases}
    with gzip.open(root/'tasks.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=row['condition'],row['case_id'];identity=(*key,row['job_id'])
            assert key in expected and identity not in unique
            unique.add(identity);counts[key]+=1;states[row['deadline_status']]+=1
            cycle_counts[(*key,int(row['cycle']))]+=1
            waits+=row['cutoff_induced_wait']=='True'
            assert float(row['input_bits'])==100000 and float(row['required_cycles'])==1e8
            assert float(row['executed_x']) in (0.,1.)
            group=groups[row['condition'],population[row['case_id']]]
            group['placement'][int(float(row['executed_x']))]+=1
            group['states'][row['deadline_status']]+=1
            if row['command_delivery_seconds']:
                latency=float(row['command_delivery_seconds']);group['latency'].append(latency)
                assert (latency<=.25+1e-9) if row['deadline_status']=='on_time' else latency>.25-1e-9
            else:
                assert row['deadline_status']=='undelivered'
    audit=json.loads((root/'accounting_audit.json').read_text())
    assert counts==episode_tasks and states==episode_statuses and waits==episode_waits
    assert len(unique)==audit['new_aggregate_counts']['tasks']==64342
    assert dict(states)=={k:v for k,v in audit['new_command_partition'].items() if v}
    assert waits==audit['new_cutoff_waits']==43550
    rounds=table(root/'rounds.csv')
    round_tasks={}
    for row in rounds:
        key=row['condition'],row['case_id'],int(row['cycle'])
        assert key not in round_tasks
        round_tasks[key]=int(row['tasks'])
    assert dict(cycle_counts)==round_tasks and len(rounds)==4050
    summary=json.loads((root/'summary.json').read_text())
    for row in summary:
        group=groups[row['condition'],row['num_agents']]
        assert row['uav_tasks']==group['placement'][1] and row['cloud_tasks']==group['placement'][0]
        total=sum(group['placement'].values());assert row['tasks']==total
        assert math.isclose(row['cloud_fraction'],row['cloud_tasks']/total,rel_tol=1e-12)
        for state in ('on_time','late','undelivered'):assert row[state]==group['states'][state]
        assert math.isclose(row['on_time_fraction'],row['on_time']/total,rel_tol=1e-12,abs_tol=1e-12)
        values=sorted(group['latency'])
        assert math.isclose(row['observed_mean_s'],math.fsum(values)/len(values),rel_tol=1e-10)
        for field,q in (('observed_p50_s',.5),('observed_p95_s',.95)):
            position=(len(values)-1)*q;low=math.floor(position);high=math.ceil(position)
            quantile=values[low]+(position-low)*(values[high]-values[low])
            assert math.isclose(row[field],quantile,rel_tol=1e-10)
    combined=json.loads((root/'combined_summary.json').read_text())
    assert len(combined)==20 and combined==json.loads((original/'summary.json').read_text())+summary
    combined_rows=table(root/'combined_episodes.csv');base_rows=table(original/'episodes.csv')
    canon=lambda row:{k:v for k,v in row.items() if v!=''}
    index=lambda rows:{(r['condition'],r['case_id']):canon(r) for r in rows}
    assert len(combined_rows)==100 and index(combined_rows)==index(base_rows+episodes)
    assert set(int(r['num_agents']) for r in combined_rows)=={2,4,8,16,32}
    assert all(v==5 for v in Counter((r['condition'],r['num_agents']) for r in combined_rows).values())
    with gzip.open(root/'audits.jsonl.gz','rt',encoding='utf-8') as stream:
        replays=[json.loads(line) for line in stream if line.strip()]
    assert len(replays)==40 and all(r['status']=='passed' for r in replays)
    assert {(r['condition'],r['case_id']) for r in replays}==expected
    reference=json.loads((root/'mapf_reference_serial.json').read_text())
    assert reference['model_sha256']==manifest['model_sha256']
    assert reference['case_bank_sha256']==manifest['cases_sha256']
    assert len(reference['records'])==10 and reference['inference_comparisons']==6682 and reference['batch_serial_disagreements']==0
    assert {r['case_id'] for r in reference['records']}==set(cases)
    assert digest(root/'analysis/finalize.py')==audit['postprocessor_sha256']
    result=dict(status='passed',label='Parameter-only pilot results',source_artifacts=len(entries),
        primary_episodes=100,added_episodes=40,added_tasks=len(unique),added_rounds=len(rounds),
        added_replay_audits=len(replays),paired_cases_verified=True,deadline_partition_verified=True,
        combined_summaries_reproduced=True,latency_quantiles_recomputed=True,only_uav_capacity_changed=True,
        command_partition={s:states[s] for s in ('on_time','late','undelivered')},cutoff_waits=waits,
        reference_action_comparisons=6682,reference_action_disagreements=0,policy_updates=0)
    return result


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
