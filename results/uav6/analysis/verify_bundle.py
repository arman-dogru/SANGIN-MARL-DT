"""Verify the published parameter-only pilot without external input assets."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET


def digest(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    root=Path(root).resolve()
    with (root.parent/'files.csv').open(encoding='utf-8',newline='') as stream:
        entries=[r for r in csv.DictReader(stream) if r['study']=='uav6']
    assert len(entries)==32
    for entry in entries:
        path=root.parent/entry['file']
        with path.open('rb') as stream: assert digest(stream)==entry['sha256'],path
        if entry['format'].startswith('gzip'):
            with gzip.open(path,'rb') as stream: assert digest(stream)==entry['source_sha256'],path
    old=tomllib.loads((root/'base.toml').read_text())
    proposed=tomllib.loads((root/'pilot.toml').read_text())
    assert old['resource_allocation']['drone_cycles_per_second']==1e9
    assert proposed['resource_allocation']['drone_cycles_per_second']==6e9
    proposed['resource_allocation']['drone_cycles_per_second']=1e9
    assert old==proposed
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['populations']==[2,4,8] and manifest['requested_populations']==[2,4,8,16]
    assert [r['num_agents'] for r in manifest['excluded_populations']]==[16]
    assert manifest['policy_updates']==0
    with (root/'cases.json').open('rb') as stream: assert digest(stream)==manifest['cases_sha256']
    cases={r['case_id']:r for r in json.loads((root/'cases.json').read_text())}
    expected={(c['name'],case) for c in manifest['conditions'] for case in cases}
    observed=set();episode_counts=Counter();episode_statuses=Counter()
    with (root/'episodes.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['condition'],row['case_id'])
            assert key in expected and key not in observed
            observed.add(key)
            case=cases[row['case_id']]
            for field in ('num_agents','seed'): assert int(row[field])==case[field]
            for field in ('starts','goals'): assert json.loads(row[field])==case[field]
            assert float(row['orbital_offset_seconds'])==case['orbital_offset_seconds']
            assert row['map_sha256']==case['map_sha256']
            episode_counts[key]=int(row['tasks_generated'])
            for status in ('on_time','late','undelivered'):
                episode_statuses[status]+=int(row['deadline_commands_'+status])
    assert observed==expected and len(observed)==60
    task_counts=Counter();task_statuses=Counter();unique=set();waits=0
    with gzip.open(root/'tasks.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            identity=(row['condition'],row['case_id'],row['job_id'])
            assert identity not in unique
            unique.add(identity)
            key=identity[:2]
            assert key in expected
            task_counts[key]+=1
            task_statuses[row['deadline_status']]+=1
            waits+=row['cutoff_induced_wait']=='True'
            assert row['input_bits']=='100000' and float(row['required_cycles'])==1e8
            if row['executed_x']: assert float(row['executed_x']) in (0.,1.)
    assert task_counts==episode_counts and task_statuses==episode_statuses
    assert len(unique)==8746 and task_statuses==Counter(on_time=8659,late=87,undelivered=0)
    assert waits==87
    with (root/'rounds.csv').open(encoding='utf-8',newline='') as stream:
        rounds=list(csv.DictReader(stream))
    assert len(rounds)==2883 and sum(int(r['tasks']) for r in rounds)==8746
    with gzip.open(root/'audits.jsonl.gz','rt',encoding='utf-8') as stream:
        audits=[json.loads(line) for line in stream if line.strip()]
    assert len(audits)==60 and all(r['status']=='passed' for r in audits)
    suite=ET.parse(root/'tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==374
    assert int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    return dict(status='passed',label='Parameter-only pilot results',source_artifacts=len(entries),
        published_hashes_verified=True,gzip_original_bytes_verified=True,only_uav_capacity_changed=True,
        episodes=len(observed),tasks=len(unique),rounds=len(rounds),command_partitions={s:task_statuses[s] for s in ('on_time','late','undelivered')},
        cutoff_waits=waits,replay_audits=len(audits),full_suite_tests=374,
        unsupported_population=16,qualified_scope_enforced=True)


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
