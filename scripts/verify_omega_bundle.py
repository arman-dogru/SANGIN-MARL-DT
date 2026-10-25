"""Portable verification of sensitivity tables, pairing, file hashes and ZIPs."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def verify(root):
    root=Path(root)
    for item in csv.DictReader((root/'files.csv').open(encoding='utf-8')):
        path=root/item['path']
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert path.stat().st_size==int(item['bytes']) and digest==item['sha256'],str(path)
    m=json.loads((root/'manifest.json').read_text());cases=json.loads((root/'cases.json').read_text())
    bank={c['case_id']:c for c in cases}
    expected={(w['id'],r['route'],c['case_id']) for w in m['weights'] for r in m['conditions'] for c in cases}
    episodes=list(csv.DictReader((root/'episodes.csv').open(encoding='utf-8')))
    actual=set();groups={}
    for row in episodes:
        identity=(row['weight_id'],row['route'],row['case_id']);assert identity not in actual;actual.add(identity)
        case=bank[row['case_id']]
        for key in ('seed','num_agents','orbital_offset_seconds'):assert float(row[key])==float(case[key])
        for key in ('starts','goals'):assert json.loads(row[key])==case[key]
        assert row['map_sha256']==case['map_sha256']
        groups[identity]=row
    publication=json.loads((root/'publication.json').read_text()) if (root/'publication.json').exists() else None
    if publication:
        assert publication['cohort']=='completed_missions'
        assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==publication['source_manifest_sha256']
        for name,field in (('publish_omega_completed.py','publication_source_sha256'),
                           ('report_omega_sensitivity.py','figure_source_sha256')):
            directory=publication.get('postprocessing_directory','analysis/postprocessing')
            assert hashlib.sha256((root/directory/name).read_bytes()).hexdigest()==publication[field]
        assert actual<=expected and actual=={tuple(i) for i in publication['included_identities']}
        assert len(episodes)==publication['included_episodes']
        assert publication['source_episodes']==m['expected_episodes']
        assert publication['excluded_episodes']+len(episodes)==m['expected_episodes']
        assert all(row['success']=='True' for row in episodes)
    else:
        assert actual==expected and len(episodes)==m['expected_episodes']
    counts=Counter();placement=Counter();total=0
    with gzip.open(root/'tasks.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            identity=(row['weight_id'],row['route'],row['case_id']);assert identity in groups
            assert row['deadline_status'] in ('on_time','late','undelivered')
            assert row['executed_x'] in ('0','1','')
            counts[identity+(row['deadline_status'],)]+=1;placement[identity+(row['executed_x'],)]+=1;total+=1
    for identity,row in groups.items():
        for status,key in (('on_time','deadline_commands_on_time'),('late','deadline_commands_late'),('undelivered','deadline_commands_undelivered')):
            assert counts[identity+(status,)]==int(row[key])
        assert sum(counts[identity+(s,)] for s in ('on_time','late','undelivered'))==int(row['tasks_generated'])
        assert placement[identity+('1',)]==int(row['executed_drone_tasks'])
        assert placement[identity+('0',)]==int(row['executed_cloud_tasks'])
    for name in ('charts.zip','results.zip'):
        with zipfile.ZipFile(root/name) as archive:assert archive.testzip() is None
    if publication:
        assert total==publication['included_tasks']
        summary=list(csv.DictReader((root/'summary.csv').open(encoding='utf-8')))
        assert len(summary)==len(m['weights'])*len(m['conditions'])*len(m['populations'])
        for row in summary:
            key=(row['weight_id'],row['route'],int(row['num_agents']))
            selected=[e for e in episodes if (e['weight_id'],e['route'],int(e['num_agents']))==key]
            assert row['cohort']=='completed_missions'
            assert int(row['included_cases'])==int(row['cases'])==len(selected)
            assert int(row['attempted_cases'])==m['cases_per_population']
            assert int(row['included_cases'])+int(row['excluded_cases'])==int(row['attempted_cases'])
            for field,source in (('tasks','tasks_generated'),('uav_tasks','executed_drone_tasks'),
                                 ('cloud_tasks','executed_cloud_tasks'),('on_time_commands','deadline_commands_on_time'),
                                 ('late_commands','deadline_commands_late'),('undelivered_commands','deadline_commands_undelivered')):
                assert int(row[field])==sum(int(e[source]) for e in selected)
            if selected:
                energy=sum(float(e['modeled_energy_lower_j']) for e in selected)/len(selected)
                assert abs(float(row['mean_episode_energy_j'])-energy)<1e-9
            else:
                for field in ('mean_episode_energy_j','observed_latency_p95_seconds','deadline_miss_percent',
                              'drone_cpu_utilization_percent','cloud_cpu_utilization_percent','uav_percent','cloud_percent'):
                    assert row[field]=='','Empty cohorts must have undefined metrics'
        for row in csv.DictReader((root/'comparisons.csv').open(encoding='utf-8')):
            n=int(row['num_agents'])
            left={e['case_id'] for e in episodes if (e['weight_id'],e['route'],int(e['num_agents']))==(row['weight_id'],row['route'],n)}
            right={e['case_id'] for e in episodes if (e['weight_id'],e['route'],int(e['num_agents']))==(row['reference_weight'],row['reference_route'],n)}
            assert set(json.loads(row['paired_case_ids']))==left&right
            assert int(row['paired_cases'])==len(left&right)
            assert row['metric']!='mission_success_rate'
        for name in ('charts.zip','results.zip'):
            with zipfile.ZipFile(root/name) as archive:
                assert not any('06_mission_success' in item or Path(item).stem=='mission_success' for item in archive.namelist())
        with zipfile.ZipFile(root/'charts.zip') as archive:
            assert Counter(Path(item).suffix for item in archive.namelist())=={'.png':8,'.pdf':8,'.svg':8}
        with gzip.open(root/'allocations.csv.gz','rt',encoding='utf-8',newline='') as stream:
            assert all((row['weight_id'],row['route'],row['case_id']) in actual for row in csv.DictReader(stream))
    return dict(status='passed',episodes=len(episodes),tasks=total,paired_cases=len(cases),exploratory_populations=m['exploratory_populations'])


if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),indent=2))
