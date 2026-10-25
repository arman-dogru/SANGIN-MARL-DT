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
    return dict(status='passed',episodes=len(episodes),tasks=total,paired_cases=len(cases),exploratory_populations=m['exploratory_populations'])


if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),indent=2))
