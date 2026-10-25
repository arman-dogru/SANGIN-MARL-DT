"""Independent accounting and optional read-back audit of the completed study."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import zipfile

from panama_uav.records import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def retain_readbacks(result,evidence):
    if (result['manifest_sha256']!=evidence['manifest_sha256'] or
            result['archive_inventory_sha256']!=evidence['archive_inventory_sha256']):
        raise ValueError('Read-back evidence belongs to a different inventory')
    result['complete_trace_readbacks']=sorted(set(result['complete_trace_readbacks'])|set(evidence['complete_trace_readbacks']))
    return result


def audit(root,read_back=False):
    root=Path(root).resolve()
    spec=importlib.util.spec_from_file_location('omega_audit_driver',root/'code/run_omega_sensitivity.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    m=driver.verify(root,check_original=True);cases=read(root/'cases.json')
    bank={c['case_id']:c for c in cases}
    expected={(w['id'],c['name'],case['case_id']) for w in m['weights'] for c in m['conditions'] for case in cases}
    actual=set();totals=Counter();groups={};readbacks=[];inventory=[]
    for path in sorted((root/'episodes').glob('*/*/*.result.json')):
        packet=read(path);row=packet['row'];case=bank[row['case_id']]
        identity=(packet['weight']['id'],row['condition'],row['case_id'])
        assert identity not in actual,'Duplicate episode'
        actual.add(identity)
        assert sha256(path)==read(path.with_suffix('.meta.json'))['sha256'],'Transaction changed'
        assert packet['manifest_sha256']==sha256(root/'manifest.json'),'Wrong experiment manifest'
        assert packet['audit']['status']=='passed','Replay failed'
        assert sha256(packet['audit']['archive'])==packet['audit']['archive_sha256'],'Trace archive changed'
        inventory.append((identity,sha256(path),packet['audit']['archive_sha256']))
        for name in ('seed','num_agents','starts','goals','orbital_offset_seconds','map_sha256'):
            assert row[name]==case[name],name+' changed'
        assert len(packet['tasks'])==row['tasks_generated']
        partition=Counter(t['deadline_status'] for t in packet['tasks'])
        assert partition['on_time']==row['deadline_commands_on_time']
        assert partition['late']==row['deadline_commands_late']
        assert partition['undelivered']==row['deadline_commands_undelivered']
        assert sum(partition.values())==sum(partition[k] for k in ('on_time','late','undelivered'))==row['tasks_generated']
        counts=Counter(t.get('executed_x') for t in packet['tasks'])
        assert set(counts)<={0,1,None}
        assert counts[1]==row['executed_drone_tasks'] and counts[0]==row['executed_cloud_tasks']
        assert len(packet['allocations'])==counts[0]+counts[1]
        for t in packet['tasks']:
            if t.get('observed_latency_seconds') is not None:
                assert abs(t['observed_latency_seconds']-(t['delivered_at']-t['source_time']))<1e-9
                assert (t['observed_latency_seconds']<=.25+1e-9)==(t['deadline_status']=='on_time')
            assert t['input_bits']==100000 and t['cycles_per_bit']==1000.
        if row['route']!='allocated':assert not counts[1 if row['route']=='cloud' else 0]
        key=(identity[0],row['route'],row['num_agents']);group=groups.setdefault(key,Counter())
        for counter in (totals,group):
            counter.update(episodes=1,tasks=row['tasks_generated'],uav=counts[1],cloud=counts[0],unassigned=counts[None],
                on_time=partition['on_time'],late=partition['late'],undelivered=partition['undelivered'],
                successes=int(row['success']),censored=row['deadline_censored_tasks'])
        if read_back and identity[0] in ('W1','W5') and row['case_id'].endswith('-000'):
            # Verify the digest before unpickling locally generated archives.
            with gzip.open(packet['audit']['archive'],'rb') as stream:trace=pickle.load(stream)
            assert trace['frames'][0]['positions']==case['starts'] and trace['goals']==case['goals']
            assert driver.signatures(row,packet['tasks'],packet['rounds'],trace)==packet['signatures']
            readbacks.append('/'.join(identity));del trace
    assert actual==expected,'Missing or unexpected simulations'
    assert not list((root/'episodes').glob('*/*/*.error.json')),'Recorded simulation failure'
    summary=list(csv.DictReader((root/'summary.csv').open(encoding='utf-8')))
    assert len(summary)==len(groups)
    for row in summary:
        group=groups[(row['weight_id'],row['route'],int(row['num_agents']))]
        for field,counter in (('cases','episodes'),('tasks','tasks'),('uav_tasks','uav'),('cloud_tasks','cloud'),
                              ('on_time_commands','on_time'),('late_commands','late'),('undelivered_commands','undelivered'),('successful_missions','successes')):
            assert int(row[field])==group[counter],field+' mismatch'
    if not m['smoke']:
        baseline=read(root/'baseline_validation.json')
        assert baseline['status']=='passed' and baseline['cases']==75
        assert all(d['passed'] and all(d['matches'].values()) for d in baseline['details'])
    formats=Counter(p.suffix for p in (root/'figures').rglob('*') if p.is_file())
    assert formats=={'.png':9,'.pdf':9,'.svg':9},'Missing figure exports'
    result=dict(status='passed',totals=dict(totals),summary_groups=len(groups),paired_identities=True,
        baseline_reproductions=75 if not m['smoke'] else None,archive_hashes_checked=len(actual),
        complete_trace_readbacks=readbacks,figure_formats=dict(formats),production_and_historical_hashes_unchanged=True)
    result.update(manifest_sha256=sha256(root/'manifest.json'),
        archive_inventory_sha256=hashlib.sha256(json.dumps(sorted(inventory),separators=(',',':')).encode()).hexdigest())
    evidence=root/'readback_audit.json'
    if read_back:evidence.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    elif evidence.exists():result=retain_readbacks(result,read(evidence))
    if evidence.exists():result['readback_evidence_sha256']=sha256(evidence)
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def export(root,destination):
    """Export only a completed, independently audited matrix; raw traces stay local."""
    root,destination=Path(root).resolve(),Path(destination).resolve()
    repository_results=Path(__file__).resolve().parents[1]/'results'
    if destination.is_relative_to(repository_results):
        raise ValueError('Full-cohort exports stay local; use publish_omega_completed.py for repository results')
    if read(root/'status.json')['status']!='complete':raise ValueError('The matrix is not complete')
    if read(root/'independent_audit.json')['status']!='passed':raise ValueError('Independent audit required')
    if destination.exists():raise ValueError('Use a new destination; result snapshots cannot be overwritten')
    destination.mkdir(parents=True)
    files=list(root.glob('*.csv'))+list(root.glob('*.csv.gz'))
    files+=[root/name for name in ('report.md','manifest.json','cases.json','baseline_validation.json','validation.json',
                                 'independent_audit.json','regression.json','status.json','baseline_validation_v1.json',
                                 'status_v1_gate_failure.json','validation_revision.json','readback_audit.json',
                                 'independent_audit.stdout.log') if (root/name).exists()]
    for folder in ('configs','figures','analysis'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file())
    for path in files:
        target=destination/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (destination/'README.md').write_text('# Latency–energy sensitivity\n\n'
        'Parameter-only results from five weights, three execution routes, five populations and five paired cases: 375 episodes. '
        'The 16/32-robot results are exploratory scale tests.\n\n'
        '- [Numerical report and research questions](report.md)\n'
        '- [Comparative table](summary.csv) and [all episodes](episodes.csv)\n'
        '- [Paired differences and confidence intervals](comparisons.csv)\n'
        '- [Independent audit](independent_audit.json) and [baseline reproduction](baseline_validation.json)\n'
        '- [Figures ZIP](charts.zip) and [tables, configurations and report ZIP](results.zip)\n\n'
        'Task and allocation CSVs use gzip. Full replay traces and the checkpoint remain local under the manifest run directory. '
        'Run `python verify_bundle.py` to verify file hashes and accounting.\n',encoding='utf-8')
    verifier=Path(__file__).with_name('verify_omega_bundle.py')
    shutil.copy2(verifier,destination/'verify_bundle.py')
    with (destination/'files.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','sha256','bytes']);writer.writeheader()
        for path in sorted(destination.rglob('*')):
            if path.is_file() and path.name!='files.csv':writer.writerow(dict(path=path.relative_to(destination).as_posix(),sha256=sha256(path),bytes=path.stat().st_size))
    for name,paths in (('charts.zip',list((destination/'figures').rglob('*'))),
                       ('results.zip',[p for p in destination.rglob('*') if 'figures' not in p.relative_to(destination).parts and p.suffix!='.zip'])):
        with zipfile.ZipFile(destination/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in paths:
                if path.is_file():archive.write(path,path.relative_to(destination))
    return dict(destination=str(destination),files=len(files),charts=27)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--read-back',action='store_true');parser.add_argument('--export',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.root,args.read_back),indent=2))
    if args.export:print(json.dumps(export(args.root,args.export),indent=2))
