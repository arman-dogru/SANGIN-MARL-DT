"""Export completed population-extension artifacts without copying large assets."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def publish(run, destination, verifier):
    run, destination, verifier = map(lambda p:Path(p).resolve(),(run,destination,verifier))
    assert json.loads((run/'status.json').read_text())['status']=='complete'
    audit=json.loads((run/'accounting_audit.json').read_text())
    assert audit['status']=='passed' and audit['new_replay_audits']==40
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    files=['base.toml','pilot.toml','manifest.json','cases.json','status.json','validation.json',
        'accounting_audit.json','environment.json','set_index.json','summary.csv','summary.json',
        'comparisons.csv','handover.csv','placement_by_active_count.csv','placement_intervals.csv',
        'episodes.csv','rounds.csv','audits.csv','combined_summary.csv','combined_summary.json',
        'combined_comparisons.csv','combined_pairwise.csv','combined_coordination.csv',
        'combined_episodes.csv','case_outcomes.csv','mapf_reference_serial.json']
    pairs=[(run/name,name,False) for name in files]
    pairs += [(run/'REPORT.md','report.md',False),(run/'tasks.csv','tasks.csv.gz',True),
              (run/'audits.jsonl','audits.jsonl.gz',True)]
    pairs += [(path,'figures/'+path.name,False) for path in sorted((run/'figures').iterdir()) if path.is_file()]
    pairs += [(run/'code/extend_uav6.py','analysis/extend.py',False),
              (run/'code/base_pilot.py','analysis/base_pilot.py',False)]
    pairs += [(run/'analysis'/name,'analysis/'+name,False) for name in
              ('finalize.py','replay_instantaneous.py','reference_reproduction.json','REPRODUCE.md')]
    pairs += [(Path(__file__),'analysis/publish.py',False),(verifier,'analysis/verify_bundle.py',False)]
    for source,name,compressed in pairs:
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            with source.open('rb') as src, target.open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
            with gzip.open(target,'rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==digest(source)
        else:
            shutil.copy2(source,target)
            assert digest(source)==digest(target)
        assert target.stat().st_size < 25_000_000,target
        project=Path.cwd().resolve()
        records.append(dict(study='uav6-extension',file=target.relative_to(project/'results').as_posix(),
            source=source.relative_to(project).as_posix(),source_sha256=digest(source),sha256=digest(target),
            format='gzip-original' if compressed else 'original'))
    catalog=destination.parents[1]/'files.csv'
    with catalog.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        old=[r for r in reader if r['study']=='uav6-extension']
    if old:
        assert old==records,'An existing export differs; preserve it and use a new version'
    else:
        with catalog.open('a',encoding='utf-8',newline='') as stream:
            csv.DictWriter(stream,fieldnames=fields).writerows(records)
    print(json.dumps(dict(status='exported',files=len(records),bytes=sum((destination/Path(r['file']).relative_to('uav6/agents16-32')).stat().st_size for r in records))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('output/uav6_pilot/uav6-20261024-01/extensions/agents16-32-01'))
    parser.add_argument('--destination',type=Path,default=Path('results/uav6/agents16-32'))
    parser.add_argument('--verifier',type=Path,default=Path('tmp/verify_uav6_extension.py'))
    args=parser.parse_args()
    publish(args.run,args.destination,args.verifier)
