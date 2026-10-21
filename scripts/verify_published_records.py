"""Verify report hashes/counts in the workspace or an actual Git revision."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def verify(directory, git_ref=None):
    directory=Path(directory).resolve()
    def read(name):
        path=directory/name
        if git_ref:
            relative=path.relative_to(ROOT).as_posix()
            return subprocess.check_output(['git','show',f'{git_ref}:{relative}'],cwd=ROOT)
        return path.read_bytes()
    verification=json.loads(read('verification.json'))
    manifest=read('manifest.json')
    assert hashlib.sha256(manifest).hexdigest()==verification['manifest_sha256']
    for name,item in verification['record_exports'].items():
        data=read(name)
        assert hashlib.sha256(data).hexdigest()==item['sha256'],name
        if name.endswith('.gz'):
            rows=[json.loads(line) for line in gzip.decompress(data).decode('utf-8').splitlines()]
        else:
            rows=list(csv.DictReader(io.StringIO(data.decode('utf-8'),newline='')))
        assert len(rows)==item['records'],name
        if name in ('episodes.jsonl.gz','episodes.csv'):
            assert len({(r['condition'],r['case_id']) for r in rows})==verification['episodes']
    return dict(status='passed',source=git_ref or 'workspace',episodes=verification['episodes'],
                exports=len(verification['record_exports']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True); parser.add_argument('--git-ref')
    args=parser.parse_args(); print(json.dumps(verify(args.directory,args.git_ref)))
