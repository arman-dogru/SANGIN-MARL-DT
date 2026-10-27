"""Replay a preserved completed episode with lossless storage, without replacement."""
import argparse
from pathlib import Path
import pickle
import gzip
import sys
import time

from run_redesign_accelerated import read, write, digest, bind
from run_redesign_storage import initialize, episode


def validate(root):
    root=Path(root).resolve();driver=bind(root)
    candidates=sorted((root/'primary/A4/W1/allocated/8').glob('*/complete.json'))
    if not candidates:raise ValueError('A preserved audited reference is required')
    reference=candidates[0].with_name('result.json');before=read(reference);job=before['job']
    folder=root/'storage_validation';folder.mkdir(exist_ok=True)
    write(folder/'status.json',dict(status='running',reference=str(reference),started_at=time.time()))
    initialize(str(root));metadata=episode(('storage_validation',job,250.))
    after=read(driver.output_path(root,job,'storage_validation'))
    if driver.stable_signature(before)!=driver.stable_signature(after):raise ValueError('Preserved episode execution differs')
    # The archive must be independently replayable, without its SQLite files.
    with gzip.open(after['audit']['archive'],'rb') as stream:trace=pickle.load(stream)
    from panama_uav.redesign_audit import audit_redesign
    replay=audit_redesign(trace,driver.infer)
    if replay['status']!='passed':raise ValueError('Stored archive replay failed')
    names=('run_redesign_storage.py','compact_telemetry.py','run_redesign_accelerated.py')
    result=dict(status='passed',reference=str(reference),reference_sha256=digest(reference),
        parent_manifest_sha256=digest(root/'manifest.json'),exact_executed_signature=True,archive_replay=replay,
        validation_metadata=metadata,source_sha256={n:digest(Path(__file__).parent/n) for n in names},finished_at=time.time())
    write(folder/'status.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:print(__import__('json').dumps(validate(args.root),indent=2))
    except Exception as exc:
        import traceback
        write(args.root/'storage_validation/status.json',dict(status='error',error=str(exc),traceback=traceback.format_exc(),time=time.time()));raise
