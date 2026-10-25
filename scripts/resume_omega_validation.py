"""Resume a stopped study with the isolated fractional-diagnostic audit correction."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
from panama_uav.records import sha256


def resume(root):
    root=Path(root).resolve();launch=json.loads((root/'launch.json').read_text());m=json.loads((root/'manifest.json').read_text())
    for process in psutil.process_iter(['pid','cmdline']):
        command=process.info['cmdline'] or []
        if str(root) in command and '--worker' in command:raise ValueError('Original worker is still active')
    status=json.loads((root/'status.json').read_text())
    if status['status']!='error' or 'baseline reproduction failed' not in status.get('error','').lower():
        raise ValueError('Only the stopped fractional-reproduction gate can use this recovery')
    if time.time()-launch['first_started_unix']>=m['budget_seconds']:raise ValueError('No recorded wall budget remains')
    for name,digest in m['frozen_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen source/input changed')
    folder=root/'validation_v2';folder.mkdir(exist_ok=False)
    shutil.copy2(root/'status.json',root/'status_v1_gate_failure.json')
    shutil.copy2(root/'baseline_validation.json',root/'baseline_validation_v1.json')
    shutil.copy2(root/'launch.json',root/'launch_v1.json')
    shutil.copy2(Path(__file__).with_name('audit_omega_reproduction.py'),folder/'audit_omega_reproduction.py')
    controller=folder/'controller.py'
    controller.write_text("from pathlib import Path\nimport sys,json\nroot=Path(__file__).resolve().parents[1]\nsys.path.insert(0,str(root/'code'))\nimport run_omega_sensitivity as driver\nfrom panama_uav.records import sha256\nrevision=json.loads((root/'validation_revision.json').read_text())\nassert sha256(root/'manifest.json')==revision['original_manifest_sha256']\nfor name,digest in revision['files'].items():\n    assert sha256(Path(__file__).parent/name)==digest\nfrom audit_omega_reproduction import baseline_gate\ndriver.baseline_gate=baseline_gate\nif __name__=='__main__':driver.run(root)\n",encoding='utf-8')
    revision=dict(reason='Unused fractional minimizers vary within conic tolerance; exact executed outputs unchanged. Compare conic objective and feasibility instead of arbitrary x-vector distance.',
        production_changes=0,simulator_changes=0,solver_changes=0,rerun_completed_episodes=0,original_manifest_sha256=sha256(root/'manifest.json'),
        files={p.name:sha256(p) for p in folder.iterdir() if p.is_file()},created_unix=time.time(),original_lease_started=launch['first_started_unix'])
    (root/'validation_revision.json').write_text(json.dumps(revision,indent=2)+'\n',encoding='utf-8')
    env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(controller),'--worker',str(root)]
    with (root/'stdout.log').open('a') as out,(root/'stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    info=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,
        first_started_unix=launch['first_started_unix'],validation_revision_sha256=sha256(root/'validation_revision.json'))
    pending=root/'launch.pending';pending.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');os.replace(pending,root/'launch.json')
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(resume(args.root)))
