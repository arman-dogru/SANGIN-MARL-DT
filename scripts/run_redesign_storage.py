"""Versioned dispatch recovery with lossless disk-backed telemetry.

The original simulation package, inputs and elapsed wall lease remain frozen.
Resource-paused attempts remain intact; exact same-case retries are explicit.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
from run_redesign_accelerated import read, write, digest, key, phases, bind, promote


def initialize(root):
    global DRIVER
    DRIVER=bind(root)
    DRIVER.initialize(root)
    from compact_telemetry import install
    install(DRIVER)


def episode(argument):
    namespace,job,horizon=argument
    output=DRIVER.output_path(DRIVER.ROOT,job,namespace)
    working=output.parent/'compact_working'
    if working.exists():
        if DRIVER.committed_output(output):raise ValueError('Completed storage cannot be overwritten')
        retained=output.parent/'interrupted_attempts'/str(time.time_ns())
        retained.mkdir(parents=True)
        shutil.move(str(working),str(retained/'compact_working'))
    return DRIVER.episode(argument)


def identity_matches(record):
    try:
        p=psutil.Process(record['pid'])
        return abs(p.create_time()-record['creation_time'])<.1 and p.cmdline()==record['command']
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def verify_recovery(root):
    """Only the recorded empty dispatcher can be replaced; held work stays held."""
    root=Path(root);folder=root/'accelerated'
    proof=read(folder/'empty_pool_verification.json')
    if proof['scientific_episodes_in_flight']!=0:raise ValueError('Previous dispatcher has unfinished work')
    old=proof['paused_coordinator']
    if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['creation_time'])<.1:
        raise ValueError('Previous dispatcher still exists')
    pause=read(folder/'resource_pause.json')
    if pause['status']!='resource_paused' or not pause['raw_state_preserved']:raise ValueError('Missing preserved resource pause')
    if not identity_matches(pause):raise ValueError('Retained attempt identity changed')
    process=psutil.Process(pause['pid'])
    # Windows reports suspended processes as running; verify every thread instead.
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenThread.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    ntdll=ctypes.WinDLL('ntdll');ntdll.NtQueryInformationThread.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.ULONG,ctypes.c_void_p]
    for thread in process.threads():
        handle=kernel.OpenThread(0x0040,False,thread.id)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        count=wintypes.ULONG()
        try:
            result=ntdll.NtQueryInformationThread(handle,35,ctypes.byref(count),ctypes.sizeof(count),None)
            if result or count.value==0:raise ValueError('Retained worker is not fully suspended')
        finally:kernel.CloseHandle(handle)
    return pause


def condition(job):return tuple(job[k] for k in ('treatment','weight_id','route','num_agents'))


def admissible(job,active,available,actual,commit_available):
    sizes={2:1.,4:1.,8:1.5,16:5.,32:20.}
    if len(active)>=16:return False
    if job['num_agents']>=16 and sum(j['num_agents']>=16 for j in active)>=8:return False
    if job['num_agents']==32 and sum(j['num_agents']==32 for j in active)>=4:return False
    planned=sum(sizes[j['num_agents']] for j in active)+sizes[job['num_agents']]
    growth=max(0.,planned-actual)
    return growth<=available-8. and growth<=commit_available-8.


def run(root,revision):
    root=Path(root).resolve();folder=root/'accelerated';revision=Path(revision)
    plan=read(revision/'manifest.json');driver=bind(root);manifest=read(root/'manifest.json')
    for path,value in plan['source_sha256'].items():
        if digest(revision/'code'/path)!=value:raise ValueError('Frozen recovery source changed')
    if digest(root/'manifest.json')!=plan['parent_manifest_sha256']:raise ValueError('Parent source identity changed')
    for gate in ('preflight.json','reference_validation.json'):
        if read(root/gate)['status']!='passed':raise ValueError('Simulation validation gate failed')
    verify_recovery(root)
    imported=promote(root,driver)
    if imported:write(revision/f'reuse-{time.time_ns()}.json',imported)
    original=read(root/'launch.json');started=time.time();new_count=0;errors=[];blocked=set()
    for marker in (root/'primary').rglob('error.json'):
        failed=read(marker)
        if failed.get('job'):blocked.add(condition(failed['job']))
    with ProcessPoolExecutor(max_workers=16,initializer=initialize,initargs=(str(root),)) as executor:
        for phase,jobs in phases(manifest):
            queue=[j for j in jobs if not driver.committed_output(driver.output_path(root,j))];pending={}
            while queue or pending:
                if time.time()-original['first_started_unix']>=manifest['budget_seconds']:
                    write(revision/'budget_stop.json',dict(time=time.time(),active=[key(j) for j in pending.values()]))
                    for p in executor._processes.values():
                        if p.is_alive():p.terminate()
                    raise TimeoutError('Original wall cap reached; lease is unchanged')
                if shutil.disk_usage(root).free<20*2**30:raise OSError('Disk reserve below 20 GiB; raw records retained')
                queue=[j for j in queue if condition(j) not in blocked]
                for job in list(queue):
                    children=psutil.Process().children(recursive=True)
                    actual=sum(p.memory_info().rss for p in children if p.is_running())/2**30
                    if not admissible(job,list(pending.values()),psutil.virtual_memory().available/2**30,actual,
                                      psutil.swap_memory().free/2**30):continue
                    pending[executor.submit(episode,('primary',job,250.))]=job;queue.remove(job)
                completed,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED) if pending else (set(),set())
                for future in completed:
                    job=pending.pop(future)
                    try:future.result();new_count+=1
                    except Exception as exc:
                        errors.append(dict(job=key(job),error=str(exc),time=time.time()));blocked.add(condition(job))
                        write(revision/'execution_errors.json',errors)
                status=dict(status='running' if pending else 'resource_wait',phase=phase,
                    phase_completed=sum(driver.committed_output(driver.output_path(root,j)) for j in jobs),phase_total=len(jobs),
                    completed_new=plan['previous_completed_new']+new_count,started_at=plan['original_accelerated_start'],
                    updated_at=time.time(),revision_started_at=started,workers=16,active=[key(j) for j in pending.values()],
                    remaining=len(queue),available_ram_gib=psutil.virtual_memory().available/2**30,
                    retained_resource_attempt=plan['retained_attempt'],storage_backend='sqlite_zlib_v1',
                    scheduler_revision=str(revision),execution_errors=len(errors),wall_cap_seconds=manifest['budget_seconds'])
                write(folder/'status.json',status)
                if not pending and queue:time.sleep(15)
            write(revision/(phase+'.json'),dict(status='complete' if not blocked else 'complete_with_blocked_conditions',time=time.time()))
    write(folder/'status.json',dict(status='complete' if not blocked else 'incomplete_errors',completed_new=plan['previous_completed_new']+new_count,
        execution_errors=len(errors),finished_at=time.time(),scheduler_revision=str(revision)))


def launch(root):
    root=Path(root).resolve();folder=root/'accelerated';driver=bind(root);pause=verify_recovery(root)
    evidence=read(root/'storage_validation/status.json')
    if evidence['status']!='passed' or evidence['parent_manifest_sha256']!=digest(root/'manifest.json'):
        raise ValueError('Storage equivalence validation is required')
    for name,value in evidence['source_sha256'].items():
        if digest(Path(__file__).parent/name)!=value:raise ValueError('Storage validation source changed')
    for p in psutil.process_iter(['cmdline']):
        argv=p.info['cmdline'] or []
        if ('--storage-worker' in argv or '--accelerated-worker' in argv) and str(root) in argv:
            raise ValueError('An accelerated dispatcher is already active')
    revision=folder/'revisions'/'storage-v1'
    revision.mkdir(parents=True,exist_ok=True);code=revision/'code';code.mkdir(exist_ok=True)
    names=('run_redesign_storage.py','compact_telemetry.py','run_redesign_accelerated.py')
    for name in names:
        source=Path(__file__).parent/name;target=code/name
        if target.exists() and digest(source)!=digest(target):raise ValueError('Recorded recovery source is immutable')
        if not target.exists():shutil.copy2(source,target)
    status=read(folder/'status.json');previous=read(folder/'launch.json')
    plan=dict(protocol='a4_lossless_storage_v1',parent_manifest_sha256=digest(root/'manifest.json'),
        source_sha256={n:digest(code/n) for n in names},created_at=time.time(),
        previous_completed_new=status.get('completed_new',0),original_accelerated_start=previous['started_at'],
        original_wall_cap_seconds=read(root/'manifest.json')['budget_seconds'],original_first_started_unix=read(root/'launch.json')['first_started_unix'],
        retained_attempt=dict(job=pause['job'],pid=pause['pid'],creation_time=pause['creation_time'],status='resource_paused',
            retry='Exact same inputs in primary namespace; original suspended process and benchmark files retained'),
        simulation_changes='None; immutable event payloads stored losslessly in SQLite and compressed archives')
    if (revision/'manifest.json').exists():plan=read(revision/'manifest.json')
    else:write(revision/'manifest.json',plan)
    command=[sys.executable,str(code/'run_redesign_storage.py'),'--storage-worker',str(root),'--revision',str(revision)]
    environment=dict(os.environ,PYTHONPATH=os.pathsep.join((str(code),str(root/'code'))),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (revision/'stdout.log').open('a') as out,(revision/'stderr.log').open('a') as err:
        p=subprocess.Popen(command,cwd=root,env=environment,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    record=dict(pid=p.pid,creation_time=psutil.Process(p.pid).create_time(),command=command,started_at=previous['started_at'],
        last_launched_at=time.time(),scheduler_revision=str(revision))
    write(revision/f'launch-{time.time_ns()}.json',record);write(folder/'launch.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--storage-worker',type=Path);parser.add_argument('--revision',type=Path)
    args=parser.parse_args()
    if args.launch:print(__import__('json').dumps(launch(args.root),indent=2))
    elif args.storage_worker:
        try:run(args.storage_worker,args.revision)
        except Exception as exc:
            import traceback
            path=args.storage_worker/'accelerated/status.json';prior=read(path) if path.exists() else {}
            write(path,dict(prior,status='error',error=str(exc),traceback=traceback.format_exc(),updated_at=time.time()));raise
