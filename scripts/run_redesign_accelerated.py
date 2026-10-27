"""A4-first scheduling over immutable, validated episode implementations.

Only dispatch order and progress metadata change. Running legacy episodes drain
before their idle pool is retired; completed benchmark transactions are reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import ctypes
from ctypes import wintypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(job):return '/'.join(str(job[k]) for k in ('treatment','weight_id','route','num_agents','case_id'))


def phases(manifest):
    first=[j for j in manifest['jobs_A'] if j['treatment']=='A4' and int(j['case_id'].split('-')[-1])<5]
    ablations=[j for j in manifest['jobs_A'] if j['treatment']!='A4' and int(j['case_id'].split('-')[-1])<5]
    expanded=[j for j in manifest['jobs_A'] if int(j['case_id'].split('-')[-1])>=5]
    seen={key(j) for j in first+ablations+expanded}
    weights=[j for j in manifest['jobs_B'] if key(j) not in seen]
    return [('A4_initial',first),('ablations_initial',ablations),('phase_A_expansion',expanded),('phase_B_weights',weights)]


def bind(root):
    sys.path.insert(0,str(Path(root)/'code'))
    driver=importlib.import_module('run_redesign');driver.verify(root,True)
    return driver


def promote(root,driver):
    reused=[]
    for marker in sorted((Path(root)/'benchmark/12').rglob('complete.json')):
        source=marker.with_name('result.json')
        if not driver.committed_output(source):continue
        packet=read(source);job=packet['job'];target=driver.output_path(root,job)
        if target.exists():
            if not driver.committed_output(target):raise ValueError('Uncommitted primary output prevents reuse')
            continue
        if packet['manifest_sha256']!=digest(Path(root)/'manifest.json'):raise ValueError('Benchmark provenance changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('result.json.pending');shutil.copy2(source,temporary);os.replace(temporary,target)
        shutil.copy2(marker,target.with_name('complete.json'))
        reused.append(dict(job=key(job),source=str(source),sha256=digest(source)))
    return reused


def suspend_process(process):
    """Suspend only the verified old dispatcher; episode children keep running."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);ntdll=ctypes.WinDLL('ntdll')
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    ntdll.NtSuspendProcess.argtypes=[wintypes.HANDLE];ntdll.NtSuspendProcess.restype=wintypes.LONG
    handle=kernel.OpenProcess(0x0800|0x1000,False,process.pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        result=ntdll.NtSuspendProcess(handle)
        if result:raise OSError('Dispatcher suspension failed: '+str(result))
    finally:kernel.CloseHandle(handle)


class LegacyDrain:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'accelerated';self.folder.mkdir(exist_ok=True)
        self.path=self.folder/'handoff.json';self.stable_since=None
        if self.path.exists():
            self.state=read(self.path)
            if self.state['status']=='complete':self.workers=[];return
            self.coordinator=psutil.Process(self.state['coordinator']['pid'])
            if self.coordinator.create_time()!=self.state['coordinator']['creation_time']:raise ValueError('Dispatcher PID reused')
            self.workers=[psutil.Process(r['pid']) for r in self.state['workers'] if psutil.pid_exists(r['pid'])]
            if 'dispatcher_suspended_at' not in self.state:
                suspend_process(self.coordinator);self.state['dispatcher_suspended_at']=time.time();write(self.path,self.state)
            return
        launch=read(self.root/'launch.json');launcher=psutil.Process(launch['pid'])
        if abs(launcher.create_time()-launch['creation_time'])>1:raise ValueError('Legacy launcher identity changed')
        candidates=[p for p in [launcher]+launcher.children(recursive=True)
            if '--worker' in p.cmdline() and str(self.root) in p.cmdline()]
        # A Windows venv launcher waits on its actual Python child.
        coordinators=[p for p in candidates if 'python312' in p.exe().lower()]
        if len(coordinators)!=1:raise ValueError('Cannot identify the original Python dispatcher uniquely')
        self.coordinator=coordinators[0]
        self.workers=[p for p in self.coordinator.children(recursive=True) if '--multiprocessing-fork' in p.cmdline()]
        status=read(self.root/'status.json')
        self.state=dict(status='draining',started=time.time(),launcher=launch,
            coordinator=dict(pid=self.coordinator.pid,creation_time=self.coordinator.create_time(),command=self.coordinator.cmdline()),
            workers=[dict(pid=p.pid,creation_time=p.create_time(),command=p.cmdline()) for p in self.workers],
            previous_status=status,protected_jobs=status.get('active',[]),
            reason='A4-first dispatch; finish active episodes and defer remaining worker-count benchmarks')
        write(self.path,self.state)
        suspend_process(self.coordinator)
        self.state['dispatcher_suspended_at']=time.time();write(self.path,self.state)

    def tick(self,driver):
        if self.state['status']=='complete':return set(),0.
        protected=set(self.state['protected_jobs']);active=[]
        for identity in self.state['workers']:
            try:
                p=psutil.Process(identity['pid'])
                if p.create_time()!=identity['creation_time']:raise ValueError('Legacy episode PID reused')
            except psutil.NoSuchProcess:continue
            path=self.root/f"worker-{p.pid}.json"
            record=read(path) if path.exists() else dict(phase='initializing')
            if record['phase'] not in ('complete','error'):
                if record.get('job'):protected.add(record['job'])
                active.append(record)
        if active:self.stable_since=None
        elif self.stable_since is None:self.stable_since=time.time()
        elif time.time()-self.stable_since>=5:
            # All owned scientific episodes have committed or saved an error.
            # Only idle children and the suspended dispatcher are terminated.
            for identity in self.state['workers']:
                try:
                    p=psutil.Process(identity['pid'])
                    if p.create_time()==identity['creation_time']:p.terminate()
                except psutil.NoSuchProcess:pass
            self.coordinator.terminate()
            self.state.update(status='complete',finished=time.time(),episodes_interrupted=0)
            write(self.path,self.state);return set(),0.
        self.state['last_checked']=time.time();self.state['active_episode_records']=active;write(self.path,self.state)
        # Reserve future trace growth of the draining job, in addition to live RSS.
        reserve=sum(max(0.,20.-r.get('rss_bytes',0)/2**30) for r in active)
        return protected,reserve


def run(root):
    root=Path(root).resolve();folder=root/'accelerated';driver=bind(root);manifest=read(root/'manifest.json')
    plan=read(folder/'manifest.json')
    if digest(root/'manifest.json')!=plan['parent_manifest_sha256']:raise ValueError('Frozen simulation identity changed')
    for gate in ('preflight.json','reference_validation.json'):
        if read(root/gate)['status']!='passed':raise ValueError('Existing simulation acceptance gate failed')
    drain=LegacyDrain(root);reused=promote(root,driver)
    if reused:write(folder/f"reuse-{time.time_ns()}.json",reused)
    sizes={2:1.,4:1.,8:1.5,16:5.,32:20.}
    new_count=read(folder/'status.json').get('completed_new',0) if (folder/'status.json').exists() else 0
    launch=read(folder/'launch.json');original_launch=read(root/'launch.json')
    for name,jobs in phases(manifest):
        queue=[j for j in jobs if not driver.committed_output(driver.output_path(root,j))];pending={}
        phase_start=time.time()
        with ProcessPoolExecutor(max_workers=16,initializer=driver.initialize,initargs=(str(root),)) as executor:
            while queue or pending:
                if time.time()-original_launch['first_started_unix']>=manifest['budget_seconds']:
                    for process in executor._processes.values():
                        if process.is_alive():process.terminate()
                    raise TimeoutError('Original wall cap reached; no active budget extension')
                protected,old_reserve=drain.tick(driver)
                imported=promote(root,driver)
                if imported:write(folder/f"reuse-{time.time_ns()}.json",imported)
                queue=[j for j in queue if not driver.committed_output(driver.output_path(root,j))]
                for job in list(queue):
                    if len(pending)>=16:break
                    if key(job) in protected:continue
                    active=list(pending.values())
                    if job['num_agents']>=16 and sum(j['num_agents']>=16 for j in active)>=8:continue
                    if job['num_agents']==32 and sum(j['num_agents']==32 for j in active)>=4:continue
                    planned=sum(sizes[j['num_agents']] for j in active)+sizes[job['num_agents']]
                    children=psutil.Process().children(recursive=True)
                    actual=sum(p.memory_info().rss for p in children if p.is_running())/2**30
                    available=psutil.virtual_memory().available/2**30
                    if max(0.,planned-actual)+old_reserve>available-8.:continue
                    future=executor.submit(driver.episode,('primary',job,250.));pending[future]=job;queue.remove(job)
                if not pending and queue and drain.state['status']=='complete':
                    raise MemoryError('RAM reserve blocks dispatch; data preserved')
                completed,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED) if pending else (set(),set())
                if not pending and not completed:time.sleep(5)
                for future in completed:
                    job=pending.pop(future);future.result();new_count+=1
                count=sum(driver.committed_output(driver.output_path(root,j)) for j in jobs)
                status=dict(status='running',phase=name,phase_completed=count,phase_total=len(jobs),completed_new=new_count,
                    phase_started_at=phase_start,started_at=launch['started_at'],updated_at=time.time(),workers=16,
                    active=[key(j) for j in pending.values()],remaining=len(queue),legacy_drain=drain.state['status'],
                    available_ram_gib=psutil.virtual_memory().available/2**30,wall_cap_seconds=manifest['budget_seconds'])
                write(folder/'status.json',status)
        write(folder/(name+'.json'),dict(status='complete',completed=len(jobs),finished_at=time.time()))
    write(folder/'status.json',dict(status='complete',finished_at=time.time(),completed_new=new_count,primary_total=1380))


def launch(root):
    root=Path(root).resolve();folder=root/'accelerated';folder.mkdir(exist_ok=True)
    for process in psutil.process_iter(['cmdline']):
        argv=process.info['cmdline'] or []
        if '--accelerated-worker' in argv and str(root) in argv:raise ValueError('Accelerated dispatcher already active')
    source=Path(__file__).resolve();frozen=folder/'code'/source.name;frozen.parent.mkdir(exist_ok=True)
    if frozen.exists() and digest(frozen)!=digest(source):raise ValueError('Existing accelerated source is immutable')
    if not frozen.exists():shutil.copy2(source,frozen)
    manifest=read(root/'manifest.json')
    plan=dict(protocol='a4_first_scheduling_v1',parent_manifest_sha256=digest(root/'manifest.json'),
        source_sha256=digest(frozen),simulation_source='unchanged parent frozen code',workers=16,
        priority_phases=[dict(name=n,jobs=[key(j) for j in jobs]) for n,jobs in phases(manifest)],
        deferred_benchmarks=[16,18],created_at=time.time(),wall_cap_seconds=manifest['budget_seconds'])
    if not (folder/'manifest.json').exists():write(folder/'manifest.json',plan)
    command=[sys.executable,str(frozen),'--accelerated-worker',str(root)]
    environment=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (folder/'stdout.log').open('a') as out,(folder/'stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    previous=read(folder/'launch.json') if (folder/'launch.json').exists() else None
    record=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,
        started_at=previous['started_at'] if previous else time.time(),last_launched_at=time.time())
    write(folder/'launch.json',record);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--accelerated-worker',type=Path)
    args=parser.parse_args()
    if args.accelerated_worker:
        try:run(args.accelerated_worker)
        except Exception as exc:
            import traceback
            path=args.accelerated_worker/'accelerated/status.json';previous=read(path) if path.exists() else {}
            write(path,dict(previous,status='error',error=str(exc),traceback=traceback.format_exc(),time=time.time()));raise
    elif args.launch:print(json.dumps(launch(args.root),indent=2))
