"""Resumable local dispatcher for the leo_study matrix. Box only; one subprocess per job."""
import json, os, subprocess, sys, time
from pathlib import Path
STUDY=Path('/workspace/leo_study');OUT=STUDY/'runs/main';OUT.mkdir(parents=True,exist_ok=True)
COUNTS=(2,4,8,16,32);CASES=10
MAX_WORKERS=int(os.environ.get('MAX_WORKERS',6));MAX_LARGE=2;MAX_16=3;MIN_AVAIL_GB=2.5

def jid(j):
    return f"{j['group']}__{j['treatment']}__{j['route']}__{j['network']}__dl{int(round(j.get('downlink_seconds',0)*1000))}__n{j['num_agents']}__c{j['case']:02d}"

def matrix():
    jobs=[]
    def add(group,treatment,route,network,counts,dl=0.,prio=0):
        for n in counts:
            for c in range(CASES):
                j=dict(group=group,treatment=treatment,route=route,network=network,downlink_seconds=dl,num_agents=n,case=c,
                       priority=(prio, c>=5, c))
                j['job_id']=jid(j);jobs.append(j)
    for p,t in enumerate(('A4','A2','A3','A1')):add('nominal',t,'allocated','physical',COUNTS,prio=p)
    for p,t in enumerate(('A4','A3')):add('baseline',t,'allocated','ideal',COUNTS,prio=4+p)
    for p,t in enumerate(('A4','A2')):
        for dl in (.15,.3):add('delay',t,'allocated','physical',(8,16,32),dl,prio=6+p)
    for p,r in enumerate(('edge','cloud')):add('routes','A4',r,'physical',COUNTS,prio=8+p)
    for p,t in enumerate(('A3','A1')):
        for dl in (.15,.3):add('delay',t,'allocated','physical',(8,16,32),dl,prio=10+p)
    return sorted(jobs,key=lambda j:j['priority'])

def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):return int(line.split()[1])/1e6
    return 99

def main():
    jobs=matrix();(STUDY/'runs/matrix.json').write_text(json.dumps(jobs,indent=0))
    skip_large=os.environ.get('SKIP_32_PRIORITY_GE')  # trimming hook: skip 32-agent jobs with priority >= value
    pending=[j for j in jobs if not (OUT/j['job_id']/'result.json').exists() and not (OUT/j['job_id']/'error.txt').exists()]
    if skip_large is not None:pending=[j for j in pending if not (j['num_agents']==32 and j['priority'][0]>=int(skip_large))]
    running={};log=open(STUDY/'runs/dispatch.log','a')
    env=dict(os.environ,PYTHONPATH=str(STUDY/'repo/src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.time()
    while pending or running:
        for p,(j,t0,fh) in list(running.items()):
            if p.poll() is not None:
                fh.close();del running[p]
                ok=(OUT/j['job_id']/'result.json').exists()
                if not ok:(OUT/j['job_id']).mkdir(parents=True,exist_ok=True);(OUT/j['job_id']/'error.txt').write_text(f'exit {p.returncode}; see job.log')
                log.write(json.dumps(dict(t=time.time(),job=j['job_id'],ok=ok,wall=time.time()-t0))+'\n');log.flush()
        large=sum(j['num_agents']==32 for j,_,_ in running.values());mid=sum(j['num_agents']==16 for j,_,_ in running.values())
        launched=False
        if len(running)<MAX_WORKERS and avail_gb()>MIN_AVAIL_GB:
            pick=None
            for j in pending:
                if j['num_agents']==32 and large>=MAX_LARGE:continue
                if j['num_agents']==16 and mid>=MAX_16:continue
                if j['num_agents']==32 or pick is None:pick=j
                if j['num_agents']==32:break
            if pick is not None:
                pending.remove(pick);folder=OUT/pick['job_id'];folder.mkdir(parents=True,exist_ok=True)
                (folder/'job.json').write_text(json.dumps(pick))
                fh=open(folder/'job.log','w')
                args=[sys.executable,str(STUDY/'harness/run_job.py'),str(folder/'job.json'),'--out',str(OUT)]
                if pick['num_agents']<=8 and pick['case']==0:args.append('--keep-trace')
                p=subprocess.Popen(args,stdout=fh,stderr=subprocess.STDOUT,env=env,cwd=str(STUDY))
                running[p]=(pick,time.time(),fh);launched=True
        done=sum((OUT/j['job_id']/'result.json').exists() for j in jobs)
        (STUDY/'runs/status.json').write_text(json.dumps(dict(total=len(jobs),done=done,running=[j['job_id'] for j,_,_ in running.values()],
            pending=len(pending),elapsed=time.time()-started,updated=time.time())))
        if not launched:time.sleep(2)
    log.close()

if __name__=='__main__':main()
