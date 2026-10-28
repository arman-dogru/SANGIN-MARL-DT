"""Priority add-on dispatcher. Phase 1: N=8/16, routes allocated/cloud/edge (240 jobs, <=2 concurrent).
Phase 2: N=32, routes allocated/cloud only (80 jobs, 1 at a time, 20-min wall cap -> timeout.txt, no retry).
Usage: python harness/dispatch_prio.py --phase 1|2      (resumable: skips jobs with result.json/timeout.txt/error.txt)
Each job starts only when MemAvailable > 4 GB. Status runs/prio/status_phase<k>.json, log runs/prio/dispatch_phase<k>.log.
"""
import json, os, sys, time, subprocess, argparse
from pathlib import Path
STUDY = Path('/workspace/leo_study'); OUT = STUDY / 'runs/prio'; PY = '/workspace/SANGIN-dev/.venv/bin/python'
MIN_AVAIL_GB = 4.0

def matrix(phase):
    jobs = []
    counts, routes = ((16, 8), ('allocated', 'cloud', 'edge')) if phase == 1 else ((32,), ('allocated', 'cloud'))
    for n in counts:
        for tr in ('A4', 'A3'):
            for dl in (0.0, 0.15):
                for route in routes:
                    for c in range(10):
                        cond = 'nominal' if dl == 0 else 'delay'
                        jid = f"prio__{cond}__{tr}__{route}__physical__dl{int(round(dl*1000))}__n{n}__c{c:02d}"
                        jobs.append(dict(group='prio', condition=cond, treatment=tr, route=route, network='physical',
                                         downlink_seconds=dl, num_agents=n, case=c, job_id=jid))
    return jobs

def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'): return int(line.split()[1]) / 1048576
    return 0.0

def main():
    a = argparse.ArgumentParser(); a.add_argument('--phase', type=int, required=True); args = a.parse_args()
    phase = args.phase; workers = 2 if phase == 1 else 1; cap = None if phase == 1 else 1200
    jobs = matrix(phase); json.dump(jobs, open(OUT / f'matrix_phase{phase}.json', 'w'), indent=1)
    log = open(OUT / f'dispatch_phase{phase}.log', 'a'); env = dict(os.environ, PYTHONPATH=str(STUDY / 'repo/src'), OMP_NUM_THREADS='1')
    def finished(j):
        d = OUT / j['job_id']; return any((d / f).exists() for f in ('result.json', 'timeout.txt', 'error.txt'))
    pending = [j for j in jobs if not finished(j)]; running = {}; t_start = time.time()
    def status():
        tmp = OUT / f'status_phase{phase}.json.tmp'
        tmp.write_text(json.dumps(dict(phase=phase, total=len(jobs), done=sum(finished(j) for j in jobs), running=list(running),
                                       pending=len(pending), elapsed=time.time() - t_start, updated=time.time())))
        os.replace(tmp, OUT / f'status_phase{phase}.json')
    while pending or running:
        for jid, (p, j, t0) in list(running.items()):
            rc = p.poll(); d = OUT / jid
            if rc is None and cap and time.time() - t0 > cap:
                p.kill(); p.wait(); (d / 'timeout.txt').write_text(f'wall cap {cap}s exceeded; not retried'); rc = 'timeout'
            if rc is None: continue
            if rc != 'timeout' and not (d / 'result.json').exists(): (d / 'error.txt').write_text(f'exit code {rc}; see job.log')
            log.write(json.dumps(dict(t=time.time(), job=jid, rc=rc, ok=(d / 'result.json').exists(), wall=time.time() - t0)) + '\n'); log.flush()
            del running[jid]
        while pending and len(running) < workers and avail_gb() > MIN_AVAIL_GB:
            j = pending.pop(0); d = OUT / j['job_id']; d.mkdir(parents=True, exist_ok=True)
            (d / 'job.json').write_text(json.dumps(j, indent=1))
            p = subprocess.Popen(['nice', '-n', '19', PY, str(STUDY / 'harness/run_job_prio.py'), str(d / 'job.json')],
                                 stdout=open(d / 'job.log', 'w'), stderr=subprocess.STDOUT, env=env, cwd=str(STUDY))
            running[j['job_id']] = (p, j, time.time())
        status(); time.sleep(3)
    status(); log.write(json.dumps(dict(t=time.time(), event='phase_done')) + '\n'); log.close()

if __name__ == '__main__':
    main()
