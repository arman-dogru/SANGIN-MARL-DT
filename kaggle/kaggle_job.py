#!/usr/bin/env python3
"""Helper for SANGIN-MARL-DT Kaggle CPU jobs. Only touches kernels named rs-sangin-marl-dt[-<suffix>].

  build  --suffix 2 --name prio-n32 --jobs ID[,ID...] | --cmd "..."  [--wall-hours 11] [--parallel 1] [--ignore-trim]
         -> writes /workspace/kaggle/build/<slug>/{run.py,kernel-metadata.json,job.json}; pushes nothing
  push   --suffix 2 --approved      -> CPU-only push (refuses if 2 of our kernels are already running/queued)
  status                            -> status of our kernels
  pull   --suffix 2                 -> outputs into /workspace/kaggle_runs/<slug>-<timestamp>/

Auth: KAGGLE_API_TOKEN (taken from KAGGLE_KEY if unset); the key is never printed.
Pushing requires --approved: every run needs Arman's approval first.
"""
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

USER = 'armandogru'; BASE = 'rs-sangin-marl-dt'; TITLE = 'rs-SANGIN-MARL-DT'
HERE = Path('/workspace/kaggle'); TEMPLATE = HERE / 'SANGIN-MARL-DT/run_template.py'
KAGGLE = str(HERE / 'venv/bin/kaggle'); MAX_RUNNING = 2
STUDY = Path('/workspace/leo_study')
MATRICES = [(STUDY / 'runs/matrix.json', 'run_job.py'), (STUDY / 'runs/matrix_extra.json', 'run_job.py'),
            (STUDY / 'runs/prio/matrix_phase1.json', 'run_job_prio.py'), (STUDY / 'runs/prio/matrix_phase2.json', 'run_job_prio.py')]
SMOKE = dict(group='smoke', condition='nominal', treatment='A4', route='allocated', network='physical',
             downlink_seconds=0.0, num_agents=2, case=0, job_id='smoke__A4__allocated__physical__dl0__n2__c00')

def slug(suffix): return BASE + (f'-{suffix}' if suffix else '')
def title(suffix): return TITLE + (f'-{suffix}' if suffix else '')
def env():
    e = dict(os.environ)
    if not e.get('KAGGLE_API_TOKEN') and e.get('KAGGLE_KEY'): e['KAGGLE_API_TOKEN'] = e['KAGGLE_KEY']
    return e
def kaggle(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env())
    if check and r.returncode: sys.exit(f'kaggle {args[0]} {args[1]} failed: {r.stderr.strip() or r.stdout.strip()}')
    return r.stdout

def lookup(job_id):
    if job_id == SMOKE['job_id']: return SMOKE, 'run_job.py'
    for path, runner in MATRICES:
        if path.exists():
            m = json.loads(path.read_text()); m = m if isinstance(m, list) else m.get('jobs', [])
            for j in m:
                if j['job_id'] == job_id: return j, runner
    sys.exit(f'unknown job id {job_id}')

def ours():
    out = kaggle('kernels', 'list', '--mine', '--page-size', '100', '--search', BASE, check=False)
    return [l.split()[0] for l in out.splitlines() if l.startswith(f'{USER}/{BASE}')]

def status_of(ref):
    out = kaggle('kernels', 'status', ref, check=False)
    return out.strip().split('status')[-1].strip(' "').replace('KernelWorkerStatus.', '') if out else 'unknown'

def build(a):
    s = slug(a.suffix); d = HERE / 'build' / s; d.mkdir(parents=True, exist_ok=True)
    cmds = list(a.cmd or [])
    for jid in (a.jobs.split(',') if a.jobs else []):
        job, runner = lookup(jid.strip())
        cmds.append(f'{{py}} leo_study/harness/{runner} {shlex.quote(json.dumps(job))}')
    if not cmds: sys.exit('nothing to run: give --jobs or --cmd')
    cfg = dict(name=a.name, ref=a.ref, wall_hours=min(a.wall_hours, 11.5), parallel=a.parallel,
               ignore_trim=a.ignore_trim, cpu_torch=True, extras='leo,optimization', commands=cmds)
    (d / 'job.json').write_text(json.dumps(cfg, indent=1))
    (d / 'run.py').write_text(TEMPLATE.read_text().replace('__JOB_CONFIG__', json.dumps(cfg)))
    meta = json.loads((HERE / 'SANGIN-MARL-DT/kernel-metadata.json').read_text())
    meta.update(id=f'{USER}/{s}', title=title(a.suffix), code_file='run.py', enable_gpu=False, enable_tpu=False,
                enable_internet=True, is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    print(f'built {d} ({len(cmds)} commands, wall {cfg["wall_hours"]} h, parallel {a.parallel}); not pushed')

def push(a):
    if not a.approved: sys.exit('refusing: every Kaggle run needs Arman\'s approval; rerun with --approved once he has said yes')
    s = slug(a.suffix); d = HERE / 'build' / s
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    assert meta['id'] == f'{USER}/{s}' and s.startswith(BASE), 'slug outside rs-sangin-marl-dt*'
    assert meta['enable_gpu'] is False and meta['enable_tpu'] is False and 'machine_shape' not in meta, 'CPU only'
    busy = [r for r in ours() if status_of(r).lower() in ('running', 'queued')]
    if f'{USER}/{s}' in busy: sys.exit(f'{s} is already running; pushing would replace it')
    if len(busy) >= MAX_RUNNING: sys.exit(f'{len(busy)} of our CPU sessions already running: {busy}')
    print(kaggle('kernels', 'push', '-p', str(d)))

def status(a):
    for r in ours(): print(r, status_of(r))

def pull(a):
    s = slug(a.suffix); dest = Path(a.dest or f'/workspace/kaggle_runs/{s}-{time.strftime("%Y%m%d-%H%M%S")}')
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle('kernels', 'output', f'{USER}/{s}', '-p', str(dest)))
    print(f'outputs in {dest}; merge result.json folders into /workspace/leo_study/runs by hand (nothing is overwritten here)')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd_name', required=True)
    b = sp.add_parser('build'); b.add_argument('--suffix', default=''); b.add_argument('--name', required=True)
    b.add_argument('--jobs'); b.add_argument('--cmd', action='append'); b.add_argument('--ref', default='main')
    b.add_argument('--wall-hours', type=float, default=11.0); b.add_argument('--parallel', type=int, default=1)
    b.add_argument('--ignore-trim', action='store_true')
    q = sp.add_parser('push'); q.add_argument('--suffix', default=''); q.add_argument('--approved', action='store_true')
    sp.add_parser('status')
    o = sp.add_parser('pull'); o.add_argument('--suffix', default=''); o.add_argument('--dest')
    a = p.parse_args(); dict(build=build, push=push, status=status, pull=pull)[a.cmd_name](a)
