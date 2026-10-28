"""SANGIN-MARL-DT Kaggle CPU job runner (rs-sangin-marl-dt).

The helper (kaggle_job.py) replaces the JOB placeholder below with a JSON job config and pushes this
file as a CPU-only script kernel. Results go to /kaggle/working. CPU only: no GPU is requested, and
CUDA is hidden from every child process.
"""
import json, os, sys, time, shutil, subprocess, threading, signal
from pathlib import Path

JOB = json.loads(r'''__JOB_CONFIG__''')
START = time.time()
WALL = min(float(JOB.get('wall_hours', 11.0)), 11.5) * 3600   # own guard, below Kaggle's 12 h limit
MARGIN = 600                                                    # leave 10 min to write the manifest
W = Path('/kaggle/working'); W.mkdir(parents=True, exist_ok=True)
LOGS = W / 'logs'; LOGS.mkdir(exist_ok=True)
REPO = Path('/tmp/SANGIN-MARL-DT')
URL = JOB.get('repo_url', 'https://github.com/arman-dogru/SANGIN-MARL-DT.git')  # public repo: no token needed
manifest = dict(job=JOB, started_unix=START, commands=[], status='starting')

def save():
    manifest['elapsed_seconds'] = time.time() - START
    tmp = W / 'manifest.json.tmp'; tmp.write_text(json.dumps(manifest, indent=1)); os.replace(tmp, W / 'manifest.json')

def sh(cmd, log, timeout=None, env=None, cwd=None):
    with open(log, 'a') as f:
        f.write(f'$ {cmd}\n'); f.flush()
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); return 'timeout'

def remaining():
    return WALL - MARGIN - (time.time() - START)

# --- CPU-only check -------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = ''
manifest['gpu_visible'] = shutil.which('nvidia-smi') is not None and subprocess.run(
    'nvidia-smi -L', shell=True, capture_output=True).returncode == 0
manifest['cpu_count'] = os.cpu_count()
try:
    manifest['mem_total_gb'] = int(open('/proc/meminfo').readline().split()[1]) / 1048576
except Exception:
    pass
save()

# --- memory monitor --------------------------------------------------------------------------
def monitor():
    with open(W / 'mem.log', 'a') as f:
        while True:
            try:
                info = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
                f.write(f"{time.time()-START:.0f} avail_gb={info['MemAvailable']/1048576:.2f}\n"); f.flush()
            except Exception:
                pass
            time.sleep(60)
threading.Thread(target=monitor, daemon=True).start()

# --- clone and install -----------------------------------------------------------------------
setup = LOGS / '00_setup.log'
ref = JOB.get('ref', 'main')
rc = sh(f'git clone --depth 50 --branch {ref} {URL} {REPO}', setup, timeout=900)
if rc != 0:
    manifest['status'] = f'clone_failed:{rc}'; save(); sys.exit(1)
manifest['commit'] = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
py = sys.executable
if JOB.get('cpu_torch', True):
    sh(f'{py} -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu', setup, timeout=1800)
extras = JOB.get('extras', 'leo,optimization')
rc = sh(f'{py} -m pip install -q -e "{REPO}[{extras}]" --extra-index-url https://download.pytorch.org/whl/cpu', setup, timeout=1800)
if rc != 0:
    manifest['status'] = f'install_failed:{rc}'; save(); sys.exit(1)

# --- path shims so the box-local harness paths resolve unchanged -----------------------------
def link(dst, src):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink(): shutil.rmtree(dst)
        else: dst.unlink()
    dst.symlink_to(src)

STUDY = REPO / 'leo_study'
runs = W / 'runs'; runs.mkdir(exist_ok=True)
for f in (STUDY / 'runs').glob('*.json'):
    shutil.copy(f, runs / f.name)
if JOB.get('ignore_trim'):
    (runs / 'trim.json').write_text(json.dumps(dict(rules=[])))   # rerun cells the box budget trimmed
link(STUDY / 'runs', runs)                     # run_job default --out lands in /kaggle/working/runs/main
link(STUDY / 'repo', REPO)                     # harness expects leo_study/repo/{src,configs}
link(REPO / 'data/leo/tremblant_full.npz', STUDY / 'inputs/tremblant_full.npz')
link(REPO / '.venv/bin/python', py)
link('/workspace/SANGIN-dev', REPO)
link('/workspace/leo_study', STUDY)
link('/workspace/panama_models', STUDY / 'inputs')
env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
manifest['status'] = 'running'; save()

# --- run commands (sequential by default; heavy.lock still serialises N>=32 episodes) ---------
cmds = JOB.get('commands', [])
per_cap = JOB.get('per_command_seconds')
lock = threading.Lock()
queue = list(enumerate(cmds))
def worker():
    while True:
        with lock:
            if not queue: return
            i, cmd = queue.pop(0)
        rec = dict(index=i, command=cmd)
        left = remaining()
        if left <= 60:
            rec['status'] = 'skipped_wall_guard'
        else:
            t0 = time.time(); to = min(left, per_cap) if per_cap else left
            rc = sh(cmd.replace('{py}', py), LOGS / f'{i+1:02d}.log', timeout=to, env=env, cwd=str(REPO))
            rec.update(status='ok' if rc == 0 else ('timeout' if rc == 'timeout' else f'exit:{rc}'), seconds=time.time() - t0)
        with lock:
            manifest['commands'].append(rec); save()
threads = [threading.Thread(target=worker) for _ in range(max(1, int(JOB.get('parallel', 1))))]
for t in threads: t.start()
for t in threads: t.join()
manifest['status'] = 'done'; save()
