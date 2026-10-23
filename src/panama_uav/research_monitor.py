"""Read-only worker checks and compact operational history; never launches work."""
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import subprocess
import time

import psutil

from .records import append_jsonl,sha256


def read_snapshot(path):
    for attempt in range(3):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==2:raise
            time.sleep(.05)


def process_identity(record):
    """A reused PID or unrelated Windows child is not the training worker."""
    try:
        proc=psutil.Process(record['pid'])
        if abs(proc.create_time()-record['creation_time'])>=1:
            return dict(verified=False,reason='pid_reused')
        expected=record['command'][1:]
        candidates=[proc,*proc.children(recursive=True)]
        matching=[]
        for candidate in candidates:
            cmd=candidate.cmdline()
            if cmd[1:]==expected:matching.append(candidate)
        if not matching:return dict(verified=False,reason='command_mismatch')
        worker=matching[-1]
        return dict(verified=True,pid=worker.pid,creation_time=worker.create_time(),command=worker.cmdline(),
            cpu_percent=worker.cpu_percent(interval=.1),cpu_seconds=sum(worker.cpu_times()[:2]),
            rss_bytes=worker.memory_info().rss,status=worker.status())
    except psutil.NoSuchProcess:
        return dict(verified=False,reason='stopped')
    except psutil.AccessDenied:
        return dict(verified=False,reason='access_denied')


def recent_rows(path,limit=256):
    # Seek from the tail so hourly checks do not parse millions of learning rows.
    path=Path(path)
    if not path.exists():return []
    with path.open('rb') as stream:
        stream.seek(0,2);end=stream.tell();start=max(0,end-2_000_000);stream.seek(start)
        if start:stream.readline()
        raw=stream.read()
        if raw and not raw.endswith(b'\n'):raw=raw.rsplit(b'\n',1)[0] if b'\n' in raw else b''
        lines=raw.splitlines()[-limit:]
    return [json.loads(line) for line in lines if line]


def inspect(root,plan_path=None):
    root=Path(root);manifest=read_snapshot(root/'manifest.json')
    state=read_snapshot(root/'status.json')
    launch=read_snapshot(root/'launch.json')
    now=datetime.now(ZoneInfo('America/Toronto'))
    process=process_identity(launch)
    learning=recent_rows(root/'learning_steps.jsonl');episodes=recent_rows(root/'training.jsonl')
    validation=recent_rows(root/'validation.jsonl',1)
    previous=recent_rows(root/'monitor_history.jsonl',1)
    steps=state.get('generated_joint_steps',0);count=state.get('episodes',0)
    advanced=(not previous or steps>previous[-1].get('joint_steps',-1) or count>previous[-1].get('episodes',-1)
        or (validation and validation[-1].get('generated_joint_steps',0)>previous[-1].get('validation_at_steps',0)))
    stale_seconds=(now.timestamp()-(root/'learning_steps.jsonl').stat().st_mtime) if learning else None
    finished=state.get('status') in ('qualified','not_qualified','not_qualified_partial','ablation_complete')
    decision=('complete' if finished else 'error' if state.get('status')=='error' else
        'stopped_incomplete' if not process['verified'] else
        'stalled' if not advanced and stale_seconds is not None and stale_seconds>600 else 'continue')
    checkpoint=root/'checkpoint.pt';checkpoint_info=dict(exists=checkpoint.exists())
    if checkpoint.exists():
        import torch
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        valid=saved.get('manifest_sha256')==sha256(root/'manifest.json')
        checkpoint_info.update(valid=valid,bytes=checkpoint.stat().st_size,
            age_seconds=now.timestamp()-checkpoint.stat().st_mtime,
            joint_steps=saved['session']['session'].generated_steps,
            population_episodes=dict(saved['session']['session'].population_episodes))
        if not valid:decision='error'
        del saved
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):gpu=None
    loss=[r['loss'] for r in learning if r.get('loss') is not None]
    probes=[r for r in learning if 'q_finite' in r]
    if any(not math.isfinite(x) for x in loss) or any(not r['q_finite'] for r in probes):decision='numerical_failure'
    row=dict(time_local=now.isoformat(),run_id=root.name,decision=decision,process=process,
        joint_steps=steps,episodes=count,population_steps=state.get('population_steps',{}),
        population_episodes=state.get('population_episodes',{}),curriculum_stage=state.get('curriculum_stage'),
        runtime_seconds=state.get('wall_seconds'),remaining_budget_seconds=max(0,state['budget_seconds']-state['wall_seconds']),
        actual_progress_since_last_check=bool(advanced),learning_file_age_seconds=stale_seconds,checkpoint=checkpoint_info,
        gpu_device_snapshot=gpu,gpu_snapshot_scope='whole device, not worker-exclusive utilization',
        loss_mean_recent=sum(loss)/len(loss) if loss else None,
        latest_q_probe=probes[-1] if probes else None,
        replay_size=learning[-1].get('replay_size') if learning else None,
        epsilon=learning[-1].get('epsilon') if learning else None,
        recent_learning_rows=len(learning),active_waits_recent=sum(r.get('waiting_steps',0) for r in learning),
        blocked_moves_recent=sum(r.get('blocked_move_steps',0) for r in learning),
        recent_completed_episodes=len(episodes),repeated_state_failures_recent=sum(not r['success'] and r.get('repeated_state',False) for r in episodes),
        validation_at_steps=validation[-1].get('generated_joint_steps') if validation else 0,
        latest_validation=validation[-1] if validation else None,
        next_check_local=(now+timedelta(hours=1)).isoformat() if process['verified'] and not finished else None,
        next_action='inspect/review before any resume' if decision!='continue' else 'continue within frozen caps')
    append_jsonl(root/'monitor_history.jsonl',row)
    if plan_path:
        with Path(plan_path).open('a',encoding='utf-8') as stream:
            stream.write(f"\n- **{now.strftime('%Y-%m-%d %H:%M %Z')}** — `{root.name}`: {decision}; "
                f"{steps:,} joint steps, {count:,} completed episodes; by population {row['population_episodes']}. "
                f"Runtime {state['wall_seconds']/3600:.2f} h; remaining wall budget {row['remaining_budget_seconds']/3600:.2f} h. "
                f"Recent loss {row['loss_mean_recent']}; WAITs/blocked moves {row['active_waits_recent']}/{row['blocked_moves_recent']} "
                f"over {len(learning)} logged joint steps. Checkpoint valid: {checkpoint_info.get('valid')}. "
                f"Next scheduled check: {row['next_check_local'] or 'paused pending review'}. Raw evidence: `{root}/monitor_history.jsonl`.\n")
    return row


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path);args=parser.parse_args()
    result=inspect(args.root,args.plan)
    print(json.dumps({k:result[k] for k in ('decision','joint_steps','episodes','population_episodes','remaining_budget_seconds','next_check_local')}))
