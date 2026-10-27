"""Recovery alters storage and dispatch only; admission keeps resource reserves."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_redesign_storage as recovery
import checkpoint_redesign as publication
from run_redesign_accelerated import write


def test_admission_preserves_ram_commit_and_size_limits():
    job=dict(num_agents=32)
    assert recovery.admissible(job,[],40,0,40)
    assert not recovery.admissible(job,[],27,0,40)
    assert not recovery.admissible(job,[],40,0,27)
    assert not recovery.admissible(job,[job]*4,200,0,200)
    assert not recovery.admissible(dict(num_agents=16),[dict(num_agents=16)]*8,200,0,200)
    assert recovery.admissible(dict(num_agents=8),[],27,0,27)


def test_dead_dispatcher_cannot_report_running_or_eta(tmp_path):
    write(tmp_path/'accelerated/launch.json',dict(pid=2147483647,creation_time=0,command=[]))
    assert publication.worker_state(tmp_path,dict(status='running'))=='stopped'
    write(tmp_path/'accelerated/resource_pause.json',dict(status='resource_paused'))
    assert publication.worker_state(tmp_path,dict(status='running'))=='resource_paused'


def test_storage_adapter_does_not_modify_frozen_source(tmp_path):
    import inspect
    import compact_telemetry
    text=inspect.getsource(compact_telemetry.install)
    assert 'DiskEvents' in text and 'experimental_class' in text
    assert 'write_text' not in text and 'write_bytes' not in text
