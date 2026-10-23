from types import SimpleNamespace

from panama_uav.research_monitor import process_identity,recent_rows


def test_reused_pid_and_wrong_command_are_not_workers(monkeypatch):
    import panama_uav.research_monitor as module
    process=SimpleNamespace(create_time=lambda:100.)
    monkeypatch.setattr(module.psutil,'Process',lambda _:process)
    assert process_identity(dict(pid=7,creation_time=101.,command=['python','worker']))['reason']=='pid_reused'
    process.children=lambda recursive:[];process.cmdline=lambda:['python','unrelated']
    assert process_identity(dict(pid=7,creation_time=100.,command=['python','worker']))['reason']=='command_mismatch'


def test_venv_launcher_child_requires_matching_full_command(monkeypatch):
    import panama_uav.research_monitor as module
    child=SimpleNamespace(pid=8,create_time=lambda:100.1,cmdline=lambda:['system-python','-m','panama_uav.phase1_scale','--worker','run'],
        cpu_percent=lambda interval:50.,cpu_times=lambda:(1.,2.),memory_info=lambda:SimpleNamespace(rss=100),status=lambda:'running')
    launcher=SimpleNamespace(create_time=lambda:100.,children=lambda recursive:[child],cmdline=lambda:['venv-python','launcher'])
    monkeypatch.setattr(module.psutil,'Process',lambda _:launcher)
    result=process_identity(dict(pid=7,creation_time=100.,command=['venv-python','-m','panama_uav.phase1_scale','--worker','run']))
    assert result['verified'] and result['pid']==8 and result['cpu_seconds']==3.


def test_partial_tail_row_is_not_a_completed_log_record(tmp_path):
    path=tmp_path/'rows.jsonl';path.write_text('{"steps":1}\n{"steps":2}\n{"steps":')
    assert recent_rows(path,1)==[dict(steps=2)]
    assert recent_rows(tmp_path/'absent')==[]
