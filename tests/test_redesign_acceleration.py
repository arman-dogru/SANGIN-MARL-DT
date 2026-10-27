"""Dispatch-only ordering, immutable reuse and conditional publication cohorts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_redesign_accelerated as accelerated
import checkpoint_redesign as publication
import run_redesign as frozen_interface


def test_accelerated_order_preserves_all_registered_jobs():
    cases=[dict(case_id=f'agents-{n}-case-{i:03d}',case=i,num_agents=n) for n in (2,4,8,16,32) for i in range(10)]
    manifest=dict(jobs_A=frozen_interface.jobs_for(cases,'A'),jobs_B=frozen_interface.jobs_for(cases,'B'))
    phases=accelerated.phases(manifest)
    assert [(n,len(j)) for n,j in phases]==[('A4_initial',45),('ablations_initial',315),('phase_A_expansion',360),('phase_B_weights',660)]
    actual=[accelerated.key(j) for n,jobs in phases for j in jobs]
    expected={accelerated.key(j) for j in manifest['jobs_A']+manifest['jobs_B']}
    assert len(actual)==len(set(actual))==1380 and set(actual)==expected
    assert all(j['treatment']=='A4' and j['weight_id']=='W1' for j in phases[0][1])


def packet(tmp_path,success,latency=.1):
    import numpy as np
    folder=tmp_path/('success' if success else 'failure');folder.mkdir()
    job=dict(treatment='A4',weight_id='W1',route='allocated',num_agents=8,case_id='case-'+str(success))
    task=dict(job_id='0:0',deadline_status='on_time' if latency<=.25 else 'late',executed_x=1,
        delivered_at=latency,observed_latency_seconds=latency)
    value=dict(job=job,audit=dict(status='passed'),manifest_sha256='parent',
        row=dict(success=success,observed_seconds=1.,modeled_energy_lower_j=1. if success else 999.,
            makespan_seconds=1. if success else None,waiting_steps=0,drone_utilization_seconds=.1,cloud_utilization_seconds=0.,
            relevant_pairs=2,fov_errors=1,eligible_decisions=1,cached_fallback_decisions=0),tasks=[task])
    path=folder/'result.json';accelerated.write(path,value)
    accelerated.write(folder/'complete.json',dict(sha256=accelerated.digest(path),wall_seconds=2.,peak_rss_bytes=1234))
    return path


def test_failed_missions_retained_but_do_not_enter_conditional_metrics(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    rows=[publication.cached_episode(packet(tmp_path,True),cache,'parent'),
          publication.cached_episode(packet(tmp_path,False,.9),cache,'parent')]
    assert len(rows)==2 and sum(not r['success'] for r in rows)==1
    summary=publication.summaries(rows,cache)[0]
    assert summary['attempted']==2 and summary['successful']==1 and summary['failed']==1
    assert summary['e2e_p95_seconds']==pytest.approx(.1)
    assert summary['energy_j']==1. and summary['tasks']==1 and summary['xi_fov']==.5


def test_empty_successful_cohort_is_undefined(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    row=publication.cached_episode(packet(tmp_path,False,.9),cache,'parent')
    summary=publication.summaries([row],cache)[0]
    assert summary['successful']==0 and summary['failed']==1
    assert all(summary[name] is None for name in ('e2e_p95_seconds','energy_j','xi_fov','deadline_miss_rate','drone_cpu_utilization'))


def test_unverified_or_changed_packet_cannot_be_published(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();path=packet(tmp_path,True)
    path.write_text('{}')
    with pytest.raises(ValueError,match='Changed committed'):publication.cached_episode(path,cache,'parent')
