"""Empirical CDF/CCDF identities and published-table reconciliation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from panama_uav.latency_plotting import empirical_steps


def test_ties_and_inclusive_cutoff():
    x,y=empirical_steps([.1,.25,.25,.3])
    assert x.tolist()==[0.,.1,.25,.3]
    assert y.tolist()==[0.,.25,.75,1.]
    assert empirical_steps([.1,.25,.25,.3],complementary=True)[1].tolist()==[1.,.75,.25,0.]


def test_empty_has_no_fabricated_data():
    x,y=empirical_steps([])
    assert len(x)==len(y)==0


@pytest.mark.parametrize('value',[-1.,float('inf'),float('nan')])
def test_invalid_latency_rejected(value):
    with pytest.raises(ValueError):empirical_steps([value])


def test_published_w1_counts_and_statistics():
    path=Path(__file__).resolve().parents[1]/'scripts/plot_omega_latency.py'
    spec=importlib.util.spec_from_file_location('omega_latency',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    groups,rows=module.collect(path.parent.parent/'results/omega_sensitivity')
    assert len(rows)==15
    assert all(r['delivered_samples']==len(groups[r['num_agents'],r['route']]) for r in rows)
    assert all(r['on_time']+r['late']+r['undelivered']==r['task_observations'] for r in rows)
