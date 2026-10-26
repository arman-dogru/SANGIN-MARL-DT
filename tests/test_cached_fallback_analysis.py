"""Paired effects retain all-attempt success and conditional exposure denominators."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyse_cached_fallback import interval


def test_success_and_energy_use_different_cohorts():
    new=[dict(case_id='a',success=True,modeled_energy_lower_j=3),dict(case_id='b',success=True,modeled_energy_lower_j=99)]
    old=[dict(case_id='a',success=True,modeled_energy_lower_j=1),dict(case_id='b',success=False,modeled_energy_lower_j=999)]
    success=interval(new,old,'mission_success_rate')
    assert success['paired_cases']==2 and success['difference']==.5
    energy=interval(new,old,'modeled_energy_j')
    assert energy['paired_cases']==1 and energy['difference']==2 and energy['single_pair_interval']


def test_zero_exposure_is_undefined_instead_of_a_zero_error_claim():
    rows=[dict(case_id='a',success=True,fov_errors=0,fov_relevant_pairs=0)]
    result=interval(rows,rows,'xi_fov')
    assert result['paired_cases']==1 and result['difference'] is None and result['ci95_low'] is None


def test_ratio_effect_pools_case_exposures():
    new=[dict(case_id='a',success=True,fov_errors=1,fov_relevant_pairs=2),
         dict(case_id='b',success=True,fov_errors=0,fov_relevant_pairs=8)]
    old=[dict(case_id='a',success=True,fov_errors=0,fov_relevant_pairs=2),
         dict(case_id='b',success=True,fov_errors=0,fov_relevant_pairs=8)]
    result=interval(new,old,'xi_fov')
    assert result['difference']==pytest.approx(.1) and result['actual_denominator']==10


def test_unpaired_or_duplicate_inputs_are_rejected():
    a=dict(case_id='a',success=True)
    with pytest.raises(ValueError,match='Identical'):
        interval([a],[dict(case_id='b',success=True)],'mission_success_rate')
    with pytest.raises(ValueError,match='Identical'):
        interval([a,a],[a],'mission_success_rate')
