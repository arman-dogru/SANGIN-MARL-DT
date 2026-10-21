from copy import deepcopy
from itertools import product

import pytest

from panama_uav.consistency import consistency_snapshot, verify_ordering_quantities
from panama_uav.information import InformationCache


VERTEX = [[(0, 0), (0, 1)], [(0, 2), (0, 1)]]
EDGE = [[(0, 0), (0, 1)], [(0, 1), (0, 0)]]


def fixture(physical_paths=VERTEX, cached_paths=None, physical_priorities=(1, 2),
            dt_priorities=(1, 2), cached_priorities=(3, 2)):
    cached_paths = physical_paths if cached_paths is None else cached_paths
    truth = {r: dict(position=path[0], path=list(path), priority=physical_priorities[r], completed=False)
             for r, path in enumerate(physical_paths)}
    def cache(peer, path, priority):
        value = InformationCache()
        value.accept(1., dict(position=path[0], path=list(path), priority=priority, completed=False), 2, {peer: 0.})
        return value
    states = {r: {p: cache(p, path, cached_priorities[p]) for p, path in enumerate(cached_paths)} for r in truth}
    dt = {r: cache(r, path, dt_priorities[r]) for r, path in enumerate(physical_paths)}
    return {r: value["position"] for r, value in truth.items()}, truth, states, dt


def snapshot(inputs, **parameters):
    positions, truth, states, dt = inputs
    return consistency_snapshot(positions, truth, states, 2., radius=2, horizon=1, dt_states=dt, **parameters)


@pytest.mark.parametrize("physical,twin,cached", list(product((0, 1), repeat=3)))
def test_all_six_ordering_quantities_from_independent_binary_truth_table(physical, twin, cached):
    values = fixture(physical_priorities=(1, 0 if physical else 2),
        dt_priorities=(1, 0 if twin else 2), cached_priorities=(1, 0 if cached else 2))
    result = snapshot(values)
    pair = result["pairs"][0]
    keys = ("current_order", "digital_twin_order", "cached_order", "uplink_ordering_error",
            "downlink_ordering_error", "e2e_ordering_error")
    assert tuple(pair[k] for k in keys) == (physical, twin, cached, physical ^ twin, twin ^ cached, physical ^ cached)
    assert result["ordering_pairs_verified"] == 2 and pair["ordering_equations_verified"]


@pytest.mark.parametrize("quantity", ("current_order", "digital_twin_order", "cached_order",
    "uplink_ordering_error", "downlink_ordering_error", "e2e_ordering_error"))
def test_verifier_rejects_each_incorrect_quantity(quantity):
    pair = snapshot(fixture())["pairs"][0]
    pair[quantity] = 1-pair[quantity]
    with pytest.raises(ValueError, match="equation mismatch"):
        verify_ordering_quantities(pair)


def test_vertex_only_has_its_own_normalizer_and_the_same_source_aoi_weight():
    paths = [EDGE[0], EDGE[1], [(1, 1), (0, 1)]]
    values = fixture(paths, physical_priorities=(1, 2, 3), dt_priorities=(1, 2, 3), cached_priorities=(1, 2, 0))
    result = snapshot(values)
    assert result["relevant_pairs"] == 4 and result["vertex_relevant_pairs"] == 2
    assert result["fov_errors"] == result["vertex_e2e_errors"] == 2
    assert result["xi_fov"] == .5 and result["xi_vertex_e2e"] == 1
    assert result["xi_aoi"] == 1 and result["xi_vertex_aoi"] == 2
    assert result["vertex_aoi_error_sum"] == 4


def test_edge_only_retains_original_error_and_zero_vertex_denominator():
    result = snapshot(fixture(EDGE))
    assert result["xi_fov"] == 1 and result["edge_relevant_pairs"] == 2
    assert result["xi_vertex_e2e"] == result["xi_vertex_aoi"] == result["vertex_relevant_pairs"] == 0
    assert result["zero_vertex_relevant_pair_case"]


@pytest.mark.parametrize("physical,cached,missed,false_positive", [(VERTEX, EDGE, 2, 0), (EDGE, VERTEX, 0, 2)])
def test_vertex_misses_and_false_positives_are_separate_from_union_relevance(physical, cached, missed, false_positive):
    result = snapshot(fixture(physical, cached))
    assert result["missed_vertex_conflicts"] == result["missed_vertex_conflict_step_count"] == missed
    assert result["false_positive_vertex_conflicts"] == result["false_positive_vertex_conflict_step_count"] == false_positive
    assert result["missed_physical_conflicts"] == result["false_positive_conflicts"] == 0
    pair = result["pairs"][0]
    assert pair["physical_vertex_conflict_steps"] == ([1] if missed else [])
    assert pair["vertex_conflict_steps"] == ([1] if false_positive else [])


def test_vertex_events_at_same_step_but_wrong_cell_are_missed_and_false_predictions():
    stale = [[(0, 0), (1, 1)], [(0, 2), (1, 1)]]
    result = snapshot(fixture(VERTEX, stale))
    assert result["missed_vertex_conflicts"] == result["false_positive_vertex_conflicts"] == 0
    assert result["missed_vertex_conflict_step_count"] == result["false_positive_vertex_conflict_step_count"] == 2


def test_square_fov_boundary_unknown_physical_reference_and_empty_pairs():
    values = fixture([[(0, 0), (1, 1)], [(0, 2), (1, 1)]])
    positions, truth, states, dt = values
    boundary = consistency_snapshot(positions, truth, states, 2., radius=1, horizon=1, dt_states=dt)
    assert boundary["vertex_relevant_pairs"] == 2  # Diagonal corners belong to the square FoV.
    outside = consistency_snapshot(positions, truth, states, 2., radius=0, horizon=1, dt_states=dt)
    assert outside["vertex_relevant_pairs"] == 0 and outside["zero_vertex_relevant_pair_case"]
    unknown = consistency_snapshot(positions, {r: dict(priority=v["priority"]) for r, v in truth.items()},
        states, 2., radius=1, horizon=1, dt_states=dt)
    assert unknown["unknown_physical_pairs"] == 2
    assert all(p["missed_vertex_conflict"] is None and p["false_positive_vertex_conflict"] is None for p in unknown["pairs"])
    empty = consistency_snapshot({}, {}, {}, 2.)
    assert empty["xi_vertex_e2e"] == empty["xi_vertex_aoi"] == empty["ordering_pairs_verified"] == 0


@pytest.mark.parametrize("physical,cached", [(VERTEX, VERTEX), (EDGE, EDGE), (VERTEX, EDGE), (EDGE, VERTEX)])
def test_original_metrics_and_six_quantities_are_unchanged_and_inputs_remain_read_only(physical, cached):
    values = fixture(physical, cached)
    before = deepcopy(values)
    legacy, extended = snapshot(values, schema_version=1), snapshot(values)
    for key, value in legacy.items():
        if key == "pairs":
            assert [{k: p[k] for k in old} for p, old in zip(extended[key], value)] == value
        else:
            assert extended[key] == value
    assert values == before
    extended["pairs"][0]["raw_intentions"]["cached_peer"].clear()
    assert values == before


def test_raw_intentions_and_source_generation_arrival_timestamps_are_retained():
    positions, truth, states, dt = fixture()
    metadata = {r: {p: dict(arrived_at=1.5) for p in truth} for r in truth}
    result = consistency_snapshot(positions, truth, states, 2., dt_states=dt, cache_metadata=metadata)
    pair = result["pairs"][0]
    assert pair["raw_intentions"]["physical_peer"] == truth[1]["path"]
    assert pair["raw_intentions"]["digital_twin_peer"] == dt[1].value["path"]
    assert pair["raw_intentions"]["cached_peer"] == states[0][1].value["path"]
    assert pair["digital_twin_peer_timestamps"] == dict(source_time=0, generation_time=1, version=2)
    assert (pair["peer_source_time"], pair["peer_generation_time"], pair["peer_arrival_time"], pair["evaluated_at"]) == (0, 1, 1.5, 2)
