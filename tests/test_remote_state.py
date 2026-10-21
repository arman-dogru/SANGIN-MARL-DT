from dataclasses import replace

import numpy as np
import pytest

from panama_uav.remote_state import (
    ActionCommand, CommandInbox, DTReplica, DTState, PhysicalReport,
    observation_hash, remote_observation,
)


def state(peer=0, generated=0, source=0, version=0):
    return DTState(peer, (1, 1+peer), (1, 3), [(1, 1+peer), (1, 2)],
        2-peer, False, generated, version, {peer: source}, f"r{peer}")


def command(cycle=0, generation=.02, version=0, sources=None):
    return ActionCommand(0, cycle, 3, generation, cycle+.1, version,
        "cloud", sources or {0: 0}, "j0", "hash", {}, {})


def test_command_cutoff_ordering_and_no_replay():
    box = CommandInbox()
    assert box.accept(command(version=2), .03) == (True, "accepted")
    assert box.accept(command(version=1), .04) == (False, "rejected_order")
    assert box.decision(0, .1)[0].version == 2
    assert box.accept(command(generation=.03, version=3), .11) == (False, "rejected_obsolete")
    assert box.decision(1, 1.1) is None
    with pytest.raises(ValueError, match="closed"):
        box.decision(1, 1.1)


def test_source_staleness_and_cutoff_miss_are_independent():
    box = CommandInbox()
    stale = command(cycle=2, generation=2.02, sources={0: 2, 1: 0})
    assert box.accept(stale, 2.03)[0]
    assert box.decision(2, 2.1)[0].sources[1] == 0
    fresh_but_late = command(cycle=3, generation=3.02, sources={0: 3})
    assert not box.accept(fresh_but_late, 3.101)[0]
    assert box.decision(3, 3.1) is None


def test_replica_delivery_is_explicit_and_source_cannot_roll_back():
    drone, cloud = DTReplica("drone"), DTReplica("cloud")
    assert cloud.accept(state(generated=2, source=1), 2.1, "cloud-v1")
    assert not drone.states  # A cloud update is not shared memory.
    assert drone.accept(state(generated=2, source=1), 3, "sync-v1")
    assert not drone.accept(state(generated=4, source=.5), 4.1, "old-source")
    assert not drone.accept(state(generated=2, source=1), 4.1, "duplicate")
    with pytest.raises(ValueError, match="causality"):
        drone.accept(state(generated=5), 4, "future")


def test_remote_tensor_and_mixed_source_provenance():
    report = PhysicalReport(0, 2, 2, (1, 1), (1, 3), False, 2,
        np.zeros((3, 3)).tolist(), np.zeros((3, 3)).tolist(), "r0")
    replica = DTReplica("drone")
    own = state(generated=2, source=2)
    replica.accept(own, 2, "own")
    replica.accept(state(1, generated=1, source=.5), 1, "peer")
    obs, sources, versions, priorities = remote_observation(report, own, replica, 1, 5)
    assert obs["spatial"].shape == (3, 3, 3) and obs["vector"].shape == (12,)
    assert obs["spatial"][2, 1, 2] == 1
    assert min(sources.values()) == .5 and versions[1]["state_event"] == "peer"
    assert observation_hash(obs) == observation_hash({k: v.tolist() for k, v in obs.items()})
    assert priorities == {0: 2, 1: 1}


def test_all_dependencies_must_exist_before_generation_not_just_oldest():
    with pytest.raises(ValueError, match="sources"):
        command(generation=.02, sources={0: 0, 1: .03})
    with pytest.raises(ValueError, match="causality"):
        replace(state(), sources={0: 0, 1: 1})
