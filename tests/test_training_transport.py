import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from panama_uav.config import LearningCondition, load_config
from panama_uav.training import TrainingSession, compare_training
from panama_uav.transport import Message, encode

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/base.toml").profile("smoke")


def session(condition):
    torch.set_num_threads(1)
    s = TrainingSession(config(), condition, 42, "test-study", torch.device("cpu"))
    s.transport.windows = {a: [(0, 100)] for a in range(2)}
    return s


@pytest.mark.parametrize("condition", list(LearningCondition))
def test_learning_counts_versions_and_motion(condition):
    s = session(condition)
    before = [a.position for a in s.environment.agents]
    for _ in range(20):
        s.step()
    assert s.generated_steps == s.learning_opportunities == 20
    assert s.model_publications == 1
    assert [a.position for a in s.environment.agents] != before
    uplink = condition in (LearningCondition.UPLINK, LearningCondition.COMBINED)
    downlink = condition in (LearningCondition.DOWNLINK, LearningCondition.COMBINED)
    assert s.delivered_experiences == (
        30 if uplink else 40
    )  # controlled default delay is five steps
    assert s.learner_updates == (12 if uplink else 17)
    assert list(s.deployed_versions.values()) == ([0, 0] if downlink else [1, 1])
    assert s.environment.communication.ages()[0] == {
        "task": 0,
        "intentions": 0,
        "priorities": 0,
    }


def test_matched_fresh_initializations():
    hashes = [
        session(condition).initial_weights_sha256 for condition in LearningCondition
    ]
    assert len(set(hashes)) == 1


def test_no_undelivered_experience_or_partial_snapshot():
    s = session(LearningCondition.COMBINED)
    s.transport.windows = {a: [(0, 100)] for a in range(2)}
    s.transport.config = replace(s.transport.config, delay_steps=20)
    for _ in range(18):
        s.step()
    assert (
        len(s.replay) == 0 and s.delivered_experiences == 0 and s.learner_updates == 0
    )
    assert s.deployed_versions == {0: 0, 1: 0}
    assert (
        s.transport.counters["messages_generated"]
        > s.transport.counters["messages_delivered"]
    )
    initial = [a.position for a in s.environment.agents]
    for _ in range(20):
        s.step()
    assert s.generated_steps == 38 and s.deployed_versions[0] >= 1
    assert [a.position for a in s.environment.agents] != initial


def test_late_model_never_rolls_back():
    s = session(LearningCondition.DOWNLINK)
    weights = {k: v.detach().cpu().numpy() for k, v in s.policy.state_dict().items()}
    wire = encode({"weights": weights, "stage": 0})
    s.receive(Message(1, 0, "model", 1, 5, wire, ()))
    s.receive(Message(2, 0, "model", 0, 4, wire, ()))
    assert s.deployed_versions[0] == 5
    for k, v in s.actor_models[5].state_dict().items():
        assert torch.equal(v, s.policy.state_dict()[k])


@pytest.mark.parametrize("physical", [False, True])
def test_resume_exact_pending_transport_and_learning(physical):
    original = session(LearningCondition.COMBINED)
    if physical:
        original.transport.config = replace(
            original.transport.config, profile="physical"
        )
    for _ in range(18):
        original.step()
    stream = io.BytesIO()
    torch.save(original.checkpoint(), stream)
    expected = [original.step()[0] for _ in range(8)]
    expected_weights = {
        k: v.detach().clone() for k, v in original.policy.state_dict().items()
    }
    expected_tree = original.replay.tree.tree.copy()
    stream.seek(0)
    restored = TrainingSession.restore(
        torch.load(stream, weights_only=False),
        config(),
        LearningCondition.COMBINED,
        42,
        "test-study",
        torch.device("cpu"),
    )
    actual = [restored.step()[0] for _ in range(8)]
    assert actual == expected
    for key, value in restored.policy.state_dict().items():
        assert torch.equal(value, expected_weights[key])
    np.testing.assert_array_equal(restored.replay.tree.tree, expected_tree)
    assert restored.transport.events == original.transport.events


def test_budget_exhaustion_partial_and_no_full_comparison(tmp_path):
    c = config()
    c = replace(c, training=replace(c.training, budget_seconds=0))
    outcomes = compare_training(c, 42, tmp_path / "run", device_name="cpu")
    assert len(outcomes) == 4 and all(o.status == "budget_exhausted" for o in outcomes)
    assert not (tmp_path / "run" / "inference").exists()
    assert (tmp_path / "run" / "clean" / "checkpoint.pt").exists()


def test_atomic_checkpoint_retries_transient_windows_lock(tmp_path, monkeypatch):
    from panama_uav import training

    original = training.os.replace
    calls = []

    def replace_after_lock(source, target):
        calls.append((source, target))
        if len(calls) < 3:
            raise PermissionError("Transient file lock")
        return original(source, target)

    sleeps = []
    monkeypatch.setattr(training.os, "replace", replace_after_lock)
    monkeypatch.setattr(training.time, "sleep", sleeps.append)
    path = tmp_path / "checkpoint.pt"
    training.atomic_save({"complete": True}, path)
    assert torch.load(path, weights_only=False) == {"complete": True}
    assert len(calls) == 3 and sleeps == [0.02, 0.04]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA restore regression")
def test_cuda_checkpoint_map_location_preserves_rng_and_continuation(monkeypatch, request):
    # Exact equality is meaningful only with deterministic CUDA kernels; this
    # affects the regression test, not the running experiment's settings.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    request.addfinalizer(lambda: torch.use_deterministic_algorithms(previous))
    torch.set_num_threads(1)
    device = torch.device("cuda")
    original = TrainingSession(config(), LearningCondition.COMBINED, 42, "cuda-resume", device)
    for _ in range(18):
        original.step()
    stream = io.BytesIO()
    torch.save(original.checkpoint(), stream)
    expected = [original.step()[0] for _ in range(4)]
    expected_weights = {key: value.clone() for key, value in original.policy.state_dict().items()}
    stream.seek(0)
    checkpoint = torch.load(stream, map_location=device, weights_only=False)
    assert all(state.device.type == "cuda" for state in checkpoint["cuda_rng"])
    restored = TrainingSession.restore(checkpoint, config(), LearningCondition.COMBINED,
                                       42, "cuda-resume", device)
    assert [restored.step()[0] for _ in range(4)] == expected
    for key, value in restored.policy.state_dict().items():
        assert torch.equal(value, expected_weights[key])
