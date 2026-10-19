from pathlib import Path

import pytest
import torch

from panama_uav.config import ExperimentMode, load_config
from panama_uav.models import DQN
from panama_uav.training import load_model, qualifies

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_study_contract():
    c = load_config(ROOT / "configs/base.toml")
    assert list(ExperimentMode) == [ExperimentMode.UAV]
    assert tuple(s.num_agents for s in c.curriculum) == (2, 4, 6, 8)
    assert all(
        s.max_episode_steps == 250
        and s.min_episodes == 500
        and s.graduation_success_rate == 0.95
        for s in c.curriculum
    )
    assert (
        c.training.validation_episodes == 200
        and c.training.validation_interval_episodes == 250
    )
    assert c.training.model_broadcast_steps == 16 and c.training.budget_seconds == 10800
    assert not hasattr(c, "static_radio") and not hasattr(c.reward, "network_penalty")


def test_01_architecture():
    model = DQN((3, 15, 15), 12)
    assert model.cnn_base[0].in_channels == 3 and model.cnn_base[0].out_channels == 32
    assert model.head[0].out_features == 512
    assert model(
        {"spatial": torch.zeros(2, 3, 15, 15), "vector": torch.zeros(2, 12)}
    ).shape == (2, 5)


def test_old_checkpoint_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": DQN((3, 15, 15), 12).state_dict()}, path)
    with pytest.raises(ValueError, match="Legacy"):
        load_model(path, load_config(ROOT / "configs/base.toml"), torch.device("cpu"))


def test_no_false_graduation():
    stage = load_config(ROOT / "configs/base.toml").curriculum[0]
    good = {"episodes": 200, "complete": True, "success_rate": 0.95}
    assert qualifies(stage, 500, good)
    assert not qualifies(stage, 499, good)
    assert not qualifies(stage, 500, {**good, "success_rate": 0.945})
    assert not qualifies(stage, 500, {**good, "episodes": 199})
    assert not qualifies(stage, 500, {**good, "complete": False})
    assert not qualifies(stage, stage.max_episodes, {**good, "success_rate": 0})
