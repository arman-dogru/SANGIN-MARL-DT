from pathlib import Path

import pytest
import torch

from panama_uav.config import LearningCondition, load_config
from panama_uav.evaluation import evaluate_suite
from panama_uav.records import read_jsonl, sha256
from panama_uav.training import TrainingSession, atomic_save, model_payload

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_resume_preserves_paired_cases_and_does_not_duplicate(tmp_path):
    config = load_config(ROOT / "configs/base.toml").profile("smoke")
    torch.set_num_threads(1)
    session = TrainingSession(
        config, LearningCondition.CLEAN, 42, "eval-test", torch.device("cpu")
    )
    model = tmp_path / "model.pt"
    atomic_save(model_payload(session), model)
    output = tmp_path / "evaluation"
    kwargs = {
        "device_name": "cpu",
        "cases": 1,
        "counts": (2,),
        "diagnostic": True,
        "selected": ("perfect",),
    }
    first = evaluate_suite(config, model, output, **kwargs)
    bank_hash = sha256(output / "held_out_cases.json")
    second = evaluate_suite(config, model, output, **kwargs)
    assert first == second
    assert len(read_jsonl(output / "perfect/episodes.jsonl")) == 1
    with pytest.raises(ValueError, match="case bank differs"):
        evaluate_suite(config, model, output, **{**kwargs, "cases": 2})
    assert sha256(output / "held_out_cases.json") == bank_hash
    with pytest.raises(ValueError, match="qualified"):
        evaluate_suite(config, model, tmp_path / "full", cases=50)
