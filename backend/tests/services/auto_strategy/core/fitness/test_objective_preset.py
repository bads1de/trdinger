import pytest

from app.services.auto_strategy.config.ga_config import EvaluationConfig, GAConfig
from app.services.auto_strategy.core.evaluation.evaluation_fidelity import (
    build_coarse_ga_config,
)


def test_objective_preset_excess_return_sets_single_objective():
    config = GAConfig(objective_preset="excess_return")
    assert config.objectives == ["excess_return"]
    assert config.objective_weights == [1.0]


def test_objective_preset_weighted_score_restores_default():
    config = GAConfig(
        objectives=["excess_return"],
        objective_weights=[1.0],
        objective_preset="weighted_score",
    )
    assert config.objectives == ["weighted_score"]
    assert config.objective_weights == [1.0]


def test_explicit_objectives_without_preset():
    config = GAConfig(objectives=["excess_return"], objective_weights=[1.0])
    assert config.objectives == ["excess_return"]


def test_evaluation_config_wfa_ga_defaults_false():
    config = GAConfig()
    assert config.evaluation_config.enable_walk_forward_for_ga is False


def test_evaluation_config_wfa_ga_roundtrip():
    config = GAConfig(
        evaluation_config=EvaluationConfig(enable_walk_forward_for_ga=True)
    )
    restored = GAConfig.from_dict(config.to_dict())
    assert restored.evaluation_config.enable_walk_forward_for_ga is True


def test_build_coarse_disables_wfa_ga():
    config = GAConfig(
        evaluation_config=EvaluationConfig(enable_walk_forward_for_ga=True)
    )
    coarse = build_coarse_ga_config(config)
    assert coarse.evaluation_config.enable_walk_forward_for_ga is False
    assert coarse.evaluation_config.enable_walk_forward is False
