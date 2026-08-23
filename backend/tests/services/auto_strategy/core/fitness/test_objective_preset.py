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


def test_evaluation_config_ga_selection_mode_defaults_to_fitness():
    config = GAConfig()
    assert config.evaluation_config.ga_selection_mode == "fitness"


def test_evaluation_config_ga_selection_mode_roundtrip():
    config = GAConfig(
        evaluation_config=EvaluationConfig(ga_selection_mode="walk_forward")
    )
    restored = GAConfig.from_dict(config.to_dict())
    assert restored.evaluation_config.ga_selection_mode == "walk_forward"


def test_build_coarse_disables_wfa_ga():
    config = GAConfig(
        evaluation_config=EvaluationConfig(ga_selection_mode="walk_forward")
    )
    coarse = build_coarse_ga_config(config)
    assert coarse.evaluation_config.ga_selection_mode == "fitness"
    assert coarse.evaluation_config.evaluation_mode == "single"
