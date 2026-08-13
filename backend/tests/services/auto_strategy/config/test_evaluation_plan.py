from types import SimpleNamespace

import pytest

from app.services.auto_strategy.config import EvaluationPlan, GAConfig


def test_legacy_config_maps_validation_and_robustness_layers():
    config = GAConfig(
        enable_purged_kfold=True,
        evaluation_config={
            "oos_split_ratio": 0.25,
            "oos_fitness_weight": 0.7,
            "enable_walk_forward": True,
            "wfa_n_folds": 3,
            "wfa_train_ratio": 0.6,
            "wfa_anchored": True,
        },
        validation_config={
            "enabled": True,
            "wfa_n_folds": 4,
            "wfa_train_ratio": 0.75,
            "wfa_anchored": False,
            "max_candidates": 7,
        },
        robustness_config={
            "validation_symbols": ["ETH/USDT:USDT"],
            "stress_slippage": [0.0002],
            "stress_commission_multipliers": [1.5],
            "aggregate_method": "mean",
        },
    )

    plan = EvaluationPlan.from_legacy_config(config)

    assert plan.selection.method == "is"
    assert plan.selection.holdout_ratio == 0.25
    assert plan.validation.method == "rolling_holdout"
    assert plan.validation.folds == 4
    assert plan.validation.train_ratio == 0.75
    assert plan.robustness.enabled is True
    assert {scenario["type"] for scenario in plan.robustness.scenarios} == {
        "symbol",
        "slippage",
        "commission",
    }
    assert any("複数の評価方式" in warning for warning in plan.warnings)
    assert any("PurgedKFold" in warning for warning in plan.warnings)
    assert any("oos_fitness_weight" in warning for warning in plan.warnings)


def test_evaluation_plan_round_trip_is_stable():
    plan = EvaluationPlan.from_dict(
        {
            "schema_version": "1",
            "dataset_id": "dataset",
            "is_period": {"start_date": "2024-01-01", "end_date": "2024-06-01"},
            "validation": {
                "enabled": True,
                "method": "rolling_holdout",
                "folds": 3,
                "train_ratio": 0.7,
            },
            "robustness": {
                "enabled": True,
                "scenarios": [{"type": "slippage", "delta": 0.001}],
                "policy": "gate_only",
                "failure_policy": "fail_closed",
            },
        }
    )

    restored = EvaluationPlan.from_dict(plan.to_dict())

    assert restored.to_dict() == plan.to_dict()


@pytest.mark.parametrize(
    "updates",
    [
        {"selection": {"method": "validation"}},
        {"validation": {"method": "unsupported"}},
        {"validation": {"train_ratio": 1.1}},
        {"robustness": {"policy": "rank_all"}},
        {"robustness": {"failure_policy": "fallback"}},
    ],
)
def test_evaluation_plan_rejects_ambiguous_or_unsafe_modes(updates):
    with pytest.raises(ValueError):
        EvaluationPlan.from_dict(updates)


def test_legacy_config_without_validation_uses_oos_as_compatibility_method():
    config = SimpleNamespace(
        evaluation_config=SimpleNamespace(
            oos_split_ratio=0.2,
            oos_fitness_weight=0.5,
            enable_walk_forward=False,
            enable_parallel=True,
            max_workers=2,
            timeout=30,
        ),
        validation_config=SimpleNamespace(enabled=False),
        robustness_config=SimpleNamespace(),
        enable_purged_kfold=False,
    )

    plan = EvaluationPlan.from_legacy_config(config)

    assert plan.validation.enabled is False
    assert plan.validation.method == "oos"
    assert plan.selection.holdout_ratio == 0.2


def test_apply_to_ga_config_syncs_enabled_and_scenarios():
    """計画のrobustness設定がGAConfigの運用設定へ反映される"""
    plan = EvaluationPlan.from_dict(
        {
            "robustness": {
                "enabled": True,
                "scenarios": [
                    {"type": "symbol", "symbol": "ETH/USDT:USDT"},
                    {"type": "slippage", "delta": 0.0002},
                    {"type": "commission", "multiplier": 1.5},
                ],
                "policy": "gate_only",
                "failure_policy": "fail_closed",
            }
        }
    )

    config = GAConfig()
    plan.apply_to_ga_config(config)

    assert config.robustness_config.enabled is True
    assert config.robustness_config.validation_symbols == ["ETH/USDT:USDT"]
    assert config.robustness_config.stress_slippage == [0.0002]
    assert config.robustness_config.stress_commission_multipliers == [1.5]


def test_apply_to_ga_config_disabled_keeps_legacy_fields():
    """無効な計画ではレガシー設定を保持する"""
    config = GAConfig(robustness_config={"validation_symbols": ["ETH/USDT:USDT"]})
    plan = EvaluationPlan()  # robustness.enabled はデフォルト False

    plan.apply_to_ga_config(config)

    assert config.robustness_config.enabled is False
    # シナリオが空の場合はレガシー設定を上書きしない
    assert config.robustness_config.validation_symbols == ["ETH/USDT:USDT"]


def test_legacy_explicit_enabled_flag_preserved_in_plan():
    """レガシー設定の明示的な enabled フラグが計画へ保持される"""
    config = GAConfig(robustness_config={"enabled": True})

    plan = EvaluationPlan.from_legacy_config(config)

    assert plan.robustness.enabled is True
