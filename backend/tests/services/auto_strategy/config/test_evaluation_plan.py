import pytest

from app.services.auto_strategy.config import EvaluationPlan, GAConfig


def test_evaluation_plan_round_trip_is_stable():
    plan = EvaluationPlan.from_dict(
        {
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


def test_evaluation_plan_from_dict_ignores_unknown_keys():
    plan = EvaluationPlan.from_dict(
        {
            "schema_version": "1",
            "validation": {"method": "oos"},
            "robustness": {"enabled": True},
        }
    )

    assert plan.robustness.enabled is True
    assert plan.robustness.scenarios == []


@pytest.mark.parametrize(
    "updates",
    [
        {"robustness": {"policy": "rank_all"}},
        {"robustness": {"failure_policy": "fallback"}},
    ],
)
def test_evaluation_plan_rejects_ambiguous_or_unsafe_modes(updates):
    with pytest.raises(ValueError):
        EvaluationPlan.from_dict(updates)


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
