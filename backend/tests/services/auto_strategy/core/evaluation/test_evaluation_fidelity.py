"""
multi-fidelity 評価ヘルパーのユニットテスト
"""

from datetime import datetime

from app.services.auto_strategy.config.ga_config import EvaluationConfig, GAConfig
from app.services.auto_strategy.core.evaluation.evaluation_fidelity import (
    adjust_backtest_config_for_fidelity,
    build_coarse_ga_config,
    get_multi_fidelity_candidate_limit,
)


def test_build_coarse_ga_config_disables_expensive_modes_and_uses_oos():
    config = GAConfig(
        evaluation_config=EvaluationConfig(
            enable_multi_fidelity_evaluation=True,
            enable_walk_forward=True,
            # 通常 OOS がデフォルト有効（0.25）でも coarse 評価では
            # multi_fidelity_oos_ratio が優先されることを確認する
            multi_fidelity_oos_ratio=0.2,
        ),
        enable_purged_kfold=True,
    )

    coarse = build_coarse_ga_config(config)

    assert coarse is not config
    assert coarse.evaluation_config.enable_walk_forward is False
    assert coarse.enable_purged_kfold is False
    assert coarse.evaluation_config.oos_split_ratio == 0.2
    assert getattr(coarse, "_evaluation_fidelity", "full") == "coarse"


def test_build_coarse_ga_config_keeps_user_oos_when_multi_fidelity_disabled():
    """multi-fidelity 無効時はユーザー指定の oos_split_ratio を維持する。"""
    config = GAConfig(
        evaluation_config=EvaluationConfig(
            enable_multi_fidelity_evaluation=False,
            enable_walk_forward=True,
            oos_split_ratio=0.25,
        ),
    )

    coarse = build_coarse_ga_config(config)

    assert coarse.evaluation_config.enable_walk_forward is False
    assert coarse.evaluation_config.oos_split_ratio == 0.25
    assert getattr(coarse, "_evaluation_fidelity", "full") == "coarse"


def test_adjust_backtest_config_for_fidelity_uses_recent_tail_window():
    config = GAConfig(
        evaluation_config=EvaluationConfig(
            enable_multi_fidelity_evaluation=True,
            multi_fidelity_window_ratio=0.3,
        ),
    )
    coarse = build_coarse_ga_config(config)
    backtest_config = {
        "start_date": "2024-01-01 00:00:00",
        "end_date": "2024-01-11 00:00:00",
    }

    adjusted = adjust_backtest_config_for_fidelity(backtest_config, coarse)

    assert adjusted["end_date"] == "2024-01-11 00:00:00"
    assert adjusted["start_date"] == "2024-01-08 00:00:00"


def test_adjust_backtest_config_for_fidelity_accepts_mixed_timezone_inputs():
    config = GAConfig(
        evaluation_config=EvaluationConfig(
            enable_multi_fidelity_evaluation=True,
            multi_fidelity_window_ratio=0.3,
        ),
    )
    coarse = build_coarse_ga_config(config)
    backtest_config = {
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-11T00:00:00Z",
    }

    adjusted = adjust_backtest_config_for_fidelity(backtest_config, coarse)

    assert adjusted["start_date"].endswith("+00:00")
    assert adjusted["end_date"].endswith("+00:00")
    assert datetime.fromisoformat(adjusted["start_date"]) < datetime.fromisoformat(
        adjusted["end_date"]
    )


def test_get_multi_fidelity_candidate_limit_respects_ratio_and_minimum():
    config = GAConfig(
        evaluation_config=EvaluationConfig(
            enable_multi_fidelity_evaluation=True,
            multi_fidelity_candidate_ratio=0.2,
            multi_fidelity_min_candidates=3,
        ),
    )

    limit = get_multi_fidelity_candidate_limit(10, config)

    assert limit == 3
