import pytest

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.core.fitness.constraint_scaling import (
    effective_max_drawdown_limit,
    effective_min_trades,
    get_effective_fitness_constraints,
)
from app.services.auto_strategy.core.fitness.fitness_calculator import FitnessCalculator


def test_effective_max_drawdown_no_scaling_for_short_window():
    assert effective_max_drawdown_limit(0.2, 90) == pytest.approx(0.2)
    assert effective_max_drawdown_limit(0.2, 180) == pytest.approx(0.2)
    assert effective_max_drawdown_limit(0.2, None) == pytest.approx(0.2)


def test_effective_max_drawdown_sqrt_scaling():
    # 360日 = 180日の2倍 → sqrt(2) 約1.414倍
    assert effective_max_drawdown_limit(0.2, 360) == pytest.approx(
        0.2 * (360 / 180) ** 0.5
    )
    # 720日 = 180日の4倍 → sqrt(4)=2倍だが cap 0.40 で止まる
    assert effective_max_drawdown_limit(0.2, 720) == pytest.approx(0.40)
    # capを上げれば 0.4 を超えて伸びる
    assert effective_max_drawdown_limit(0.2, 720, cap=0.5) == pytest.approx(0.4)


def test_effective_max_drawdown_capped():
    assert effective_max_drawdown_limit(0.2, 720, cap=0.35) == pytest.approx(0.35)
    assert effective_max_drawdown_limit(0.2, 9999, cap=0.35) == pytest.approx(0.35)


def test_effective_min_trades_no_scaling_for_long_window():
    assert effective_min_trades(10, 360) == 10
    assert effective_min_trades(10, 180) == 10


def test_effective_min_trades_scaled_for_short_window():
    # 半分の窓 → 半分の取引数
    assert effective_min_trades(10, 90) == 5
    # 30日フォールド: 10 * 30/180 ≒ 1.67 → 切り捨てで 1（1トレードでも失格にしない）
    assert effective_min_trades(10, 30) == 1
    # 極短窓でも最低1
    assert effective_min_trades(10, 1) == 1


def test_get_effective_constraints_combined():
    base = {"min_trades": 10, "max_drawdown_limit": 0.2, "min_sharpe_ratio": 0.5}
    # 360日: DD緩和、min_tradesは据え置き
    eff = get_effective_fitness_constraints(base, 360)
    assert eff["max_drawdown_limit"] == pytest.approx(0.2 * (360 / 180) ** 0.5)
    assert eff["min_trades"] == 10
    assert eff["min_sharpe_ratio"] == 0.5
    # 元のdictは変更されない
    assert base["max_drawdown_limit"] == 0.2

    # 30日fold: DD据え置き、min_trades縮小（切り捨てで1まで緩和）
    eff2 = get_effective_fitness_constraints(base, 30)
    assert eff2["max_drawdown_limit"] == pytest.approx(0.2)
    assert eff2["min_trades"] == 1


def test_meets_constraints_backward_compatible():
    calc = FitnessCalculator()
    config = GAConfig(
        fitness_constraints={
            "min_trades": 10,
            "max_drawdown_limit": 0.2,
            "min_total_return": 0.0,
            "min_sharpe_ratio": 0.5,
        }
    )
    # 従来通りwindow_days無しでも動作
    assert calc.meets_constraints(
        {
            "total_trades": 10,
            "max_drawdown": 0.15,
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
        },
        config,
    )
    assert not calc.meets_constraints(
        {
            "total_trades": 10,
            "max_drawdown": 0.25,
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
        },
        config,
    )


def test_meets_constraints_window_scaling_allows_long_window_drawdown():
    calc = FitnessCalculator()
    config = GAConfig(
        fitness_constraints={
            "min_trades": 10,
            "max_drawdown_limit": 0.2,
            "min_total_return": 0.0,
            "min_sharpe_ratio": 0.5,
        }
    )
    # DD 25%: 180日では違反
    assert not calc.meets_constraints(
        {
            "total_trades": 10,
            "max_drawdown": 0.25,
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
        },
        config,
        window_days=180,
    )
    # 360日では sqrt(2)で 0.2→0.283 に緩和されるため通過
    assert calc.meets_constraints(
        {
            "total_trades": 10,
            "max_drawdown": 0.25,
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
        },
        config,
        window_days=360,
    )
    # 700日でもcap 0.40を超えるDD 41%は違反
    assert not calc.meets_constraints(
        {
            "total_trades": 10,
            "max_drawdown": 0.41,
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
        },
        config,
        window_days=700,
    )


def test_meets_constraints_window_scaling_short_window_trades():
    calc = FitnessCalculator()
    config = GAConfig(
        fitness_constraints={
            "min_trades": 10,
            "max_drawdown_limit": 0.2,
            "min_total_return": None,
            "min_sharpe_ratio": None,
        }
    )
    # 5トレード: 180日では違反
    assert not calc.meets_constraints(
        {"total_trades": 5, "max_drawdown": 0.1},
        config,
        window_days=180,
    )
    # 30日foldでは min_tradesが1に縮小されるため通過
    assert calc.meets_constraints(
        {"total_trades": 5, "max_drawdown": 0.1},
        config,
        window_days=30,
    )


def test_calculate_fitness_uses_window_days_from_result_dates():
    calc = FitnessCalculator()
    config = GAConfig(
        fitness_constraints={
            "min_trades": 10,
            "max_drawdown_limit": 0.2,
            "min_total_return": 0.0,
            "min_sharpe_ratio": 0.5,
        },
        fitness_weights={
            "total_return": 0.3,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.2,
            "win_rate": 0.15,
            "balance_score": 0.1,
            "ulcer_index_penalty": 0.1,
            "trade_frequency_penalty": 0.05,
        },
    )
    # DD 25%の戦略: 180日窓なら制約違反でペナルティ0.0
    result_180 = {
        "performance_metrics": {
            "total_return": 10.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 25.0,
            "win_rate": 55.0,
            "total_trades": 50,
        },
        "trade_history": [{"size": 1, "pnl": 100}],
        "start_date": "2024-01-01",
        "end_date": "2024-06-29",
    }
    assert calc.calculate_fitness(result_180, config) == pytest.approx(0.0)

    # 同じ成績でも360日窓なら DD上限が0.283に緩和されるため有効fitness > 0
    result_360 = dict(result_180)
    result_360["end_date"] = "2024-12-26"
    # start_dateが同じでend_dateだけ360日に延ばす
    result_360["start_date"] = "2024-01-01"
    # 明示的にwindow_daysを渡しても同じ効果
    fitness_explicit = calc.calculate_fitness(result_180, config, window_days=360)
    assert fitness_explicit > 0.0
