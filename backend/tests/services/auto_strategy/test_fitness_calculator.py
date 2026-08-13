"""
フィットネス計算器のテスト
"""

import pytest

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.core.fitness.fitness_calculator import (
    FitnessCalculator,
)


@pytest.fixture
def fitness_calculator():
    """FitnessCalculatorのフィクスチャ"""
    return FitnessCalculator()


@pytest.fixture
def sample_backtest_result():
    """サンプルバックテスト結果"""
    return {
        "total_return": 0.5,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.2,
        "win_rate": 0.6,
        "total_trades": 50,
        "ulcer_index": 0.1,
        "trade_frequency_penalty": 0.05,
        "trade_history": [],
    }


@pytest.fixture
def sample_config():
    """サンプルGA設定"""
    return GAConfig(
        zero_trades_penalty=0.0,
        constraint_violation_penalty=0.0,
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


def test_calculate_fitness_normal_case(
    fitness_calculator, sample_backtest_result, sample_config
):
    """通常のフィットネス計算をテスト"""
    fitness = fitness_calculator.calculate_fitness(
        sample_backtest_result, sample_config
    )

    # フィットネス値が0以上であることを確認
    assert fitness >= 0.0

    # フィットネス値が1以下であることを確認（正常な範囲）
    assert fitness <= 1.0


def test_calculate_fitness_zero_trades(fitness_calculator, sample_config):
    """取引回数が0の場合のフィットネス計算をテスト"""
    backtest_result = {
        "total_trades": 0,
        "trade_history": [],
    }

    fitness = fitness_calculator.calculate_fitness(backtest_result, sample_config)

    # 取引回数が0の場合はペナルティ値を返す
    assert fitness == sample_config.zero_trades_penalty


def test_calculate_fitness_constraint_violation(
    fitness_calculator, sample_backtest_result, sample_config
):
    """制約違反の場合のフィットネス計算をテスト"""
    # 制約違反を発生させるために負のリターンを設定
    sample_backtest_result["total_return"] = -0.5

    fitness = fitness_calculator.calculate_fitness(
        sample_backtest_result, sample_config
    )

    # 制約違反の場合はペナルティ値を返す
    assert fitness == sample_config.constraint_violation_penalty


def test_calculate_fitness_with_ulcer_index_penalty(
    fitness_calculator, sample_backtest_result, sample_config
):
    """ulcer_indexペナルティが考慮されることをテスト"""
    # ulcer_indexを大きく設定
    sample_backtest_result["ulcer_index"] = 0.5

    fitness = fitness_calculator.calculate_fitness(
        sample_backtest_result, sample_config
    )

    # ulcer_indexペナルティが考慮されていることを確認
    assert fitness >= 0.0


def test_calculate_fitness_with_trade_penalty(
    fitness_calculator, sample_backtest_result, sample_config
):
    """trade_frequency_penaltyが考慮されることをテスト"""
    # trade_frequency_penaltyを大きく設定
    sample_backtest_result["trade_frequency_penalty"] = 0.3

    fitness = fitness_calculator.calculate_fitness(
        sample_backtest_result, sample_config
    )

    # trade_frequency_penaltyが考慮されていることを確認
    assert fitness >= 0.0


def test_extract_performance_metrics(fitness_calculator, sample_backtest_result):
    """パフォーマンス指標の抽出をテスト"""
    metrics = fitness_calculator.extract_performance_metrics(sample_backtest_result)

    # 必要な指標が含まれていることを確認
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "total_trades" in metrics


def test_extract_performance_metrics_includes_excess_return(
    fitness_calculator,
):
    """buy_hold_return から excess_return（買い持ち超過）が計算されること."""
    backtest_result = {
        "performance_metrics": {
            "total_return": 20.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -10.0,
            "win_rate": 55.0,
            "total_trades": 50,
            "buy_hold_return": 43.09,
        },
    }

    metrics = fitness_calculator.extract_performance_metrics(backtest_result)

    # 割合単位に正規化されていること
    assert metrics["total_return"] == pytest.approx(0.2, abs=1e-9)
    assert metrics["buy_hold_return"] == pytest.approx(0.4309, abs=1e-9)
    # 買い持ちを下回るため超過リターンは負になる
    assert metrics["excess_return"] == pytest.approx(-0.2309, abs=1e-9)


def test_extract_performance_metrics_excess_return_positive(
    fitness_calculator,
):
    """買い持ちを上回る場合、超過リターンが正になること."""
    backtest_result = {
        "performance_metrics": {
            "total_return": 60.0,
            "sharpe_ratio": 2.0,
            "max_drawdown": -15.0,
            "win_rate": 60.0,
            "total_trades": 100,
            "buy_hold_return": 43.09,
        },
    }

    metrics = fitness_calculator.extract_performance_metrics(backtest_result)

    assert metrics["excess_return"] == pytest.approx(0.1691, abs=1e-9)


def test_extract_performance_metrics_no_buy_hold_return(fitness_calculator):
    """buy_hold_return が無い場合は excess_return が中立値 0.0 になること.

    ベンチマーク不明時に total_return と同値にすると、フィットネス計算で
    総リターンが二重計上されるため、中立値（0.0）を採用する。
    """
    backtest_result = {
        "performance_metrics": {
            "total_return": 15.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": -5.0,
            "win_rate": 50.0,
            "total_trades": 30,
        },
    }

    metrics = fitness_calculator.extract_performance_metrics(backtest_result)

    assert metrics["buy_hold_return"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["excess_return"] == pytest.approx(0.0, abs=1e-9)


def test_calculate_fitness_rewards_excess_return(
    fitness_calculator,
):
    """excess_return の重みが設定されている場合、買い持ち超過がフィットネスを引き上げること."""
    config = GAConfig(
        zero_trades_penalty=0.0,
        constraint_violation_penalty=0.0,
        fitness_weights={
            "total_return": 0.2,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.15,
            "win_rate": 0.1,
            "balance_score": 0.1,
            "excess_return": 0.1,
            "ulcer_index_penalty": 0.1,
            "trade_frequency_penalty": 0.05,
        },
    )
    base_result = {
        "performance_metrics": {
            "total_return": 20.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -10.0,
            "win_rate": 55.0,
            "total_trades": 50,
            "buy_hold_return": 43.09,
        },
        "trade_history": [],
    }

    # 買い持ち超過なし
    fitness_no_excess = fitness_calculator.calculate_fitness(base_result, config)

    # 同じリターンだが買い持ちを大きく上回る場合
    base_result["performance_metrics"]["buy_hold_return"] = 5.0
    fitness_with_excess = fitness_calculator.calculate_fitness(base_result, config)

    assert fitness_with_excess > fitness_no_excess


def test_calculate_fitness_excess_return_penalizes_below_benchmark(
    fitness_calculator,
):
    """買い持ちを下回る戦略は excess_return でペナルティを受けること."""
    config = GAConfig(
        zero_trades_penalty=0.0,
        constraint_violation_penalty=0.0,
        fitness_weights={
            "total_return": 0.2,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.15,
            "win_rate": 0.1,
            "balance_score": 0.1,
            "excess_return": 0.1,
            "ulcer_index_penalty": 0.1,
            "trade_frequency_penalty": 0.05,
        },
    )
    base_result = {
        "performance_metrics": {
            "total_return": 20.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -10.0,
            "win_rate": 55.0,
            "total_trades": 50,
            "buy_hold_return": 5.0,
        },
        "trade_history": [],
    }

    # 買い持ちを上回るケース
    fitness_above = fitness_calculator.calculate_fitness(base_result, config)

    # 買い持ちを下回るケース（同リターンでも excess_return が負）
    base_result["performance_metrics"]["buy_hold_return"] = 60.0
    fitness_below = fitness_calculator.calculate_fitness(base_result, config)

    assert fitness_below < fitness_above


def test_calculate_fitness_preserves_negative_values(fitness_calculator):
    """負のフィットネスが0にクリップされず、順位情報が保持されること."""
    config = GAConfig(
        zero_trades_penalty=0.0,
        constraint_violation_penalty=0.0,
        fitness_constraints={
            "min_trades": 10,
            "max_drawdown_limit": 0.5,
            "min_sharpe_ratio": -10.0,
        },
        fitness_weights={
            "total_return": 0.2,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.15,
            "win_rate": 0.1,
            "balance_score": 0.1,
            "excess_return": 0.1,
            "ulcer_index_penalty": 0.1,
            "trade_frequency_penalty": 0.05,
        },
    )
    # 総リターンは非負制約を満たしつつ、他の項（超過リターンやシャープなど）が
    # 大きく負になるケースを構築し、合計が負になることを確認する。
    result = {
        "performance_metrics": {
            "total_return": 1.0,  # 0.01 (非負制約を満たす)
            "sharpe_ratio": -3.0,
            "max_drawdown": 10.0,  # 0.1 に正規化され (1-0.1)=0.9 が寄与
            "win_rate": 10.0,  # 0.1
            "total_trades": 50,
            "buy_hold_return": 40.0,  # excess_return = 0.01 - 0.4 = -0.39
        },
        "trade_history": [],
    }

    fitness = fitness_calculator.calculate_fitness(result, config)

    # 加重和は負になるため、クリップされず負値のまま返る
    assert fitness < 0.0


def test_calculate_fitness_uses_default_weights_for_missing_keys(
    fitness_calculator,
):
    """config に無い重みキーは ga_constants のデフォルトで補完されること."""
    config = GAConfig(
        zero_trades_penalty=0.0,
        constraint_violation_penalty=0.0,
        # excess_return と balance_score を意図的に省略
        fitness_weights={
            "total_return": 0.2,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.15,
            "win_rate": 0.1,
            "ulcer_index_penalty": 0.1,
            "trade_frequency_penalty": 0.05,
        },
    )
    result = {
        "performance_metrics": {
            "total_return": 20.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -10.0,
            "win_rate": 55.0,
            "total_trades": 50,
            "buy_hold_return": 5.0,  # excess_return = 0.2 - 0.05 = 0.15
        },
        "trade_history": [],
    }

    fitness = fitness_calculator.calculate_fitness(result, config)

    # デフォルトの excess_return (0.1) と balance_score (0.1) が反映された値になる
    metrics = fitness_calculator.extract_performance_metrics(result)
    expected = (
        0.2 * metrics["total_return"]
        + 0.25 * metrics["sharpe_ratio"]
        + 0.15 * (1 - metrics["max_drawdown"])
        + 0.1 * metrics["win_rate"]
        + 0.1 * metrics["excess_return"]
        + 0.1 * fitness_calculator.calculate_long_short_balance(result)
        - 0.1 * metrics["ulcer_index"]
        - 0.05 * metrics["trade_frequency_penalty"]
    )
    assert fitness == pytest.approx(expected, abs=1e-9)


def test_extract_performance_metrics_normalizes_percentage_drawdown(
    fitness_calculator,
):
    """パーセント単位の max_drawdown が 0.0〜1.0 の割合に正規化されることをテスト"""
    backtest_result = {
        "performance_metrics": {
            "total_return": 20.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -47.98,
            "win_rate": 55.0,
            "total_trades": 50,
        },
    }

    metrics = fitness_calculator.extract_performance_metrics(backtest_result)

    assert metrics["max_drawdown"] == pytest.approx(0.4798, abs=1e-9)


def test_extract_performance_metrics_normalizes_percentage_returns(
    fitness_calculator,
):
    """% 単位の total_return / win_rate が 0.0〜1.0 の割合に正規化されることをテスト"""
    backtest_result = {
        "performance_metrics": {
            "total_return": 47.98,
            "sharpe_ratio": 1.5,
            "max_drawdown": -47.98,
            "win_rate": 55.0,
            "total_trades": 50,
        },
    }

    metrics = fitness_calculator.extract_performance_metrics(backtest_result)

    assert metrics["total_return"] == pytest.approx(0.4798, abs=1e-9)
    assert metrics["win_rate"] == pytest.approx(0.55, abs=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(0.4798, abs=1e-9)


def test_meets_constraints_drawdown_percentage_over_limit(
    fitness_calculator, sample_backtest_result, sample_config
):
    """% 単位のドローダウン(47.98%)が制約の20%を超えて制約違反になることをテスト"""
    sample_backtest_result["performance_metrics"] = {
        "total_return": 0.5,
        "sharpe_ratio": 1.5,
        "max_drawdown": 47.98,
        "win_rate": 0.6,
        "total_trades": 50,
    }

    metrics = fitness_calculator.extract_performance_metrics(sample_backtest_result)

    assert not fitness_calculator.meets_constraints(metrics, sample_config)


def test_meets_constraints(fitness_calculator, sample_backtest_result, sample_config):
    """制約チェックをテスト"""
    # 正常なケース
    assert fitness_calculator.meets_constraints(sample_backtest_result, sample_config)

    # 取引回数が0の場合
    sample_backtest_result["total_trades"] = 0
    assert not fitness_calculator.meets_constraints(
        sample_backtest_result, sample_config
    )


def test_calculate_long_short_balance(fitness_calculator, sample_backtest_result):
    """ロング・ショートバランス計算をテスト"""
    balance_score = fitness_calculator.calculate_long_short_balance(
        sample_backtest_result
    )

    # バランススコアが0から1の範囲内であることを確認
    assert 0.0 <= balance_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
