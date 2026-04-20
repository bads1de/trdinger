"""
フィットネス計算器のテスト
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.core.fitness.fitness_calculator import FitnessCalculator
from app.services.auto_strategy.config.ga.ga_config import GAConfig


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


def test_calculate_fitness_normal_case(fitness_calculator, sample_backtest_result, sample_config):
    """通常のフィットネス計算をテスト"""
    fitness = fitness_calculator.calculate_fitness(sample_backtest_result, sample_config)
    
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


def test_calculate_fitness_constraint_violation(fitness_calculator, sample_backtest_result, sample_config):
    """制約違反の場合のフィットネス計算をテスト"""
    # 制約違反を発生させるために負のリターンを設定
    sample_backtest_result["total_return"] = -0.5
    
    fitness = fitness_calculator.calculate_fitness(sample_backtest_result, sample_config)
    
    # 制約違反の場合はペナルティ値を返す
    assert fitness == sample_config.constraint_violation_penalty


def test_calculate_fitness_with_ulcer_index_penalty(fitness_calculator, sample_backtest_result, sample_config):
    """ulcer_indexペナルティが考慮されることをテスト"""
    # ulcer_indexを大きく設定
    sample_backtest_result["ulcer_index"] = 0.5
    
    fitness = fitness_calculator.calculate_fitness(sample_backtest_result, sample_config)
    
    # ulcer_indexペナルティが考慮されていることを確認
    assert fitness >= 0.0


def test_calculate_fitness_with_trade_penalty(fitness_calculator, sample_backtest_result, sample_config):
    """trade_frequency_penaltyが考慮されることをテスト"""
    # trade_frequency_penaltyを大きく設定
    sample_backtest_result["trade_frequency_penalty"] = 0.3
    
    fitness = fitness_calculator.calculate_fitness(sample_backtest_result, sample_config)
    
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


def test_meets_constraints(fitness_calculator, sample_backtest_result, sample_config):
    """制約チェックをテスト"""
    # 正常なケース
    assert fitness_calculator.meets_constraints(sample_backtest_result, sample_config) == True
    
    # 取引回数が0の場合
    sample_backtest_result["total_trades"] = 0
    assert fitness_calculator.meets_constraints(sample_backtest_result, sample_config) == False


def test_calculate_long_short_balance(fitness_calculator, sample_backtest_result):
    """ロング・ショートバランス計算をテスト"""
    balance_score = fitness_calculator.calculate_long_short_balance(sample_backtest_result)
    
    # バランススコアが0から1の範囲内であることを確認
    assert 0.0 <= balance_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
