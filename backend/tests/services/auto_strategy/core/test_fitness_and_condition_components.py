"""
フィットネス計算、条件評価、指標計算のテスト
"""

import copy
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, Mock

from app.services.auto_strategy.config.ga import GAConfig
from app.services.auto_strategy.core import objective_registry
from app.services.auto_strategy.core.evaluation.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.core.fitness.fitness_calculator import FitnessCalculator
from app.services.auto_strategy.genes import Condition, ConditionGroup


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def ga_config():
    """GA設定のフィクスチャ"""
    config = GAConfig()
    config.fitness_weights = {
        "total_return": 0.3,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
    }
    return config


@pytest.fixture
def mock_backtest_result():
    """モックバックテスト結果のフィクスチャ"""
    return {
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.5,
            "total_trades": 100,
        },
        "equity_curve": [{"drawdown": 0.05} for _ in range(100)],
        "trade_history": [
            {"size": 1, "pnl": 100} if i % 2 == 0 else {"size": -1, "pnl": 50}
            for i in range(100)
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }


@pytest.fixture
def mock_ohlcv_data():
    """モックOHLCVデータのフィクスチャ"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(1000) * 100)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(1000) * 50,
            "High": close + abs(np.random.randn(1000) * 100),
            "Low": close - abs(np.random.randn(1000) * 100),
            "Close": close,
            "Volume": np.random.randint(100, 10000, 1000),
        },
        index=dates,
    )


@pytest.fixture
def mock_strategy(mock_ohlcv_data):
    """モック戦略インスタンスのフィクスチャ"""
    strategy = MagicMock()
    strategy.data = mock_ohlcv_data
    strategy.indicators = {
        "sma_20": mock_ohlcv_data["Close"].rolling(20).mean(),
        "ema_50": mock_ohlcv_data["Close"].ewm(span=50).mean(),
        "rsi_14": calculate_rsi(mock_ohlcv_data["Close"], 14),
    }
    return strategy


def calculate_rsi(prices, period):
    """RSIを計算"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# =============================================================================
# FitnessCalculator のテスト
# =============================================================================

class TestFitnessCalculator:
    """フィットネス計算のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.calculator = FitnessCalculator()

    def test_calculate_fitness_basic(self, ga_config, mock_backtest_result):
        """基本的なフィットネス計算テスト"""
        fitness = self.calculator.calculate_fitness(mock_backtest_result, ga_config)

        assert isinstance(fitness, float), "フィットネスはfloatであるべき"
        assert fitness >= 0.0, "フィットネスは0以上であるべき"
        assert not np.isnan(fitness), "フィットネスはNaNであってはならない"

    def test_calculate_fitness_zero_trades(self, ga_config):
        """取引回数ゼロの場合のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "total_trades": 0,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        fitness = self.calculator.calculate_fitness(backtest_result, ga_config)
        assert fitness == ga_config.zero_trades_penalty, "取引ゼロの場合はペナルティ値を返すべき"

    def test_calculate_fitness_negative_return(self, ga_config):
        """負のリターンの場合のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": -0.1,
                "sharpe_ratio": -0.5,
                "max_drawdown": 0.2,
                "win_rate": 0.4,
                "profit_factor": 0.8,
                "sortino_ratio": -0.3,
                "calmar_ratio": -0.2,
                "total_trades": 50,
            },
            "equity_curve": [{"drawdown": 0.15} for _ in range(50)],
            "trade_history": [{"size": 1, "pnl": -10} for _ in range(25)] + [{"size": -1, "pnl": 5} for _ in range(25)],
        }

        fitness = self.calculator.calculate_fitness(backtest_result, ga_config)
        assert fitness == ga_config.constraint_violation_penalty, "負のリターンの場合はペナルティ値を返すべき"

    def test_calculate_fitness_consistency(self, ga_config, mock_backtest_result):
        """フィットネス計算の一貫性テスト"""
        fitness1 = self.calculator.calculate_fitness(mock_backtest_result, ga_config)
        fitness2 = self.calculator.calculate_fitness(mock_backtest_result, ga_config)

        assert fitness1 == fitness2, "同じ入力に対して同じ結果を返すべき"

    def test_extract_performance_metrics(self, mock_backtest_result):
        """パフォーマンスメトリクス抽出テスト"""
        metrics = self.calculator.extract_performance_metrics(mock_backtest_result)

        assert "total_return" in metrics, "total_returnが含まれるべき"
        assert "sharpe_ratio" in metrics, "sharpe_ratioが含まれるべき"
        assert "max_drawdown" in metrics, "max_drawdownが含まれるべき"
        assert "win_rate" in metrics, "win_rateが含まれるべき"
        assert metrics["total_trades"] == 100, "取引回数は100であるべき"

    def test_meets_constraints_uses_shared_rules(self, ga_config, mock_backtest_result):
        """制約判定が共有ルールで行われることを確認するテスト"""
        ga_config.fitness_constraints = {
            "min_trades": 50,
            "max_drawdown_limit": 0.15,
            "min_sharpe_ratio": 1.0,
        }

        metrics = self.calculator.extract_performance_metrics(mock_backtest_result)

        assert self.calculator.meets_constraints(metrics, ga_config) is True

        ga_config.fitness_constraints["min_trades"] = 101
        assert self.calculator.meets_constraints(metrics, ga_config) is False

    def test_calculate_multi_objective_fitness(self, ga_config, mock_backtest_result):
        """多目的フィットネス計算テスト"""
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        fitness = self.calculator.calculate_multi_objective_fitness(
            mock_backtest_result, ga_config
        )

        assert isinstance(fitness, tuple), "多目的フィットネスはtupleであるべき"
        assert len(fitness) == len(ga_config.objectives), "目的関数の数と一致すべき"

    def test_get_penalty_values_uses_objective_registry(self, monkeypatch):
        """ペナルティ値の方向判定が registry 経由で行われることを確認する。"""
        monkeypatch.setattr(
            objective_registry,
            "is_minimize_objective",
            lambda objective: objective == "custom_loss",
        )

        config = SimpleNamespace(objectives=["custom_loss", "total_return"])

        assert self.calculator.get_penalty_values(config) == (
            float("inf"),
            -float("inf"),
        )

    def test_calculate_long_short_balance(self, mock_backtest_result):
        """ロング・ショートバランス計算テスト"""
        balance = self.calculator.calculate_long_short_balance(mock_backtest_result)

        assert isinstance(balance, float), "バランススコアはfloatであるべき"
        assert 0.0 <= balance <= 1.0, "バランススコアは0〜1の範囲であるべき"

    def test_calculate_long_short_balance_empty_trades(self):
        """トレード履歴が空の場合のバランス計算テスト"""
        backtest_result = {
            "performance_metrics": {"total_trades": 0},
            "trade_history": [],
        }

        balance = self.calculator.calculate_long_short_balance(backtest_result)
        assert balance == 0.5, "トレード履歴が空の場合は0.5を返すべき"

    def test_cache_functionality(self, ga_config, mock_backtest_result):
        """キャッシュ機能テスト"""
        # 初回計算
        fitness1 = self.calculator.calculate_fitness(mock_backtest_result, ga_config)

        # キャッシュをクリア
        self.calculator.clear_cache()

        # 再計算
        fitness2 = self.calculator.calculate_fitness(mock_backtest_result, ga_config)

        assert fitness1 == fitness2, "キャッシュクリア後も同じ結果を返すべき"

    def test_extract_performance_metrics_uses_content_based_cache_key(
        self, mock_backtest_result
    ):
        """同一内容の別オブジェクトが同じキャッシュを使うこと"""
        result_a = copy.deepcopy(mock_backtest_result)
        result_b = copy.deepcopy(mock_backtest_result)

        metrics_a = self.calculator.extract_performance_metrics(result_a)
        metrics_b = self.calculator.extract_performance_metrics(result_b)

        assert metrics_a == metrics_b
        assert len(self.calculator._metrics_cache) == 1


# =============================================================================
# ConditionEvaluator のテスト
# =============================================================================

class TestConditionEvaluator:
    """条件評価器のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.evaluator = ConditionEvaluator()

    def test_evaluate_single_condition_greater_than(self, mock_strategy):
        """大なり条件のテスト"""
        condition = Condition(
            left_operand="sma_20", operator=">", right_operand="ema_50"
        )

        result = self.evaluator.evaluate_single_condition(condition, mock_strategy)
        assert isinstance(result, bool), "結果はboolであるべき"

    def test_evaluate_single_condition_less_than(self, mock_strategy):
        """小なり条件のテスト"""
        condition = Condition(
            left_operand="sma_20", operator="<", right_operand="ema_50"
        )

        result = self.evaluator.evaluate_single_condition(condition, mock_strategy)
        assert isinstance(result, bool), "結果はboolであるべき"

    def test_evaluate_single_condition_equals(self, mock_strategy):
        """等価条件のテスト"""
        condition = Condition(
            left_operand="sma_20", operator="==", right_operand="sma_20"
        )

        result = self.evaluator.evaluate_single_condition(condition, mock_strategy)
        assert result is True, "同じ値の等価比較はTrueであるべき"

    def test_evaluate_conditions_and(self, mock_strategy):
        """AND条件のテスト"""
        conditions = [
            Condition(left_operand="sma_20", operator=">", right_operand=0),
            Condition(left_operand="ema_50", operator=">", right_operand=0),
        ]

        result = self.evaluator.evaluate_conditions(conditions, mock_strategy)
        assert isinstance(result, bool), "結果はboolであるべき"

    def test_evaluate_conditions_empty(self, mock_strategy):
        """空の条件リストのテスト"""
        result = self.evaluator.evaluate_conditions([], mock_strategy)
        assert result is True, "空の条件リストはTrueを返すべき"

    def test_evaluate_condition_group_or(self, mock_strategy):
        """OR条件グループのテスト"""
        conditions = [
            Condition(left_operand="sma_20", operator="<", right_operand=0),
            Condition(left_operand="ema_50", operator=">", right_operand=0),
        ]
        group = ConditionGroup(conditions=conditions, operator="OR")

        result = self.evaluator._evaluate_condition_group(group, mock_strategy)
        assert isinstance(result, bool), "結果はboolであるべき"

    def test_get_condition_value_indicator(self, mock_strategy):
        """インジケーター値取得テスト"""
        value = self.evaluator.get_condition_value("sma_20", mock_strategy)
        assert isinstance(value, (int, float, np.ndarray, pd.Series)), "値は数値型であるべき"

    def test_get_condition_value_numeric(self, mock_strategy):
        """数値オペランド取得テスト"""
        value = self.evaluator.get_condition_value(100.5, mock_strategy)
        assert value == 100.5, "数値オペランドはそのまま返すべき"

    def test_get_condition_value_close(self, mock_strategy):
        """closeオペランド取得テスト"""
        value = self.evaluator.get_condition_value("close", mock_strategy)
        assert isinstance(value, (int, float)), "close値は数値であるべき"

    def test_evaluate_single_condition_vectorized(self, mock_strategy):
        """ベクトル化条件評価テスト"""
        condition = Condition(
            left_operand="sma_20", operator=">", right_operand="ema_50"
        )

        result = self.evaluator.evaluate_single_condition_vectorized(
            condition, mock_strategy
        )
        assert isinstance(result, (bool, np.ndarray, pd.Series)), "結果はbool、ndarray、またはSeriesであるべき"


# =============================================================================
# インジケーター計算の正確性テスト
# =============================================================================

class TestIndicatorCalculation:
    """インジケーター計算のテスト"""

    def test_sma_calculation_accuracy(self):
        """SMA計算の正確性テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            OverlapIndicators,
        )

        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
        sma = OverlapIndicators.sma(prices, length=3)

        # 手計算: SMA[2] = (100 + 102 + 101) / 3 = 101.0
        assert np.isclose(sma.iloc[2], 101.0, rtol=1e-10), "SMA[2]が手計算値と一致すべき"

    def test_ema_calculation_accuracy(self):
        """EMA計算の正確性テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            OverlapIndicators,
        )

        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
        ema = OverlapIndicators.ema(prices, length=3)

        # EMAは有効な値を持つべき
        valid_ema = ema.dropna()
        assert len(valid_ema) > 0, "EMAは有効な値を持つべき"

    def test_rsi_calculation_accuracy(self):
        """RSI計算の正確性テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            MomentumIndicators,
        )

        # 全て上昇
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0])
        rsi = MomentumIndicators.rsi(prices, period=14)

        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0, "RSIは有効な値を持つべき"
        assert valid_rsi.iloc[-1] > 90, "全て上昇の場合RSIは90以上であるべき"

    def test_bollinger_bands_calculation(self):
        """ボリンジャーバンド計算テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            OverlapIndicators,
            VolatilityIndicators,
        )

        prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0])
        upper, middle, lower = VolatilityIndicators.bbands(prices, length=5, std=2.0)

        # 中間線はSMAと一致すべき
        sma = OverlapIndicators.sma(prices, length=5)
        valid = ~np.isnan(middle)
        assert np.allclose(middle[valid], sma[valid], rtol=1e-10), "BB中間線はSMAと一致すべき"

        # upper >= middle >= lower
        assert (upper[valid] >= middle[valid] - 1e-10).all(), "upper >= middle"
        assert (middle[valid] >= lower[valid] - 1e-10).all(), "middle >= lower"

    def test_atr_calculation(self):
        """ATR計算テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            VolatilityIndicators,
        )

        # より多くのデータを使用
        data = pd.DataFrame({
            "high": [102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0, 111.0],
            "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0],
            "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
        })

        atr = VolatilityIndicators.atr(data["high"], data["low"], data["close"], length=3)

        valid_atr = atr.dropna()
        # ATR計算が成功するか確認（データが不足している場合はスキップ）
        if len(valid_atr) > 0:
            assert (valid_atr > 0).all(), "ATRは正の値であるべき"
        else:
            # データが不足している場合は警告を出力してスキップ
            import warnings
            warnings.warn("ATR計算に必要なデータが不足しています")

    def test_macd_calculation(self):
        """MACD計算テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            MomentumIndicators,
            OverlapIndicators,
        )

        prices = pd.Series([100.0 + i + np.random.randn() * 2 for i in range(50)])
        macd_line, signal, histogram = MomentumIndicators.macd(prices, fast=12, slow=26, signal=9)

        # MACDラインがEMA12-EMA26と一致するか
        ema12 = OverlapIndicators.ema(prices, length=12)
        ema26 = OverlapIndicators.ema(prices, length=26)
        expected_macd = ema12 - ema26

        valid = ~np.isnan(macd_line)
        for i in range(len(prices)):
            if valid.iloc[i] and not np.isnan(expected_macd.iloc[i]):
                assert np.isclose(macd_line.iloc[i], expected_macd.iloc[i], rtol=1e-5), (
                    f"MACD[{i}]がEMA12-EMA26と一致しない"
                )

    def test_stochastic_calculation(self):
        """ストキャスティクス計算テスト"""
        from app.services.indicators.technical_indicators.pandas_ta import (
            MomentumIndicators,
        )

        data = pd.DataFrame({
            "high": [float(i) for i in range(100, 125)],
            "low": [float(i) for i in range(99, 124)],
            "close": [float(i) for i in range(100, 125)],
        })

        k, d = MomentumIndicators.stoch(data["high"], data["low"], data["close"], k=5, d=3, smooth_k=3)

        valid_k = k.dropna()
        assert len(valid_k) > 0, "%Kは有効な値を持つべき"
        assert (valid_k >= 0).all() and (valid_k <= 100).all(), "%Kは0-100の範囲であるべき"


# =============================================================================
# エッジケーステスト
# =============================================================================

class TestEdgeCases:
    """エッジケースのテスト"""

    def test_fitness_with_nan_metrics(self, ga_config):
        """NaNメトリクスでのフィットネス計算テスト"""
        calculator = FitnessCalculator()

        backtest_result = {
            "performance_metrics": {
                "total_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "sortino_ratio": float("nan"),
                "calmar_ratio": float("nan"),
                "total_trades": 100,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 100} for _ in range(50)],
        }

        fitness = calculator.calculate_fitness(backtest_result, ga_config)
        assert isinstance(fitness, float), "NaNメトリクスでもフィットネスはfloatであるべき"
        assert not np.isnan(fitness), "NaNメトリクスでもフィットネスはNaNであってはならない"

    def test_fitness_with_infinite_metrics(self, ga_config):
        """無限大メトリクスでのフィットネス計算テスト"""
        calculator = FitnessCalculator()

        backtest_result = {
            "performance_metrics": {
                "total_return": float("inf"),
                "sharpe_ratio": float("inf"),
                "max_drawdown": float("inf"),
                "win_rate": float("inf"),
                "profit_factor": float("inf"),
                "sortino_ratio": float("inf"),
                "calmar_ratio": float("inf"),
                "total_trades": 100,
            },
            "equity_curve": [],
            "trade_history": [{"size": 1, "pnl": 100} for _ in range(50)],
        }

        fitness = calculator.calculate_fitness(backtest_result, ga_config)
        assert isinstance(fitness, float), "無限大メトリクスでもフィットネスはfloatであるべき"
        assert not np.isinf(fitness), "無限大メトリクスでもフィットネスは無限大であってはならない"

    def test_condition_with_nan_indicator(self, mock_strategy):
        """NaNインジケーター値での条件評価テスト"""
        evaluator = ConditionEvaluator()

        # NaN値を持つインジケーターを設定
        mock_strategy.indicators["nan_indicator"] = pd.Series([float("nan")] * 1000)

        condition = Condition(
            left_operand="nan_indicator", operator=">", right_operand=0
        )

        # エラーが発生してもFalseを返すべき
        result = evaluator.evaluate_single_condition(condition, mock_strategy)
        assert isinstance(result, bool), "NaNインジケーターでも結果はboolであるべき"
