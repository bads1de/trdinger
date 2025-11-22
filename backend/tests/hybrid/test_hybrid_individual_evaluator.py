"""
HybridIndividualEvaluatorのテスト
"""

from unittest.mock import Mock, patch

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.core.hybrid_individual_evaluator import (
    HybridIndividualEvaluator,
)


class TestHybridIndividualEvaluator:
    """HybridIndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_hybrid_predictor = Mock()
        self.evaluator = HybridIndividualEvaluator(
            self.mock_backtest_service, self.mock_hybrid_predictor
        )

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator.predictor == self.mock_hybrid_predictor

    def test_evaluate_individual_success(self):
        """ハイブリッド個体評価成功のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [],
        }

        # バックテストサービスのモック
        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result

        # ハイブリッド予測器のモック
        self.mock_hybrid_predictor.predict.return_value = {"up": 0.6, "down": 0.2, "range": 0.2}

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1,
        }

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_evaluate_individual_multi_objective(self):
        """ハイブリッド多目的評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        self.mock_hybrid_predictor.predict.return_value = {"up": 0.65, "down": 0.2, "range": 0.15}

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "hybrid_score"]

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_evaluate_individual_exception(self):
        """ハイブリッド評価例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        # バックテストで例外
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert result == (0.0,)

    def test_evaluate_individual_volatility_mode(self):
        """ボラティリティ予測モードでの評価テスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10,
            },
            "equity_curve": [],
            "trade_history": [],
        }

        self.mock_backtest_service.run_backtest.return_value = mock_backtest_result
        # ボラティリティ予測のモック
        self.mock_hybrid_predictor.predict.return_value = {"trend": 0.8, "range": 0.2}

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "prediction_score": 0.1,
        }

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1
        # prediction_score = 0.8 - 0.5 = 0.3
        # fitness should include 0.1 * 0.3 = 0.03 boost
