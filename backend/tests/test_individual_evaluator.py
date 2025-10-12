"""
IndividualEvaluatorのテスト
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import StrategyGene


class TestIndividualEvaluator:
    """IndividualEvaluatorのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.evaluator = IndividualEvaluator(self.mock_backtest_service)

    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.backtest_service == self.mock_backtest_service
        assert self.evaluator.regime_detector is None
        assert self.evaluator._fixed_backtest_config is None

    def test_set_backtest_config(self):
        """バックテスト設定のテスト"""
        config = {"symbol": "BTC/USDT:USDT", "timeframe": "1h"}
        self.evaluator.set_backtest_config(config)
        assert self.evaluator._fixed_backtest_config == config

    def test_evaluate_individual_success(self):
        """個体評価成功のテスト"""
        # モック設定
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "profit_factor": 1.8
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": [{"size": 1, "pnl": 10}, {"size": -1, "pnl": -5}]
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_constraints = {}
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }

        # テスト実行
        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 1  # 単一目的最適化

    def test_evaluate_individual_multi_objective(self):
        """多目的最適化評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1
            },
            "equity_curve": [],
            "trade_history": []
        }

        self.mock_backtest_service.run_backtest.return_value = mock_result

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 3  # 3つの目的

    def test_evaluate_individual_exception(self):
        """個体評価例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        # バックテストで例外が発生
        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert result == (0.0,)

    def test_evaluate_individual_multi_objective_exception(self):
        """多目的最適化例外のテスト"""
        mock_individual = [1, 2, 3, 4, 5]

        self.mock_backtest_service.run_backtest.side_effect = Exception("Test error")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "sharpe_ratio"]

        result = self.evaluator.evaluate_individual(mock_individual, ga_config)

        assert result == (0.0, 0.0)  # 目的数に応じた0.0が返される

    def test_extract_performance_metrics(self):
        """パフォーメンスメトリクス抽出のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "profit_factor": 1.9,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5
            },
            "equity_curve": [100, 110, 105, 120, 115],
            "trade_history": [
                {"size": 1, "pnl": 10},
                {"size": -1, "pnl": -5},
                {"size": 1, "pnl": 15}
            ],
            "start_date": "2024-01-01",
            "end_date": "2024-12-19"
        }

        metrics = self.evaluator._extract_performance_metrics(backtest_result)

        assert metrics["total_return"] == 0.15
        assert metrics["sharpe_ratio"] == 1.2
        assert metrics["max_drawdown"] == 0.08
        assert metrics["win_rate"] == 0.55
        assert "ulcer_index" in metrics
        assert "trade_frequency_penalty" in metrics

    def test_extract_performance_metrics_invalid_values(self):
        """無効な値の処理テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float('inf'),  # 無限大
                "sharpe_ratio": None,  # None
                "max_drawdown": -0.1,  # 負のドローダウン
                "win_rate": "invalid"  # 無効な型
            },
            "equity_curve": [],
            "trade_history": []
        }

        metrics = self.evaluator._extract_performance_metrics(backtest_result)

        # 無効な値が適切に処理されているか確認
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0  # 負の値は0に修正
        assert metrics["win_rate"] == 0.0

    def test_calculate_fitness_zero_trades(self):
        """取引回数0のフィットネス計算テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 0  # 取引なし
            },
            "equity_curve": [],
            "trade_history": []
        }

        ga_config = GAConfig()
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }

        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.1  # 取引回数0の特別処理

    def test_calculate_fitness_constraints(self):
        """フィットネス制約のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 0.2,  # 最低シャープレシオ未満
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 5
            },
            "equity_curve": [],
            "trade_history": []
        }

        ga_config = GAConfig()
        ga_config.fitness_constraints = {
            "min_sharpe_ratio": 0.5,
            "min_trades": 3,
            "max_drawdown_limit": 0.15
        }
        ga_config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }

        fitness = self.evaluator._calculate_fitness(backtest_result, ga_config)
        assert fitness == 0.0  # シャープレシオが最低要件を満たしていない

    def test_calculate_long_short_balance(self):
        """ロング・ショートバランス計算のテスト"""
        # ロングとショートがバランスしている取引履歴
        trade_history = [
            {"size": 1, "pnl": 10},   # ロング
            {"size": -1, "pnl": 5},   # ショート
            {"size": 1, "pnl": 15},   # ロング
            {"size": -1, "pnl": 10}   # ショート
        ]

        backtest_result = {"trade_history": trade_history}

        balance = self.evaluator._calculate_long_short_balance(backtest_result)
        assert 0.0 <= balance <= 1.0

    def test_calculate_long_short_balance_no_trades(self):
        """取引なしのバランス計算テスト"""
        backtest_result = {"trade_history": []}
        balance = self.evaluator._calculate_long_short_balance(backtest_result)
        assert balance == 0.5  # デフォルトの中立スコア

    def test_calculate_multi_objective_fitness(self):
        """多目的フィットネス計算のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55,
                "profit_factor": 1.9,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
                "total_trades": 1
            },
            "equity_curve": [],
            "trade_history": [{"id": 1, "type": "long", "entry_price": 100, "exit_price": 115, "pnl": 0.15}]
        }

        ga_config = GAConfig()
        ga_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]

        result = self.evaluator._calculate_multi_objective_fitness(backtest_result, ga_config)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result[0] == 0.15  # total_return
        assert result[1] == 1.2   # sharpe_ratio
        assert result[2] == 0.08  # max_drawdown
        assert result[3] == 0.55  # win_rate

    def test_calculate_multi_objective_fitness_unknown_objective(self):
        """未知の目的のテスト"""
        backtest_result = {"performance_metrics": {"total_trades": 1}}
        ga_config = GAConfig()
        ga_config.objectives = ["unknown_objective"]

        result = self.evaluator._calculate_multi_objective_fitness(backtest_result, ga_config)

        assert result == (0.0,)  # 未知の目的は0.0