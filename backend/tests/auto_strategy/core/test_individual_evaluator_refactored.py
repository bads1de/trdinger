"""
個体評価器のリファクタリングテスト

フィットネス計算の共通化テスト
"""

import pytest
from unittest.mock import Mock, patch

from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.config.ga_runtime import GAConfig


class TestIndividualEvaluatorRefactored:
    """個体評価器のリファクタリングテスト"""

    @pytest.fixture
    def evaluator(self):
        """個体評価器インスタンスを作成"""
        backtest_service = Mock()
        return IndividualEvaluator(backtest_service)

    @pytest.fixture
    def sample_backtest_result(self):
        """サンプルバックテスト結果を作成"""
        return {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.65,
                "profit_factor": 1.2,
                "sortino_ratio": 1.8,
                "calmar_ratio": 1.5,
                "total_trades": 50
            },
            "trade_history": [
                {"size": 1.0, "pnl": 100.0},
                {"size": -1.0, "pnl": -50.0},
                {"size": 1.0, "pnl": 150.0},
                {"size": -1.0, "pnl": 75.0}
            ]
        }

    @pytest.fixture
    def ga_config(self):
        """GA設定を作成"""
        config = GAConfig()
        config.fitness_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.2,
            "win_rate": 0.1
        }
        config.enable_multi_objective = False
        config.objectives = ["total_return", "max_drawdown"]
        config.fitness_constraints = {
            "min_trades": 10,
            "max_drawdown_limit": 0.5,
            "min_sharpe_ratio": 0.5
        }
        return config

    def test_extract_performance_metrics_basic(self, evaluator, sample_backtest_result):
        """基本的なパフォーマンスメトリクス抽出"""
        metrics = evaluator._extract_performance_metrics(sample_backtest_result)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics

        assert metrics["total_return"] == 0.15
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["max_drawdown"] == 0.1
        assert metrics["win_rate"] == 0.65
        assert metrics["total_trades"] == 50

    def test_extract_performance_metrics_missing_data(self, evaluator):
        """データ不足時のメトリクス抽出"""
        incomplete_result = {
            "performance_metrics": {
                "total_return": 0.1,
                # sharpe_ratioなし
                # max_drawdownなし
                # win_rateなし
                "total_trades": 1
            }
        }

        metrics = evaluator._extract_performance_metrics(incomplete_result)

        assert metrics["total_return"] == 0.1
        assert metrics["sharpe_ratio"] == 0.0  # デフォルト値
        assert metrics["max_drawdown"] == 1.0  # デフォルト値
        assert metrics["win_rate"] == 0.0  # デフォルト値
        assert metrics["total_trades"] == 1

    def test_calculate_fitness_uses_extracted_metrics(self, evaluator, sample_backtest_result, ga_config):
        """個体適応度計算が抽出したメトリクスを使用する"""
        with patch.object(evaluator, '_extract_performance_metrics') as mock_extract, \
             patch.object(evaluator, '_calculate_long_short_balance') as mock_balance:
            mock_extract.return_value = {
                "total_return": 0.2,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.15,
                "win_rate": 0.6,
                "total_trades": 20
            }
            mock_balance.return_value = 0.8

            fitness = evaluator._calculate_fitness(sample_backtest_result, ga_config)

            mock_extract.assert_called_once_with(sample_backtest_result)
            assert isinstance(fitness, float)
            # 正の適応度値が返されることを確認
            assert fitness > 0

    def test_calculate_multi_objective_fitness_uses_extracted_metrics(self, evaluator, sample_backtest_result, ga_config):
        """多目的適応度計算が抽出したメトリクスを使用する"""
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["total_return", "max_drawdown", "win_rate"]

        with patch.object(evaluator, '_extract_performance_metrics') as mock_extract, \
             patch.object(evaluator, '_calculate_long_short_balance') as mock_balance:
            mock_extract.return_value = {
                "total_return": 0.2,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.15,
                "win_rate": 0.6,
                "total_trades": 20
            }

            # total_return: 0.2
            # max_drawdown: 0.15 (最小化目的なのでそのまま)
            # win_rate: 0.6
            fitness_values = evaluator._calculate_multi_objective_fitness(sample_backtest_result, ga_config)

            mock_extract.assert_called_once_with(sample_backtest_result)
            assert isinstance(fitness_values, tuple)
            assert len(fitness_values) == 3

            # 戻り値が正しい順序であることを確認
            assert fitness_values[0] == 0.2  # total_return
            assert fitness_values[1] == 0.15  # max_drawdown
            assert fitness_values[2] == 0.6  # win_rate

    def test_extract_performance_metrics_handles_empty_result(self, evaluator):
        """空のバックテスト結果を適切に処理"""
        empty_result = {}

        metrics = evaluator._extract_performance_metrics(empty_result)

        # 全てのキーがあることを確認
        expected_keys = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades", "profit_factor", "sortino_ratio", "calmar_ratio"]

        for key in expected_keys:
            assert key in metrics
            # 数値型であることを確認
            assert isinstance(metrics[key], (int, float))

    def test_fitness_calculation_preserves_existing_constraints(self, evaluator, sample_backtest_result, ga_config):
        """既存の制約条件が保持される"""
        ga_config.fitness_constraints["min_trades"] = 100  # 50以上の取引を要求

        with patch.object(evaluator, '_extract_performance_metrics') as mock_extract:
            mock_extract.return_value = sample_backtest_result["performance_metrics"]

            fitness = evaluator._calculate_fitness(sample_backtest_result, ga_config)

            # 取引回数制約違反のため0を返すことを確認
            assert fitness == 0.0

    def test_extract_performance_metrics_handles_none_values(self, evaluator):
        """None値が混入しても適切に処理"""
        problematic_result = {
            "performance_metrics": {
                "total_return": None,
                "sharpe_ratio": float('inf'),
                "max_drawdown": -0.1,  # 負の値
                "win_rate": 1.0,
                "total_trades": 0
            }
        }

        metrics = evaluator._extract_performance_metrics(problematic_result)

        # デフォルト値に適切に置き換えられていることを確認
        assert isinstance(metrics["total_return"], float)
        assert metrics["total_return"] == 0.0
        assert isinstance(metrics["sharpe_ratio"], float)
        assert metrics["sharpe_ratio"] == 0.0  # inf -> 0
        assert isinstance(metrics["max_drawdown"], float)
        assert metrics["max_drawdown"] == 0.0  # 負の値 -> 0
        assert metrics["win_rate"] == 1.0
        assert metrics["total_trades"] == 0