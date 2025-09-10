"""
individual_evaluator.py のテストモジュール

IndividualEvaluatorクラスのユニットテスト。
バックテストサービスなどはmockを使用してテストを軽量に保つ。
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.config import GAConfig


class TestIndividualEvaluator:
    """IndividualEvaluatorクラスのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """BacktestServiceモック"""
        return MagicMock()

    @pytest.fixture
    def evaluator(self, mock_backtest_service):
        """IndividualEvaluatorインスタンス"""
        evaluator = IndividualEvaluator(mock_backtest_service)
        evaluator.set_backtest_config({"timeframe": "1h", "symbol": "BTC/USDT"})
        return evaluator

    @pytest.fixture
    def sample_config(self):
        """サンプルGA設定"""
        return GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.2,
            objectives=["total_return", "sharpe_ratio"],
            enable_multi_objective=False,
            fitness_weights={
                "total_return": 0.3,
                "sharpe_ratio": 0.4,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
                "balance_score": 0.1
            },
            fitness_constraints={
                "min_trades": 10,
                "max_drawdown_limit": 0.5,
                "min_sharpe_ratio": 0.0
            }
        )

    @pytest.fixture
    def sample_backtest_result(self):
        """サンプルバックテスト結果"""
        return {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "sortino_ratio": 0.8,
                "calmar_ratio": 2.0,
                "total_trades": 20
            },
            "trade_history": [
                {"size": 1.0, "pnl": 10.0},
                {"size": -1.0, "pnl": 15.0},
                {"size": 1.0, "pnl": -5.0},
                {"size": -1.0, "pnl": -2.0},
            ]
        }

    def test_initialization(self, evaluator, mock_backtest_service):
        """初期化テスト"""
        assert evaluator.backtest_service == mock_backtest_service
        assert evaluator._fixed_backtest_config == {"timeframe": "1h", "symbol": "BTC/USDT"}

    def test_evaluate_individual_success(self, evaluator, mock_backtest_service, sample_config, sample_backtest_result):
        """evaluate_individualの正常テスト"""
        from unittest.mock import patch, MagicMock

        # Mock設定
        with patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_gene_serializer_class:
            mock_gene_serializer = mock_gene_serializer_class.return_value
            mock_gene = MagicMock()
            mock_gene_serializer.from_list.return_value = mock_gene
            mock_backtest_service.run_backtest.return_value = sample_backtest_result

            individual = [0.1, 0.2, 0.3, 0.4, 0.5]

            # テスト実行
            result = evaluator.evaluate_individual(individual, sample_config)

            # 検証
            mock_backtest_service.run_backtest.assert_called_once()
            assert isinstance(result, tuple)
            assert len(result) == 1  # 単一目的

    def test_evaluate_individual_multi_objective(self, evaluator, mock_backtest_service, sample_config, sample_backtest_result):
        """evaluate_individualの多目的テスト"""
        from unittest.mock import patch, MagicMock

        # Mock設定
        sample_config.enable_multi_objective = True
        with patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_gene_serializer_class:
            mock_gene_serializer = mock_gene_serializer_class.return_value
            mock_gene = MagicMock()
            mock_gene_serializer.from_list.return_value = mock_gene
            mock_backtest_service.run_backtest.return_value = sample_backtest_result

            individual = [0.1, 0.2, 0.3, 0.4, 0.5]

            # テスト実行
            result = evaluator.evaluate_individual(individual, sample_config)

            # 検証
            assert isinstance(result, tuple)
            assert len(result) == len(sample_config.objectives)  # 多目的

    def test_evaluate_individual_with_error(self, evaluator, mock_backtest_service, sample_config):
        """evaluate_individualのエラーテスト"""
        from unittest.mock import patch, MagicMock

        # Mock設定
        with patch('app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_gene_serializer_class:
            mock_gene_serializer = mock_gene_serializer_class.return_value
            mock_gene = MagicMock()
            mock_gene_serializer.from_list.return_value = mock_gene
            mock_backtest_service.run_backtest.side_effect = Exception("Test error")

            individual = [0.1, 0.2, 0.3, 0.4, 0.5]

            # テスト実行
            result = evaluator.evaluate_individual(individual, sample_config)

            # エラー時のデフォルト値を検証
            assert isinstance(result, tuple)
            assert result[0] == 0.0

    def test_extract_performance_metrics_normal(self, evaluator, sample_backtest_result):
        """_extract_performance_metricsの正常テスト"""
        metrics = evaluator._extract_performance_metrics(sample_backtest_result)

        assert metrics["total_return"] == 0.1
        assert metrics["sharpe_ratio"] == 1.2
        assert metrics["max_drawdown"] == 0.05

    def test_extract_performance_metrics_missing_keys(self, evaluator):
        """メトリクスがない場合のテスト"""
        backtest_result = {"performance_metrics": {}}

        metrics = evaluator._extract_performance_metrics(backtest_result)

        assert metrics["total_return"] == 0.0
        assert metrics["max_drawdown"] == 1.0  # デフォルト

    def test_extract_performance_metrics_invalid_values(self, evaluator):
        """無効な値の処理テスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float('inf'),
                "max_drawdown": float('nan'),
                "total_trades": "not_number"
            }
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)

        assert metrics["total_return"] == 0.0  # inf -> 0.0
        assert metrics["max_drawdown"] == 1.0  # nan -> 1.0
        assert metrics["total_trades"] == 0  # invalid -> 0

    def test_calculate_fitness_normal(self, evaluator, sample_config, sample_backtest_result):
        """_calculate_fitnessの正常テスト"""
        fitness = evaluator._calculate_fitness(sample_backtest_result, sample_config)

        assert isinstance(fitness, float)
        assert fitness >= 0.0

    def test_calculate_fitness_zero_trades(self, evaluator, sample_config):
        """取引回数0の場合のテスト"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.05,
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "total_trades": 0
            },
            "trade_history": []
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)

        # 取引回数0の場合は低いフィットネス値
        assert fitness == 0.1

    def test_calculate_fitness_constraints_violated(self, evaluator, sample_config):
        """制約違反時のテスト"""
        config = GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.2,
            fitness_constraints={
                "min_trades": 30,
                "max_drawdown_limit": 0.1,
                "min_sharpe_ratio": 1.5
            }
        )

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.05,
                "sharpe_ratio": 1.0,  # min_sharpe_ratio違反
                "max_drawdown": 0.2,   # max_drawdown_limit違反
                "win_rate": 0.5,
                "total_trades": 20     # min_trades違反
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, config)

        assert fitness == 0.0  # 制約違反で0

    def test_calculate_long_short_balance_normal(self, evaluator, sample_backtest_result):
        """_calculate_long_short_balanceの正常テスト"""
        balance = evaluator._calculate_long_short_balance(sample_backtest_result)

        assert 0.0 <= balance <= 1.0

    def test_calculate_long_short_balance_no_trades(self, evaluator):
        """取引がない場合のテスト"""
        backtest_result = {"trade_history": []}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        assert balance == 0.5

    def test_calculate_long_short_balance_all_long(self, evaluator):
        """全てロングの場合のテスト"""
        backtest_result = {
            "trade_history": [
                {"size": 1.0, "pnl": 10.0},
                {"size": 1.0, "pnl": 20.0}
            ]
        }

        balance = evaluator._calculate_long_short_balance(backtest_result)

        assert balance < 1.0  # 不均衡なので1.0未満

    def test_calculate_multi_objective_fitness_normal(self, evaluator, sample_config, sample_backtest_result):
        """_calculate_multi_objective_fitnessの正常テスト"""
        config = sample_config
        config.enable_multi_objective = True
        config.objectives = ["total_return", "sharpe_ratio"]

        fitness_values = evaluator._calculate_multi_objective_fitness(
            sample_backtest_result, config
        )

        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == 2

    def test_calculate_multi_objective_fitness_zero_trades(self, evaluator, sample_config):
        """多目的、取引回数0の場合のテスト"""
        config = sample_config
        config.enable_multi_objective = True
        config.objectives = ["total_return", "sharpe_ratio"]

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.05,
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.1,
                "total_trades": 0
            }
        }

        fitness_values = evaluator._calculate_multi_objective_fitness(
            backtest_result, config
        )

        # 全ての目的で低い値が設定される
        assert all(val == 0.1 for val in fitness_values)

    def test_select_timeframe_config_normal(self, evaluator):
        """_select_timeframe_configのテスト"""
        config = {"timeframe": "4h", "symbol": "ETH/USDT"}

        result = evaluator._select_timeframe_config(config)

        assert result == config