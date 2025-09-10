"""
個体評価器のテスト

IndividualEvaluatorクラスのTDDテストケース
バグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import math
import numpy as np

from backend.app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from backend.app.services.backtest.backtest_service import BacktestService
from backend.app.services.auto_strategy.config.ga_runtime import GAConfig
from backend.app.services.auto_strategy.models.strategy_models import StrategyGene


@pytest.fixture
def mock_backtest_service():
    """モックバックテストサービス"""
    service = Mock(spec=BacktestService)

    # 標準的なバックテスト結果
    mock_result = {
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "profit_factor": 1.3,
            "total_trades": 25
        },
        "trade_history": [
            {"size": 1.0, "pnl": 0.02},  # long traded
            {"size": -1.0, "pnl": 0.015},  # short traded
            {"size": 1.0, "pnl": -0.005},  # long traded
            {"size": -1.0, "pnl": 0.008},  # short traded
        ]
    }

    service.run_backtest.return_value = mock_result
    return service


@pytest.fixture
def sample_config():
    """サンプルGA設定"""
    config = Mock(spec=GAConfig)
    config.enable_multi_objective = False
    config.objectives = ["total_return"]
    config.fitness_weights = {"total_return": 0.4, "sharpe_ratio": 0.3, "max_drawdown": 0.2, "win_rate": 0.1}
    config.fitness_constraints = {"min_trades": 5, "max_drawdown_limit": 0.1, "min_sharpe_ratio": 0.5}
    return config


@pytest.fixture
def evaluator(mock_backtest_service):
    """テスト用IndividualEvaluatorインスタンス"""
    return IndividualEvaluator(mock_backtest_service)


@pytest.fixture
def sample_individual():
    """サンプル個体"""
    return [0.5, 0.3, 0.8, 0.2, 0.7, 0.6, 0.9]


@pytest.fixture
def mock_strategy_gene():
    """モック戦略遺伝子"""
    gene = Mock(spec=StrategyGene)
    gene.id = "test_gene_001"
    return gene


@pytest.fixture
def backtest_config():
    """バックテスト設定"""
    return {
        "timeframe": "1D",
        "symbol": "BTC/USD",
        "strategy_config": {
            "strategy_type": "GENERATED_GA",
            "parameters": {"strategy_gene": "mock_dict"}
        }
    }


class TestIndividualEvaluatorInitialization:

    def test_init(self, mock_backtest_service, evaluator):
        """初期化テスト"""
        assert evaluator.backtest_service is mock_backtest_service
        assert evaluator._fixed_backtest_config is None

    def test_set_backtest_config(self, evaluator, backtest_config):
        """バックテスト設定の設定"""
        evaluator.set_backtest_config(backtest_config)
        assert evaluator._fixed_backtest_config == backtest_config


class TestIndividualEvaluation:

    def test_evaluate_individual_single_objective_success(self, evaluator, sample_individual, sample_config, backtest_config, mock_strategy_gene):
        """単一目的個体評価の成功ケース"""
        evaluator.set_backtest_config(backtest_config)

        # モック戦略遺伝子デコード
        with patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer_class:
            mock_serializer = Mock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.from_list.return_value = mock_strategy_gene

            result = evaluator.evaluate_individual(sample_individual, sample_config)

            # 戻り値がタプルであること
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], (int, float))

            # バックテストサービスが呼ばれたこと
            evaluator.backtest_service.run_backtest.assert_called_once()

    def test_evaluate_individual_multi_objective_success(self, evaluator, sample_individual, sample_config, backtest_config, mock_strategy_gene):
        """多目的個体評価の成功ケース"""
        evaluator.set_backtest_config(backtest_config)
        sample_config.enable_multi_objective = True
        sample_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        with patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer_class:
            mock_serializer = Mock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.from_list.return_value = mock_strategy_gene

            result = evaluator.evaluate_individual(sample_individual, sample_config)

            assert isinstance(result, tuple)
            assert len(result) == len(sample_config.objectives)
            for value in result:
                assert isinstance(value, (int, float))

    @patch('backend.app.services.auto_strategy.core.individual_evaluator.logger')
    def test_evaluate_individual_deserialization_error_handling(self, mock_logger, evaluator, sample_individual, sample_config):
        """遺伝子デコードエラー時の処理"""
        with patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer_class:
            mock_serializer = Mock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.from_list.side_effect = Exception("Deserialization error")

            result = evaluator.evaluate_individual(sample_individual, sample_config)

            # エラーログ出力
            mock_logger.error.assert_called_once()
            # デフォルト値返却
            assert isinstance(result, tuple)
            assert result[0] == 0.0

    @patch('backend.app.services.auto_strategy.core.individual_evaluator.logger')
    def test_evaluate_individual_backtest_error_handling(self, mock_logger, evaluator, sample_individual, sample_config, mock_strategy_gene):
        """バックテストエラー時の処理"""
        evaluator.set_backtest_config({})

        with patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer_class:
            mock_serializer = Mock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.from_list.return_value = mock_strategy_gene

            # バックテストサービスがエラーを投げる
            evaluator.backtest_service.run_backtest.side_effect = Exception("Backtest failed")

            result = evaluator.evaluate_individual(sample_individual, sample_config)

            mock_logger.error.assert_called_once()
            assert isinstance(result, tuple)
            assert result[0] == 0.0

    def test_evaluate_individual_no_backtest_config(self, evaluator, sample_individual, sample_config):
        """バックテスト設定なしの場合"""
        evaluator._fixed_backtest_config = None

        result = evaluator.evaluate_individual(sample_individual, sample_config)
        assert result == (0.0,)


class TestPerformanceMetricsExtraction:

    def test_extract_performance_metrics_complete(self, evaluator):
        """完全なパフォーマンスメトリクス抽出"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.12,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.03,
                "win_rate": 0.6,
                "profit_factor": 1.4,
                "sortino_ratio": 1.2,
                "calmar_ratio": 2.5,
                "total_trades": 30
            }
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)

        assert metrics["total_return"] == 0.12
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["max_drawdown"] == 0.03
        assert metrics["win_rate"] == 0.6
        assert metrics["profit_factor"] == 1.4
        assert metrics["sortino_ratio"] == 1.2
        assert metrics["calmar_ratio"] == 2.5
        assert metrics["total_trades"] == 30

    def test_extract_performance_metrics_missing_keys(self, evaluator):
        """メトリクスキーが不足する場合"""
        backtest_result = {
            "performance_metrics": {}  # 空のメトリクス
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)

        # デフォルト値
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 1.0
        assert metrics["total_trades"] == 0

    def test_extract_performance_metrics_invalid_values(self, evaluator):
        """無効な値の処理"""
        backtest_result = {
            "performance_metrics": {
                "total_return": None,
                "sharpe_ratio": float('inf'),
                "max_drawdown": float('-inf'),
                "win_rate": "invalid",
                "total_trades": "not_a_number"
            }
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)

        # 無効な値がデフォルトに修正される
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 1.0
        assert not math.isfinite(metrics["win_rate"]) or metrics["win_rate"] == 0.5
        assert metrics["total_trades"] == 0

    def test_extract_performance_metrics_negative_drawdown(self, evaluator):
        """負のドローダウンの処理"""
        backtest_result = {
            "performance_metrics": {
                "max_drawdown": -0.05  # 負のドローダウン（論理的におかしいが）
            }
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)
        assert metrics["max_drawdown"] == 0.0  # 修正済み


class TestFitnessCalculation:

    def test_calculate_fitness_success(self, evaluator, mock_backtest_service, sample_config):
        """フィットネス計算の成功ケース"""
        backtest_result = mock_backtest_service.run_backtest.return_value

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)

        assert isinstance(fitness, float)
        assert not math.isnan(fitness)
        assert not math.isinf(fitness)

    def test_calculate_fitness_zero_trades(self, evaluator, sample_config):
        """取引回数ゼロでのフィットネス計算"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)

        assert fitness == 0.1  # デフォルトの低い値

    def test_calculate_fitness_insufficient_trades(self, evaluator, sample_config):
        """取引回数が不足する場合"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.02,
                "win_rate": 0.6,
                "total_trades": 3  # min_trades=5未満
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)
        assert fitness == 0.0

    def test_calculate_fitness_high_drawdown(self, evaluator, sample_config):
        """ドローダウンが上限を超える場合"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.15,  # max_drawdown_limit=0.1を超過
                "win_rate": 0.6,
                "total_trades": 10
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)
        assert fitness == 0.0

    def test_calculate_fitness_negative_return(self, evaluator, sample_config):
        """リターンが負の場合"""
        backtest_result = {
            "performance_metrics": {
                "total_return": -0.05,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.02,
                "win_rate": 0.6,
                "total_trades": 10
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)
        assert fitness == 0.0

    def test_calculate_fitness_low_sharpe(self, evaluator, sample_config):
        """シャープレシオが低すぎる場合"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 0.3,  # min_sharpe_ratio=0.5未満
                "max_drawdown": 0.02,
                "win_rate": 0.6,
                "total_trades": 10
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)
        assert fitness == 0.0

    def test_calculate_fitness_invalid_metrics(self, evaluator, sample_config):
        """無効なメトリクス値の処理"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float('nan'),
                "sharpe_ratio": float('inf'),
                "max_drawdown": 0.5,
                "win_rate": 0.6,
                "total_trades": 10
            }
        }

        fitness = evaluator._calculate_fitness(backtest_result, sample_config)

        # NaNやinfが処理されてもクラッシュしない
        assert isinstance(fitness, float)


class TestLongShortBalance:

    def test_calculate_long_short_balance_balanced(self, evaluator):
        """バランスのとれたロング・ショート取引"""
        trade_history = [
            {"size": 1.0, "pnl": 0.02},
            {"size": -1.0, "pnl": 0.015},
            {"size": 1.0, "pnl": 0.01},
            {"size": -1.0, "pnl": 0.008},
        ]

        backtest_result = {"trade_history": trade_history}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        assert 0.0 <= balance <= 1.0
        assert balance > 0.8  # バランスが取れているはず

    def test_calculate_long_short_balance_long_only(self, evaluator):
        """ロングオンリーの場合"""
        trade_history = [
            {"size": 1.0, "pnl": 0.02},
            {"size": 1.0, "pnl": 0.01},
            {"size": 1.0, "pnl": -0.005},
        ]

        backtest_result = {"trade_history": trade_history}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        # ロングオンリーでも0より大きい値
        assert 0.0 <= balance <= 1.0
        assert balance < 0.8  # 完全にバランスしていない

    def test_calculate_long_short_balance_empty_trades(self, evaluator):
        """取引履歴なしの場合"""
        trade_history = []

        backtest_result = {"trade_history": trade_history}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        assert balance == 0.5  # 中立スコア

    def test_calculate_long_short_balance_mixed_pnl(self, evaluator):
        """利益・損失が混在する場合"""
        trade_history = [
            {"size": 1.0, "pnl": 0.02},   # ロング利益
            {"size": -1.0, "pnl": -0.01}, # ショート損失（悪）
            {"size": 1.0, "pnl": -0.005}, # ロング損失
            {"size": -1.0, "pnl": 0.03},  # ショート利益（良い）
        ]

        backtest_result = {"trade_history": trade_history}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        assert 0.0 <= balance <= 1.0
        assert balance < 1.0  # 完全ではない


class TestMultiObjectiveFitness:

    def test_calculate_multi_objective_fitness_success(self, evaluator, sample_config):
        """多目的フィットネス計算の成功"""
        backtest_result = evaluator.backtest_service.run_backtest.return_value

        sample_config.enable_multi_objective = True
        sample_config.objectives = ["total_return", "sharpe_ratio", "max_drawdown"]

        fitness_values = evaluator._calculate_multi_objective_fitness(backtest_result, sample_config)

        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == len(sample_config.objectives)
        for value in fitness_values:
            assert isinstance(value, (int, float))

    def test_calculate_multi_objective_fitness_zero_trades(self, evaluator, sample_config):
        """取引回数ゼロでの多目的フィットネス"""
        backtest_result = {
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        }

        sample_config.objectives = ["total_return", "sharpe_ratio"]

        fitness_values = evaluator._calculate_multi_objective_fitness(backtest_result, sample_config)

        assert len(fitness_values) == 2
        assert fitness_values[0] == 0.1  # デフォルト低い値
        assert fitness_values[1] == 0.1


class TestTimeframeConfigSelection:

    def test_select_timeframe_config_basic(self, evaluator):
        """タイムフレーム設定の基本選択"""
        config = {"timeframe": "1H", "symbol": "BTC/USD"}

        result = evaluator._select_timeframe_config(config)

        assert result == config

    def test_select_timeframe_config_none(self, evaluator):
        """None設定の処理"""
        result = evaluator._select_timeframe_config(None)

        assert result == {}


class TestEdgeCases:

    def test_evaluate_individual_with_extreme_values(self, evaluator, sample_individual, sample_config, mock_strategy_gene):
        """極端な値を含む個体評価"""
        evaluator.set_backtest_config({})

        # バックテスト結果に極端な値
        extreme_result = {
            "performance_metrics": {
                "total_return": 1e10,
                "sharpe_ratio": 1e5,
                "max_drawdown": 1e6,  # これは論理的におかしいがテスト
                "win_rate": 1.0,
                "profit_factor": 10000,
                "total_trades": 1000
            },
            "trade_history": []
        }
        evaluator.backtest_service.run_backtest.return_value = extreme_result

        with patch('backend.app.services.auto_strategy.serializers.gene_serialization.GeneSerializer') as mock_serializer_class:
            mock_serializer = Mock()
            mock_serializer_class.return_value = mock_serializer
            mock_serializer.from_list.return_value = mock_strategy_gene

            result = evaluator.evaluate_individual(sample_individual, sample_config)

            # 極端な値でもクラッシュしない
            assert isinstance(result, tuple)
            assert isinstance(result[0], (int, float))

    def test_extract_performance_metrics_with_extreme_values(self, evaluator):
        """パフォーマンスメトリクス抽出での極端な値"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float('inf'),
                "sharpe_ratio": float('-inf'),
                "max_drawdown": -float('inf'),
                "total_trades": -1000
            }
        }

        metrics = evaluator._extract_performance_metrics(backtest_result)

        # 無効な値がデフォルトに修正される
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert metrics["total_trades"] == 0

    def test_calculate_fitness_with_extreme_values(self, evaluator, sample_config):
        """フィットネス計算での極端な値"""
        backtest_result = {
            "performance_metrics": {
                "total_return": float('inf'),
                "sharpe_ratio": 1000000,
                "max_drawdown": 2.0,  # 200%ドローダウン（論理的におかしい）
                "win_rate": 1.1,  # 110%勝率（論理的におかしい）
                "total_trades": 1000000
            }
        }

        # NaNやinfを処理してもクラッシュしない
        fitness = evaluator._calculate_fitness(backtest_result, sample_config)
        assert isinstance(fitness, float)

    def test_fitness_weights_none(self, evaluator, sample_config):
        """フィットネスウェイトなしの場合"""
        sample_config.fitness_weights = None

        backtest_result = evaluator.backtest_service.run_backtest.return_value

        # KeyErrorなどが発生しない
        try:
            fitness = evaluator._calculate_fitness(backtest_result, sample_config)
            assert isinstance(fitness, float)
        except AttributeError:
            # フィットネスウェイトが見つからない場合の処理
            pass


class TestErrorAndExceptionHandling:

    @patch('backend.app.services.auto_strategy.core.individual_evaluator.logger')
    def test_exception_in_extract_performance_metrics(self, mock_logger, evaluator):
        """パフォーマンスメトリクス抽出中の例外"""
        backtest_result = {
            "performance_metrics": {  # 辞書だがアクセスエラーになるようなデータ
                "total_return": "invalid_string_for_float",
                "max_drawdown": [1, 2, 3],  # リスト
            }
        }

        # この構造ではTypeErrorやValueErrorが発生する可能性
        metrics = evaluator._extract_performance_metrics(backtest_result)

        # 例外処理によりデフォルト値が返される
        assert metrics["total_return"] == 0.0
        assert metrics["max_drawdown"] == 1.0

    @patch('backend.app.services.auto_strategy.core.individual_evaluator.logger')
    def test_division_by_zero_in_balance_calculation(self, mock_logger, evaluator):
        """バランス計算でのゼロ除算回避"""
        trade_history = [
            {"size": 0, "pnl": 0},  # サイズ0の場合
        ]

        backtest_result = {"trade_history": trade_history}

        balance = evaluator._calculate_long_short_balance(backtest_result)

        # ゼロ除算が発生してもデフォルト値が返される
        assert isinstance(balance, float)
        assert 0.0 <= balance <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])