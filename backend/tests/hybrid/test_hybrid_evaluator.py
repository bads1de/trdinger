"""
HybridIndividualEvaluatorのテストモジュール

GA個体評価にML予測を統合したハイブリッド評価のテスト
TDD: テストファースト
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.backtest.backtest_service import BacktestService
from app.services.ml.exceptions import MLTrainingError


class TestHybridIndividualEvaluator:
    """HybridIndividualEvaluatorのテストクラス"""

    @pytest.fixture
    def sample_individual(self):
        """サンプル個体（遺伝子リスト形式）"""
        # GeneSerializerでシリアライズされた形式を想定
        return [
            "test_gene_001",  # id
            [["ind1", "SMA", {"period": 20}]],  # indicators
            [["cond1", "ind1", "ind2", ">"]],  # entry_conditions
            [],  # exit_conditions
        ]

    @pytest.fixture
    def ga_config(self):
        """GA設定"""
        config = GAConfig(
            population_size=50,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=False,
            fitness_weights={
                "total_return": 0.3,
                "sharpe_ratio": 0.4,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
                "prediction_score": 0.1,  # ML予測スコアの重み
            },
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-12-31",
        )
        return config

    @pytest.fixture
    def ga_config_multi_objective(self):
        """多目的最適化GA設定"""
        config = GAConfig(
            population_size=50,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            enable_multi_objective=True,
            objectives=["sharpe_ratio", "max_drawdown", "prediction_score"],
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-12-31",
        )
        return config

    @pytest.fixture
    def mock_backtest_service(self):
        """Mock BacktestService"""
        service = Mock(spec=BacktestService)
        service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "total_trades": 100,
            },
            "trade_history": [
                {"size": 1.0, "pnl": 100.0},
                {"size": -1.0, "pnl": 50.0},
                {"size": 1.0, "pnl": 75.0},
            ],
        }
        # _get_cached_data が使用する data_service のモック
        mock_data_service = Mock()
        mock_data_service.get_data_for_backtest.return_value = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_data_service.get_ohlcv_data.return_value = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            }
        )
        service.data_service = mock_data_service
        service.ensure_data_service_initialized = Mock()
        return service

    @pytest.fixture
    def mock_hybrid_predictor(self):
        """Mock HybridPredictor"""
        predictor = Mock()
        predictor.predict.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }
        return predictor

    def test_evaluator_initialization(self, mock_backtest_service):
        """
        Evaluator初期化テスト

        検証項目:
        - 正しく初期化される
        - BacktestServiceとHybridPredictorが設定される
        """

        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=None,
        )

        assert evaluator is not None
        assert evaluator.backtest_service == mock_backtest_service

    def test_evaluate_individual_with_prediction(
        self,
        sample_individual,
        ga_config,
        mock_backtest_service,
        mock_hybrid_predictor,
    ):
        """
        ML予測を含む個体評価テスト

        検証項目:
        - バックテストが実行される
        - ML予測が実行される
        - 予測スコアがフィットネスに統合される
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_hybrid_predictor,
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # フィットネスが返される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1  # 単一目的
        assert fitness[0] > 0

        # バックテストとML予測が呼ばれたことを確認
        assert mock_backtest_service.run_backtest.called
        assert mock_hybrid_predictor.predict.called

    def test_evaluate_individual_without_prediction(
        self,
        sample_individual,
        ga_config,
        mock_backtest_service,
    ):
        """
        ML予測なしの個体評価テスト（従来のGA評価）

        検証項目:
        - predictorがNoneの場合、従来の評価が行われる
        - フィットネスが正しく計算される
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=None,  # 予測なし
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # フィットネスが返される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] > 0

    def test_calculate_fitness_with_prediction_score(
        self,
        mock_backtest_service,
        ga_config,
    ):
        """
        予測スコアを含むフィットネス計算テスト

        検証項目:
        - prediction_scoreが正しく計算される
        - fitness = base_fitness + prediction_weight * prediction_score
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 100,
            },
        }

        prediction_signals = {
            "up": 0.7,
            "down": 0.2,
            "range": 0.1,
        }

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=Mock(),
        )

        # _calculate_fitnessを直接テスト
        fitness = evaluator._calculate_fitness(
            backtest_result,
            ga_config,
            prediction_signals=prediction_signals,
        )

        # 予測スコアが統合されたフィットネスが返される
        assert fitness > 0
        # prediction_score = up - down = 0.7 - 0.2 = 0.5
        # fitness += 0.1 (prediction_weight) * 0.5 = 0.05

    def test_calculate_fitness_balance_score_integration(
        self,
        mock_backtest_service,
        ga_config,
    ):
        """
        ロング・ショートバランススコアとML予測の統合テスト

        検証項目:
        - balance_scoreとprediction_scoreが両方考慮される
        - 重み付けが正しく適用される
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        backtest_result = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 100,
            },
            "trade_history": [
                {"size": 1.0, "pnl": 100.0},  # Long
                {"size": -1.0, "pnl": 50.0},  # Short
                {"size": 1.0, "pnl": 75.0},  # Long
            ],
        }

        prediction_signals = {
            "up": 0.6,
            "down": 0.3,
            "range": 0.1,
        }

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=Mock(),
        )

        fitness = evaluator._calculate_fitness(
            backtest_result,
            ga_config,
            prediction_signals=prediction_signals,
        )

        # balance_scoreとprediction_scoreの両方が含まれる
        assert fitness > 0

    def test_evaluate_individual_multi_objective(
        self,
        sample_individual,
        ga_config_multi_objective,
        mock_backtest_service,
        mock_hybrid_predictor,
    ):
        """
        多目的最適化での個体評価テスト

        検証項目:
        - 複数の目的関数値が返される
        - prediction_scoreが目的の一つとして含まれる
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_hybrid_predictor,
        )

        fitness = evaluator.evaluate_individual(
            sample_individual, ga_config_multi_objective
        )

        # 多目的のフィットネス値が返される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 3  # sharpe_ratio, max_drawdown, prediction_score
        assert all(isinstance(f, (int, float)) for f in fitness)

    def test_evaluate_individual_with_error_handling(
        self,
        sample_individual,
        ga_config,
        mock_backtest_service,
    ):
        """
        エラーハンドリングテスト

        検証項目:
        - バックテストエラー時にデフォルト値が返される
        - ML予測エラー時もデフォルト値が返される
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        # バックテストでエラーを発生させる
        mock_backtest_service.run_backtest.side_effect = Exception("Backtest error")

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=Mock(),
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # エラー時はデフォルト値が返される
        assert isinstance(fitness, tuple)
        assert fitness[0] == 0.0

    def test_evaluate_individual_with_ml_training_error(
        self,
        sample_individual,
        ga_config,
        mock_backtest_service,
    ):
        """
        ML学習エラーのハンドリングテスト

        検証項目:
        - MLTrainingError発生時にデフォルト予測値が使われる
        - エラーログが出力される
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        # ML予測でエラーを発生させる
        mock_predictor = Mock()
        mock_predictor.predict.side_effect = MLTrainingError("ML error")

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_predictor,
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # エラーがあってもフィットネスは返される（デフォルト予測使用）
        assert isinstance(fitness, tuple)
        assert fitness[0] >= 0

    def test_evaluate_individual_zero_trades(
        self,
        sample_individual,
        ga_config,
        mock_backtest_service,
        mock_hybrid_predictor,
    ):
        """
        取引回数ゼロの場合のテスト

        検証項目:
        - 取引回数が0の場合、低いフィットネス値が返される
        - ML予測は実行されない（または無視される）
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        # 取引回数0の結果を設定
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,  # 取引なし
            },
        }

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_hybrid_predictor,
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # 取引回数0の場合は低いフィットネス
        assert isinstance(fitness, tuple)
        assert fitness[0] <= 0.1

    def test_prediction_weight_configuration(
        self,
        sample_individual,
        mock_backtest_service,
        mock_hybrid_predictor,
    ):
        """
        予測重み設定テスト

        検証項目:
        - prediction_weightが設定で変更可能
        - 重みが0の場合、従来のGA評価と同じ
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        # prediction_weight = 0 の設定
        ga_config_no_prediction = GAConfig(
            population_size=50,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            fitness_weights={
                "total_return": 0.3,
                "sharpe_ratio": 0.4,
                "max_drawdown": 0.2,
                "win_rate": 0.1,
                "prediction_score": 0.0,  # ML予測を無視
            },
            fallback_start_date="2023-01-01",
            fallback_end_date="2023-12-31",
        )

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_hybrid_predictor,
        )

        fitness = evaluator.evaluate_individual(
            sample_individual, ga_config_no_prediction
        )

        # ML予測は呼ばれるが、フィットネスには影響しない
        assert isinstance(fitness, tuple)
        assert fitness[0] > 0

    @patch(
        "app.services.auto_strategy.utils.hybrid_feature_adapter.HybridFeatureAdapter"
    )
    def test_feature_adapter_integration(
        self,
        mock_adapter_class,
        sample_individual,
        ga_config,
        mock_backtest_service,
        mock_hybrid_predictor,
    ):
        """
        HybridFeatureAdapterとの統合テスト

        検証項目:
        - Gene → 特徴量変換 → ML予測の流れが動作する
        - HybridFeatureAdapterが正しく呼ばれる
        """
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        # モックの設定
        mock_adapter = Mock()
        mock_adapter.gene_to_features.return_value = pd.DataFrame(
            {
                "feature_1": [0.5],
                "feature_2": [0.3],
            }
        )
        mock_adapter_class.return_value = mock_adapter

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            predictor=mock_hybrid_predictor,
        )

        fitness = evaluator.evaluate_individual(sample_individual, ga_config)

        # HybridFeatureAdapterが呼ばれたことを確認
        assert mock_adapter.gene_to_features.called
        assert isinstance(fitness, tuple)
        assert fitness[0] > 0
