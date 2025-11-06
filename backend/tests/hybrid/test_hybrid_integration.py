"""
ハイブリッドGA+ML統合のテストモジュール

ハイブリッドGA+ML統合機能をテストする。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config.ga import GASettings


@pytest.mark.skip(reason="GASettings schema changed - no hybrid_mode field")
class TestHybridIntegration:
    """ハイブリッドGA+ML統合テスト"""

    @pytest.fixture
    def hybrid_ga_config(self):
        """ハイブリッドモード有効のGA設定"""
        return GASettings(
            population_size=10,
            generations=2,
            crossover_rate=0.8,
            mutation_rate=0.2,
            hybrid_mode=True,
            hybrid_model_type="lightgbm",
            hybrid_automl_config={"enabled": True},
        )

    @pytest.fixture
    def sample_training_data(self):
        """サンプル学習データ"""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [105, 106, 107, 108, 109] * 20,
                "low": [95, 96, 97, 98, 99] * 20,
                "close": [102, 103, 104, 105, 106] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
                "target": [1, 0, 1, 2, 1] * 20,  # 3クラス分類
            }
        )

    @pytest.fixture
    def mock_hybrid_predictor(self):
        """Mock HybridPredictor"""
        predictor = Mock()
        predictor.predict.return_value = 0.8
        predictor.is_trained = True
        predictor.feature_columns = ["close", "volume"]
        predictor.model = Mock()
        predictor.scaler = Mock()
        predictor.scaler.transform.return_value = np.array([[1.0, 2.0]])
        return predictor

    @pytest.fixture
    def mock_feature_adapter(self):
        """Mock HybridFeatureAdapter"""
        adapter = Mock()
        adapter.adapt_features.return_value = pd.DataFrame(
            {"adapted_feature": [1, 2, 3]}
        )
        adapter.automl_config = {"enabled": True}
        return adapter

    def test_ga_engine_hybrid_initialization(
        self, hybrid_ga_config, mock_hybrid_predictor, mock_feature_adapter
    ):
        """GAエンジンのハイブリッド初期化テスト"""
        with patch(
            "backend.app.services.auto_strategy.generators.random_gene_generator.RandomGeneGenerator"
        ) as mock_generator_class:
            mock_generator = Mock()
            mock_generator_class.return_value = mock_generator

            with patch(
                "backend.app.services.auto_strategy.services.backtest.backtest_service.BacktestService"
            ) as mock_backtest_class:
                mock_backtest = Mock()
                mock_backtest_class.return_value = mock_backtest

                engine = GeneticAlgorithmEngine(
                    gene_generator=mock_generator,
                    hybrid_mode=True,
                    hybrid_predictor=mock_hybrid_predictor,
                    hybrid_feature_adapter=mock_feature_adapter,
                )

        assert engine.hybrid_mode is True
        assert engine.individual_evaluator is not None

        # HybridIndividualEvaluatorが使用されていることを確認
        from backend.app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        assert isinstance(engine.individual_evaluator, HybridIndividualEvaluator)

    def test_hybrid_individual_evaluation(
        self,
        sample_training_data,
        hybrid_ga_config,
        mock_hybrid_predictor,
        mock_feature_adapter,
    ):
        """ハイブリッド個体評価テスト"""
        from backend.app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest,
            predictor=mock_hybrid_predictor,
            feature_adapter=mock_feature_adapter,
        )

        individual = [0.1, 0.2, 0.3]

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }
        evaluator.set_backtest_config(backtest_config)

        with patch(
            "backend.app.services.auto_strategy.core.hybrid_individual_evaluator.RegimeDetector"
        ) as mock_regime_class:
            mock_regime_detector = Mock()
            mock_regime_detector.detect_regimes.return_value = np.array([0, 1, 2])
            mock_regime_class.return_value = mock_regime_detector

            fitness = evaluator.evaluate_individual(individual, hybrid_ga_config)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 1  # 単一目的

        # バックテストと予測が呼ばれたことを確認
        assert mock_backtest.run_backtest.called
        assert mock_hybrid_predictor.predict.called

    def test_hybrid_multi_objective_evaluation(
        self, hybrid_ga_config, mock_hybrid_predictor, mock_feature_adapter
    ):
        """ハイブリッド多目的評価テスト"""
        from backend.app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        hybrid_ga_config.enable_multi_objective = True
        hybrid_ga_config.objectives = ["total_return", "sharpe_ratio", "hybrid_score"]

        mock_backtest = Mock()
        mock_backtest.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "total_trades": 50,
        }

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest,
            predictor=mock_hybrid_predictor,
            feature_adapter=mock_feature_adapter,
        )

        individual = [0.1, 0.2, 0.3]

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }
        evaluator.set_backtest_config(backtest_config)

        fitness = evaluator.evaluate_individual(individual, hybrid_ga_config)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 3  # 多目的

    def test_hybrid_predictor_integration(self, mock_hybrid_predictor):
        """ハイブリッド予測器統合テスト"""
        from backend.app.services.auto_strategy.core.hybrid_predictor import (
            HybridPredictor,
        )

        # HybridPredictorの初期化テスト
        with patch(
            "backend.app.services.ml.ml_training_service.MLTrainingService"
        ) as mock_ml_service_class:
            mock_ml_service = Mock()
            mock_ml_service.generate_signals.return_value = {
                "up": 0.3,
                "down": 0.4,
                "range": 0.3,
            }
            mock_ml_service.predict.return_value = np.array([0.2, 0.3, 0.5])
            mock_ml_service_class.return_value = mock_ml_service

            predictor = HybridPredictor(
                trainer_type="single",
                model_type="lightgbm",
            )

            features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
            prediction = predictor.predict(features_df)

            assert prediction == 0.8  # mockの戻り値

    def test_feature_adapter_integration(self, mock_feature_adapter):
        """特徴量アダプタ統合テスト"""
        from backend.app.services.auto_strategy.utils.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        # HybridFeatureAdapterの初期化テスト
        adapter = HybridFeatureAdapter(automl_config={"enabled": True})

        # 特徴量適応テスト
        original_features = pd.DataFrame(
            {"close": [100, 101, 102], "volume": [1000, 1100, 1200]}
        )

        adapted = adapter.adapt_features(original_features)
        assert isinstance(adapted, pd.DataFrame)
        assert len(adapted) == len(original_features)

    def test_hybrid_ga_full_integration(
        self,
        sample_training_data,
        hybrid_ga_config,
        mock_hybrid_predictor,
        mock_feature_adapter,
    ):
        """ハイブリッドGA完全統合テスト"""
        with patch(
            "backend.app.services.auto_strategy.generators.random_gene_generator.RandomGeneGenerator"
        ) as mock_generator_class:
            mock_generator = Mock()
            mock_generator_class.return_value = mock_generator

            with patch(
                "backend.app.services.auto_strategy.services.backtest.backtest_service.BacktestService"
            ) as mock_backtest_class:
                mock_backtest = Mock()
                mock_backtest.run_backtest.return_value = {
                    "sharpe_ratio": 1.5,
                    "total_return": 0.25,
                    "max_drawdown": 0.1,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "total_trades": 50,
                }
                mock_backtest_class.return_value = mock_backtest

                with patch(
                    "backend.app.services.auto_strategy.core.ga_engine.GeneticAlgorithmEngine.run_evolution"
                ) as mock_run_evolution:
                    mock_run_evolution.return_value = {
                        "best_strategy": {"id": "test_strategy"},
                        "best_fitness": 1.5,
                        "statistics": {"avg_fitness": 1.2},
                    }

                    engine = GeneticAlgorithmEngine(
                        gene_generator=mock_generator,
                        hybrid_mode=True,
                        hybrid_predictor=mock_hybrid_predictor,
                        hybrid_feature_adapter=mock_feature_adapter,
                    )

                    result = engine.run_ga(sample_training_data, hybrid_ga_config)

        assert "best_strategy" in result
        assert engine.hybrid_mode is True

    def test_hybrid_error_handling(
        self, hybrid_ga_config, mock_hybrid_predictor, mock_feature_adapter
    ):
        """ハイブリッド統合時のエラー処理テスト"""
        from backend.app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest = Mock()
        mock_backtest.run_backtest.side_effect = Exception("Backtest error")

        evaluator = HybridIndividualEvaluator(
            backtest_service=mock_backtest,
            predictor=mock_hybrid_predictor,
            feature_adapter=mock_feature_adapter,
        )

        individual = [0.1, 0.2, 0.3]

        # エラーが発生してもデフォルトフィットネスが返されることを確認
        fitness = evaluator.evaluate_individual(individual, hybrid_ga_config)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 1

    def test_hybrid_config_validation(self):
        """ハイブリッド設定の検証テスト"""
        config = GASettings(hybrid_mode=True)

        # ハイブリッド関連設定がデフォルト値で設定されていることを確認
        assert config.hybrid_mode is True
        assert config.hybrid_model_type == "lightgbm"
        assert config.log_level == "ERROR"

        # バリデーションが通ることを確認
        is_valid, errors = config.validate()
        assert is_valid is True

    def test_hybrid_automl_integration(self):
        """ハイブリッドAutoML統合テスト"""
        automl_config = {
            "enabled": True,
            "feature_selection": True,
            "apply_preprocessing": True,
        }

        config = GASettings(hybrid_mode=True, hybrid_automl_config=automl_config)

        assert config.hybrid_automl_config == automl_config

        # AutoML設定が正しく反映されていることを確認
        assert config.hybrid_automl_config["enabled"] is True
        assert config.hybrid_automl_config["feature_selection"] is True

    def test_hybrid_multiple_models(self):
        """複数モデルハイブリッドテスト"""
        model_types = ["lightgbm", "xgboost", "randomforest"]

        config = GASettings(hybrid_mode=True, hybrid_model_types=model_types)

        assert config.hybrid_model_types == model_types
        assert len(config.hybrid_model_types) == 3

    def test_hybrid_predictor_fallback(self, mock_hybrid_predictor):
        """ハイブリッド予測器フォールバックテスト"""
        # 予測器が未学習の場合のフォールバック
        mock_hybrid_predictor.is_trained = False

        from backend.app.services.auto_strategy.core.hybrid_predictor import (
            HybridPredictor,
        )

        with patch(
            "backend.app.services.ml.ml_training_service.MLTrainingService"
        ) as mock_ml_service_class:
            mock_ml_service = Mock()
            mock_ml_service.config.prediction.get_default_predictions.return_value = {
                "up": 0.33,
                "down": 0.33,
                "range": 0.34,
            }
            mock_ml_service_class.return_value = mock_ml_service

            predictor = HybridPredictor(trainer_type="single", model_type="lightgbm")

            features_df = pd.DataFrame([[1.0, 2.0]], columns=["close", "volume"])
            signals = predictor.generate_signals(features_df)

            # デフォルト値が返されることを確認
            assert isinstance(signals, dict)
            assert "up" in signals
            assert "down" in signals
            assert "range" in signals
