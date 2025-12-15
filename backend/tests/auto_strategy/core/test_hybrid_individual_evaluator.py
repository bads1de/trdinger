"""
HybridIndividualEvaluatorのテスト

ML予測を統合したハイブリッドGA個体評価器のテスト
"""

import pytest
from unittest.mock import Mock, patch

import pandas as pd


class TestHybridIndividualEvaluatorInit:
    """初期化のテスト"""

    def test_init_with_defaults(self):
        """デフォルト設定での初期化"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        evaluator = HybridIndividualEvaluator(mock_backtest_service)

        assert evaluator.backtest_service == mock_backtest_service
        assert evaluator.predictor is None
        assert evaluator.feature_adapter is not None

    def test_init_with_predictor(self):
        """予測器を指定しての初期化"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor

        mock_backtest_service = Mock()
        mock_predictor = Mock(spec=HybridPredictor)
        evaluator = HybridIndividualEvaluator(
            mock_backtest_service, predictor=mock_predictor
        )

        assert evaluator.predictor == mock_predictor

    def test_init_with_feature_adapter(self):
        """特徴量アダプタを指定しての初期化"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        mock_adapter = Mock()
        evaluator = HybridIndividualEvaluator(
            mock_backtest_service, feature_adapter=mock_adapter
        )

        assert evaluator.feature_adapter == mock_adapter


class TestEnsureBacktestDefaults:
    """_ensure_backtest_defaultsのテスト"""

    @pytest.fixture
    def evaluator(self):
        """テスト用のevaluatorを作成"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            return HybridIndividualEvaluator(mock_backtest_service)

    def test_uses_existing_symbol_and_timeframe(self, evaluator):
        """既存のシンボルとタイムフレームを使用"""
        from app.services.auto_strategy.config.ga_runtime import GAConfig

        backtest_config = {"symbol": "ETHUSDT", "timeframe": "4h"}
        ga_config = Mock(spec=GAConfig)
        ga_config.target_symbol = "BTCUSDT"
        ga_config.target_timeframe = "1h"

        result = evaluator._ensure_backtest_defaults(backtest_config, ga_config)

        assert result["symbol"] == "ETHUSDT"
        assert result["timeframe"] == "4h"

    def test_uses_ga_config_defaults(self, evaluator):
        """GA設定のデフォルト値を使用"""
        from app.services.auto_strategy.config.ga_runtime import GAConfig

        backtest_config = {}
        ga_config = Mock(spec=GAConfig)
        ga_config.target_symbol = "BTCUSDT"
        ga_config.target_timeframe = "1h"
        ga_config.base_symbol = None
        ga_config.timeframe = None
        ga_config.fallback_start_date = "2023-01-01"
        ga_config.fallback_end_date = "2023-12-31"

        result = evaluator._ensure_backtest_defaults(backtest_config, ga_config)

        assert result["symbol"] == "BTCUSDT"
        assert result["timeframe"] == "1h"
        assert result["start_date"] == "2023-01-01"
        assert result["end_date"] == "2023-12-31"

    def test_uses_fallback_defaults(self, evaluator):
        """フォールバックのデフォルト値を使用"""
        from app.services.auto_strategy.config.ga_runtime import GAConfig

        backtest_config = {}
        ga_config = Mock(spec=GAConfig)
        # すべてのシンボル・タイムフレーム属性をNoneに設定
        ga_config.target_symbol = None
        ga_config.target_timeframe = None
        ga_config.base_symbol = None
        ga_config.timeframe = None
        ga_config.fallback_start_date = None
        ga_config.fallback_end_date = None

        result = evaluator._ensure_backtest_defaults(backtest_config, ga_config)

        # デフォルト値が使用される
        assert result["symbol"] == "BTCUSDT"
        assert result["timeframe"] == "1h"


class TestFetchOhlcvData:
    """_fetch_ohlcv_dataのテスト"""

    @pytest.fixture
    def evaluator_with_data_service(self):
        """data_serviceを持つevaluatorを作成"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_data_service = Mock()
        mock_backtest_service = Mock()
        mock_backtest_service.data_service = mock_data_service

        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            return HybridIndividualEvaluator(mock_backtest_service)

    def test_returns_none_when_missing_parameters(self, evaluator_with_data_service):
        """必須パラメータが不足している場合はNoneを返す"""
        backtest_config = {"symbol": "BTCUSDT"}  # timeframe, start_date, end_dateがない
        ga_config = Mock()

        result = evaluator_with_data_service._fetch_ohlcv_data(
            backtest_config, ga_config
        )

        assert result is None

    def test_returns_cached_data(self, evaluator_with_data_service):
        """キャッシュにデータがある場合はキャッシュを返す"""
        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }
        ga_config = Mock()

        # キャッシュにデータを設定
        cache_key = ("ohlcv", "BTCUSDT", "1h", "2023-01-01", "2023-12-31")
        cached_df = pd.DataFrame({"close": [100, 101, 102]})
        evaluator_with_data_service._data_cache[cache_key] = cached_df

        result = evaluator_with_data_service._fetch_ohlcv_data(
            backtest_config, ga_config
        )

        assert result is not None
        pd.testing.assert_frame_equal(result, cached_df)

    def test_fetches_from_db_on_cache_miss(self, evaluator_with_data_service):
        """キャッシュミスの場合はDBから取得"""
        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }
        ga_config = Mock()

        # DBから取得するデータをモック
        db_df = pd.DataFrame({"close": [100, 101, 102]})
        evaluator_with_data_service.backtest_service.data_service.get_ohlcv_data.return_value = (
            db_df
        )

        result = evaluator_with_data_service._fetch_ohlcv_data(
            backtest_config, ga_config
        )

        assert result is not None
        pd.testing.assert_frame_equal(result, db_df)
        # キャッシュに保存されていることを確認
        cache_key = ("ohlcv", "BTCUSDT", "1h", "2023-01-01", "2023-12-31")
        assert cache_key in evaluator_with_data_service._data_cache

    def test_returns_none_when_data_service_unavailable(self):
        """data_serviceが利用できない場合はNoneを返す"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        mock_backtest_service.data_service = None

        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            evaluator = HybridIndividualEvaluator(mock_backtest_service)

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }
        ga_config = Mock()

        result = evaluator._fetch_ohlcv_data(backtest_config, ga_config)

        assert result is None


class TestCalculateFitness:
    """_calculate_fitnessのテスト"""

    @pytest.fixture
    def evaluator(self):
        """テスト用evaluatorを作成"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            return HybridIndividualEvaluator(mock_backtest_service)

    def test_returns_base_fitness_without_prediction(self, evaluator):
        """予測がない場合はベースフィットネスを返す"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
                "profit_factor": 2.0,
                "win_rate": 0.6,
            }
        }
        config = Mock()
        config.fitness_weights = {"sharpe_ratio": 1.0}

        with patch.object(
            evaluator.__class__.__bases__[0], "_calculate_fitness", return_value=0.5
        ):
            result = evaluator._calculate_fitness(backtest_result, config)

        assert result == 0.5

    def test_integrates_ml_prediction_score(self, evaluator):
        """ML予測スコアを統合"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
                "profit_factor": 2.0,
                "win_rate": 0.6,
            }
        }
        config = Mock()
        config.fitness_weights = {"prediction_score": 0.2}

        prediction_signals = {"up": 0.7, "down": 0.3}

        with patch.object(
            evaluator.__class__.__bases__[0], "_calculate_fitness", return_value=0.5
        ):
            result = evaluator._calculate_fitness(
                backtest_result, config, prediction_signals
            )

        # base_fitness (0.5) + prediction_weight (0.2) * prediction_score (0.4) = 0.58
        expected = 0.5 + 0.2 * (0.7 - 0.3)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_handles_trend_prediction(self, evaluator):
        """トレンド予測を処理"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
            }
        }
        config = Mock()
        config.fitness_weights = {"prediction_score": 0.1}

        prediction_signals = {"trend": 0.8}

        with patch.object(
            evaluator.__class__.__bases__[0], "_calculate_fitness", return_value=0.5
        ):
            result = evaluator._calculate_fitness(
                backtest_result, config, prediction_signals
            )

        # base_fitness (0.5) + prediction_weight (0.1) * (trend - 0.5) = 0.5 + 0.1 * 0.3 = 0.53
        expected = 0.5 + 0.1 * (0.8 - 0.5)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_returns_zero_for_no_trades(self, evaluator):
        """取引なしの場合はMLスコアを統合しない"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 0,
                "sharpe_ratio": 0,
            }
        }
        config = Mock()
        config.fitness_weights = {"prediction_score": 0.1}

        prediction_signals = {"up": 0.9, "down": 0.1}

        with patch.object(
            evaluator.__class__.__bases__[0], "_calculate_fitness", return_value=-0.5
        ):
            result = evaluator._calculate_fitness(
                backtest_result, config, prediction_signals
            )

        # 取引なしの場合は max(0, base_fitness) を返す
        assert result == 0.0


class TestCalculateMultiObjectiveFitness:
    """_calculate_multi_objective_fitnessのテスト"""

    @pytest.fixture
    def evaluator(self):
        """テスト用evaluatorを作成"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            return HybridIndividualEvaluator(mock_backtest_service)

    def test_returns_base_fitness_without_prediction(self, evaluator):
        """予測がない場合はベース多目的フィットネスを返す"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
            }
        }
        config = Mock()
        config.objectives = ["sharpe_ratio", "profit_factor"]
        config.fitness_weights = {}

        base_values = (0.5, 0.6)

        with patch.object(
            evaluator.__class__.__bases__[0],
            "_calculate_multi_objective_fitness",
            return_value=base_values,
        ):
            result = evaluator._calculate_multi_objective_fitness(
                backtest_result, config
            )

        assert result == base_values

    def test_integrates_weighted_score_with_prediction(self, evaluator):
        """weighted_scoreにML予測スコアを統合"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
            }
        }
        config = Mock()
        config.objectives = ["weighted_score", "profit_factor"]
        config.fitness_weights = {"prediction_score": 0.1}

        prediction_signals = {"up": 0.8, "down": 0.2}
        base_values = (0.5, 0.6)

        with patch.object(
            evaluator.__class__.__bases__[0],
            "_calculate_multi_objective_fitness",
            return_value=base_values,
        ):
            result = evaluator._calculate_multi_objective_fitness(
                backtest_result, config, prediction_signals
            )

        # weighted_score = 0.5 + 0.1 * (0.8 - 0.2) = 0.56
        expected_ws = max(0.0, 0.5 + 0.1 * (0.8 - 0.2))
        assert result[0] == pytest.approx(expected_ws, rel=1e-5)
        assert result[1] == 0.6

    def test_handles_prediction_score_objective(self, evaluator):
        """prediction_scoreが独立した目的として含まれる場合"""
        backtest_result = {
            "performance_metrics": {
                "total_trades": 10,
            }
        }
        config = Mock()
        config.objectives = ["sharpe_ratio", "prediction_score"]
        config.fitness_weights = {}

        prediction_signals = {"up": 0.7, "down": 0.3}
        base_values = (0.5, 0.0)

        with patch.object(
            evaluator.__class__.__bases__[0],
            "_calculate_multi_objective_fitness",
            return_value=base_values,
        ):
            result = evaluator._calculate_multi_objective_fitness(
                backtest_result, config, prediction_signals
            )

        # prediction_score = 0.7 - 0.3 = 0.4
        assert result[0] == 0.5
        assert result[1] == pytest.approx(0.4, rel=1e-5)

    def test_handles_trend_in_prediction_score_objective(self, evaluator):
        """prediction_score目的でトレンド予測を処理"""
        backtest_result = {}
        config = Mock()
        config.objectives = ["prediction_score"]
        config.fitness_weights = {}

        prediction_signals = {"trend": 0.9}
        base_values = (0.0,)

        with patch.object(
            evaluator.__class__.__bases__[0],
            "_calculate_multi_objective_fitness",
            return_value=base_values,
        ):
            result = evaluator._calculate_multi_objective_fitness(
                backtest_result, config, prediction_signals
            )

        # prediction_score = 0.9 - 0.5 = 0.4
        assert result[0] == pytest.approx(0.4, rel=1e-5)


class TestPerformSingleEvaluation:
    """_perform_single_evaluationのテスト"""

    @pytest.fixture
    def mock_dependencies(self):
        """モック依存関係"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_trades": 10,
                "sharpe_ratio": 1.5,
            }
        }

        mock_predictor = Mock()
        mock_predictor.predict.return_value = {"up": 0.7, "down": 0.3}

        mock_feature_adapter = Mock()
        mock_feature_adapter.gene_to_features.return_value = pd.DataFrame(
            {"feature1": [1, 2, 3]}
        )

        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            evaluator = HybridIndividualEvaluator(
                mock_backtest_service,
                predictor=mock_predictor,
                feature_adapter=mock_feature_adapter,
            )

        return evaluator

    def test_successful_evaluation_with_prediction(self, mock_dependencies):
        """予測ありの評価が成功"""
        evaluator = mock_dependencies
        gene = Mock()
        gene.id = "test_gene_12345678"

        config = Mock()
        config.objectives = ["weighted_score"]
        config.fitness_weights = {"prediction_score": 0.1}
        config.target_symbol = "BTCUSDT"
        config.target_timeframe = "1h"
        config.base_symbol = None
        config.timeframe = None
        config.fallback_start_date = None
        config.fallback_end_date = None
        config.preprocess_features = True

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }

        # OHLCVデータのモック
        ohlcv_data = pd.DataFrame({"close": [100, 101, 102]})

        # GeneSerializerをモック
        mock_serializer = Mock()
        mock_serializer.strategy_gene_to_dict.return_value = {"test": "payload"}

        with patch(
            "app.services.auto_strategy.serializers.gene_serialization.GeneSerializer",
            return_value=mock_serializer,
        ):
            with patch.object(evaluator, "_fetch_ohlcv_data", return_value=ohlcv_data):
                with patch.object(evaluator, "_get_cached_data", return_value=None):
                    with patch.object(
                        evaluator,
                        "_calculate_multi_objective_fitness",
                        return_value=(0.8,),
                    ):
                        result = evaluator._perform_single_evaluation(
                            gene, backtest_config, config
                        )

        assert isinstance(result, tuple)
        assert result == (0.8,)

    def test_continues_when_ml_prediction_fails(self, mock_dependencies):
        """ML予測が失敗しても評価を続行"""
        evaluator = mock_dependencies
        evaluator.predictor.predict.side_effect = Exception("ML error")

        gene = Mock()
        gene.id = "test_gene_12345678"

        config = Mock()
        config.objectives = ["weighted_score"]
        config.fitness_weights = {}
        config.target_symbol = "BTCUSDT"
        config.target_timeframe = "1h"
        config.base_symbol = None
        config.timeframe = None
        config.fallback_start_date = None
        config.fallback_end_date = None
        config.preprocess_features = True

        backtest_config = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        }

        ohlcv_data = pd.DataFrame({"close": [100, 101, 102]})

        # GeneSerializerをモック
        mock_serializer = Mock()
        mock_serializer.strategy_gene_to_dict.return_value = {"test": "payload"}

        with patch(
            "app.services.auto_strategy.serializers.gene_serialization.GeneSerializer",
            return_value=mock_serializer,
        ):
            with patch.object(evaluator, "_fetch_ohlcv_data", return_value=ohlcv_data):
                with patch.object(evaluator, "_get_cached_data", return_value=None):
                    with patch.object(
                        evaluator,
                        "_calculate_multi_objective_fitness",
                        return_value=(0.5,),
                    ) as mock_fitness:
                        evaluator._perform_single_evaluation(
                            gene, backtest_config, config
                        )

        # 予測がNoneで呼び出されることを確認
        mock_fitness.assert_called_once()
        call_args = mock_fitness.call_args
        assert call_args[0][2] is None  # prediction_signals がNone

    def test_returns_zero_fitness_on_error(self, mock_dependencies):
        """エラー時にゼロフィットネスを返す"""
        evaluator = mock_dependencies
        evaluator.backtest_service.run_backtest.side_effect = Exception(
            "Backtest error"
        )

        gene = Mock()
        gene.id = "test_gene_123"

        config = Mock()
        config.objectives = ["weighted_score", "profit_factor"]
        config.target_symbol = "BTCUSDT"
        config.target_timeframe = "1h"
        config.base_symbol = None
        config.timeframe = None
        config.fallback_start_date = None
        config.fallback_end_date = None

        backtest_config = {}

        result = evaluator._perform_single_evaluation(gene, backtest_config, config)

        assert result == (0.0, 0.0)


class TestShouldApplyPreprocessing:
    """_should_apply_preprocessingのテスト"""

    def test_returns_true_when_preprocess_features_is_true(self):
        """preprocess_featuresがTrueの場合Trueを返す"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            evaluator = HybridIndividualEvaluator(mock_backtest_service)

        config = Mock()
        config.preprocess_features = True

        assert evaluator._should_apply_preprocessing(config) is True

    def test_returns_false_when_preprocess_features_is_false(self):
        """preprocess_featuresがFalseの場合Falseを返す"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            evaluator = HybridIndividualEvaluator(mock_backtest_service)

        config = Mock()
        config.preprocess_features = False

        assert evaluator._should_apply_preprocessing(config) is False

    def test_returns_default_true_when_attribute_missing(self):
        """属性が存在しない場合はTrueを返す"""
        from app.services.auto_strategy.core.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        mock_backtest_service = Mock()
        with patch.object(
            HybridIndividualEvaluator, "_create_feature_adapter", return_value=Mock()
        ):
            evaluator = HybridIndividualEvaluator(mock_backtest_service)

        config = Mock(spec=[])  # preprocess_features属性なし

        assert evaluator._should_apply_preprocessing(config) is True




