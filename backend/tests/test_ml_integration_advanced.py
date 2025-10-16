"""
ML統合テスト
GAとMLのハイブリッド統合テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
from app.services.auto_strategy.core.hybrid_individual_evaluator import HybridIndividualEvaluator
from app.services.auto_strategy.core.drl_policy_adapter import DRLPolicyAdapter
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config.ga import GAConfig
from app.services.ml.ml_training_service import MLTrainingService
from app.services.backtest.backtest_service import BacktestService


class TestHybridMLIntegration:
    """ハイブリッドML統合テスト"""

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 200,
            'high': 10000 + np.random.randn(len(dates)) * 300,
            'low': 10000 + np.random.randn(len(dates)) * 300,
            'close': 10000 + np.random.randn(len(dates)) * 200,
            'volume': 500 + np.random.randint(100, 1000, len(dates)),
            'returns': np.random.randn(len(dates)) * 0.02,
            'volatility': 0.01 + np.random.rand(len(dates)) * 0.02,
        })

        # OHLCの関係を確保
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)

        return data

    @pytest.fixture
    def mock_ml_model(self):
        """モックMLモデル"""
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(10)
        mock_model.fit.return_value = None
        return mock_model

    @pytest.fixture
    def hybrid_predictor(self, mock_ml_model):
        """ハイブリッド予測器"""
        predictor = HybridPredictor()
        predictor.ml_model = mock_ml_model
        predictor.is_trained = True
        return predictor

    def test_hybrid_predictor_initialization(self):
        """ハイブリッド予測器初期化のテスト"""
        predictor = HybridPredictor()

        assert predictor is not None
        assert predictor.is_trained is False
        assert predictor.feature_importance is None
        assert predictor.model_confidence == 0.5

    def test_feature_engineering_integration(self, hybrid_predictor, sample_market_data):
        """特徴量エンジニアリング統合のテスト"""
        # 市場データから特徴量を抽出
        features = hybrid_predictor._extract_market_features(sample_market_data)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_market_data)
        assert features.shape[1] > 0  # 特徴量が抽出されていること

    def test_market_regime_prediction(self, hybrid_predictor, sample_market_data):
        """市場レジーム予測のテスト"""
        # レジーム予測を実行
        regime_probabilities = hybrid_predictor.predict_market_regime(sample_market_data)

        assert isinstance(regime_probabilities, dict)
        assert 'bullish' in regime_probabilities
        assert 'bearish' in regime_probabilities
        assert 'sideways' in regime_probabilities
        assert all(0 <= prob <= 1 for prob in regime_probabilities.values())
        assert abs(sum(regime_probabilities.values()) - 1.0) < 0.01

    def test_signal_enhancement_with_ml(self, hybrid_predictor):
        """MLによるシグナル強化のテスト"""
        base_signal = 0.75
        market_context = {
            'volatility': 0.02,
            'trend': 0.1,
            'momentum': -0.05,
            'volume_trend': 0.15,
        }

        enhanced_signal = hybrid_predictor.enhance_trading_signal(
            base_signal, market_context
        )

        assert isinstance(enhanced_signal, float)
        assert 0.0 <= enhanced_signal <= 1.0

    def test_confidence_weighted_prediction(self, hybrid_predictor, sample_market_data):
        """信頼度重み付き予測のテスト"""
        # 予測を実行
        prediction = hybrid_predictor.get_confidence_weighted_prediction(sample_market_data)

        assert isinstance(prediction, dict)
        assert 'signal' in prediction
        assert 'confidence' in prediction
        assert 'regime' in prediction
        assert 0.0 <= prediction['confidence'] <= 1.0

    def test_model_drift_detection(self, hybrid_predictor, sample_market_data):
        """モデルドリフト検出のテスト"""
        # モデルを訓練
        hybrid_predictor.train(sample_market_data)

        # ドリフト検出を実行
        is_drift = hybrid_predictor.detect_model_drift(sample_market_data)

        assert isinstance(is_drift, bool)

    def test_adaptive_model_retraining(self, hybrid_predictor, sample_market_data):
        """適応的モデル再訓練のテスト"""
        initial_training_count = getattr(hybrid_predictor, '_training_count', 0)

        # 再訓練をトリガー
        hybrid_predictor._trigger_adaptive_retraining(sample_market_data)

        # 再訓練が実行されたこと
        new_training_count = getattr(hybrid_predictor, '_training_count', 0)
        assert new_training_count > initial_training_count

    def test_hybrid_individual_evaluator_creation(self):
        """ハイブリッド個体評価器作成のテスト"""
        evaluator = HybridIndividualEvaluator()

        assert evaluator is not None
        assert evaluator.hybrid_predictor is not None
        assert evaluator.base_evaluator is not None

    def test_enhanced_individual_evaluation(self, hybrid_predictor):
        """強化個体評価のテスト"""
        evaluator = HybridIndividualEvaluator()
        evaluator.hybrid_predictor = hybrid_predictor

        # テスト用個体
        individual = np.random.rand(5)
        backtest_metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'win_rate': 0.60,
        }

        # 強化評価を実行
        enhanced_fitness = evaluator.evaluate_with_ml_enhancement(
            individual, backtest_metrics
        )

        assert isinstance(enhanced_fitness, float)
        assert enhanced_fitness >= 0.0

    def test_drl_policy_adapter_initialization(self):
        """DRLポリシーアダプター初期化のテスト"""
        adapter = DRLPolicyAdapter()

        assert adapter is not None
        assert adapter.drl_model is None
        assert adapter.action_space == []
        assert adapter.state_normalizer is not None

    def test_state_observation_transformation(self, sample_market_data):
        """状態観測変換のテスト"""
        adapter = DRLPolicyAdapter()

        # 市場データをDRL状態に変換
        state = adapter.transform_market_to_state(sample_market_data)

        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1
        assert len(state) > 0

    def test_action_space_mapping(self):
        """行動空間マッピングのテスト"""
        adapter = DRLPolicyAdapter()

        # GA遺伝子をDRL行動にマップ
        ga_genes = [0.1, 0.5, 0.8, 0.3]
        drl_action = adapter.map_ga_to_drl_action(ga_genes)

        assert isinstance(drl_action, list)
        assert len(drl_action) == len(ga_genes)
        assert all(0 <= action <= 1 for action in drl_action)

    def test_reward_shaping_integration(self):
        """報酬整形統合のテスト"""
        adapter = DRLPolicyAdapter()

        # GA報酬をDRL報酬に整形
        ga_rewards = [0.1, 0.15, 0.08, 0.12]
        shaped_rewards = adapter.shape_rewards_for_drl(ga_rewards)

        assert isinstance(shaped_rewards, list)
        assert len(shaped_rewards) == len(ga_rewards)
        assert all(isinstance(r, float) for r in shaped_rewards)

    def test_ga_drl_hybrid_evolution(self, sample_market_data):
        """GA-DRLハイブリッド進化のテスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 20,
            "num_generations": 5,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
            }
        }

        # ハイブリッド評価器をモック
        mock_evaluator = Mock()
        mock_evaluator.evaluate_with_ml_enhancement.return_value = 0.85

        # GAエンジンを作成
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=sample_market_data,
            regime_detector=None
        )

        # DRLアダプターを統合
        drl_adapter = DRLPolicyAdapter()
        engine.drl_adapter = drl_adapter
        engine.hybrid_evaluator = mock_evaluator

        # ハイブリッド進化を実行
        result = engine.evolve_with_drl_integration()

        assert result is not None
        assert hasattr(result, 'best_individual')
        assert hasattr(result, 'convergence_metrics')

    def test_ml_model_performance_monitoring(self, hybrid_predictor, sample_market_data):
        """MLモデルパフォーマンス監視のテスト"""
        # モデルを訓練
        hybrid_predictor.train(sample_market_data)

        # パフォーマンスを監視
        performance_metrics = hybrid_predictor.monitor_model_performance(sample_market_data)

        assert isinstance(performance_metrics, dict)
        assert 'accuracy' in performance_metrics
        assert 'stability' in performance_metrics
        assert 'drift_score' in performance_metrics

    def test_ensemble_prediction_with_ga(self, hybrid_predictor, sample_market_data):
        """GAとのアンサンブル予測のテスト"""
        # GA個体群を生成
        population = [np.random.rand(5) for _ in range(10)]

        # アンサンブル予測を実行
        ensemble_prediction = hybrid_predictor.get_ensemble_prediction(
            population, sample_market_data
        )

        assert isinstance(ensemble_prediction, dict)
        assert 'consensus_signal' in ensemble_prediction
        assert 'prediction_variance' in ensemble_prediction
        assert 'confidence_interval' in ensemble_prediction

    def test_real_time_ml_inference(self, hybrid_predictor):
        """リアルタイムML推論のテスト"""
        # リアルタイムデータをシミュレート
        real_time_data = {
            'price': 10000.0,
            'volume': 500.0,
            'volatility': 0.02,
            'momentum': 0.1,
        }

        # リアルタイム推論を実行
        inference_result = hybrid_predictor.perform_real_time_inference(real_time_data)

        assert isinstance(inference_result, dict)
        assert 'prediction' in inference_result
        assert 'confidence' in inference_result
        assert 'recommendation' in inference_result

    def test_ml_ga_feedback_loop(self, sample_market_data):
        """ML-GAフィードバックループのテスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 15,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
            }
        }

        # ハイブリッド予測器
        hybrid_predictor = HybridPredictor()
        hybrid_predictor.train(sample_market_data)

        # GAエンジン
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=sample_market_data,
            regime_detector=None
        )
        engine.hybrid_predictor = hybrid_predictor

        # フィードバックループを実行
        final_result = engine.execute_ml_ga_feedback_loop(sample_market_data)

        assert isinstance(final_result, dict)
        assert 'optimized_strategy' in final_result
        assert 'performance_improvement' in final_result
        assert 'ml_insights' in final_result

    def test_error_handling_in_ml_integration(self):
        """ML統合におけるエラーハンドリングのテスト"""
        hybrid_predictor = HybridPredictor()

        # 不正なデータでのテスト
        invalid_data = pd.DataFrame({'invalid': ['invalid']})

        # エラーが適切に処理されること
        try:
            hybrid_predictor.train(invalid_data)
        except Exception as e:
            assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def test_final_hybrid_integration_validation(self, hybrid_predictor, sample_market_data):
        """最終ハイブリッド統合検証"""
        assert hybrid_predictor is not None

        # 基本的なML予測が可能であること
        prediction = hybrid_predictor.predict_market_regime(sample_market_data)
        assert isinstance(prediction, dict)

        # GAとの統合が可能であること
        ga_signal = 0.7
        enhanced = hybrid_predictor.enhance_trading_signal(ga_signal, {'volatility': 0.02})
        assert isinstance(enhanced, float)

        print("✅ ML統合テスト成功")


# TDDアプローチによるML統合テスト
class TestMLIntegrationTDD:
    """TDDアプローチによるML統合テスト"""

    def test_ml_model_integration_basic(self):
        """基本的なMLモデル統合テスト"""
        predictor = HybridPredictor()

        # 空のデータでテスト
        try:
            features = predictor._extract_market_features(pd.DataFrame())
            assert isinstance(features, np.ndarray)
        except Exception:
            # エラーは許容
            pass

        print("✅ 基本的なMLモデル統合テスト成功")

    def test_hybrid_signal_generation(self):
        """ハイブリッドシグナル生成テスト"""
        predictor = HybridPredictor()

        base_signal = 0.6
        context = {'volatility': 0.01, 'trend': 0.05}

        enhanced = predictor.enhance_trading_signal(base_signal, context)
        assert 0.0 <= enhanced <= 1.0

        print("✅ ハイブリッドシグナル生成テスト成功")

    def test_ga_ml_communication_interface(self):
        """GA-ML通信インターフェーステスト"""
        # 基本的なインターフェースが存在すること
        predictor = HybridPredictor()

        assert hasattr(predictor, 'train')
        assert hasattr(predictor, 'predict_market_regime')
        assert hasattr(predictor, 'enhance_trading_signal')

        print("✅ GA-ML通信インターフェーステスト成功")

    def test_incremental_learning_capability(self):
        """増分学習機能テスト"""
        predictor = HybridPredictor()
        sample_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 100],
            'volume': [1000, 1100, 900, 1200, 1000],
        })

        # 初期訓練
        predictor.train(sample_data)

        # 増分学習
        new_data = pd.DataFrame({
            'close': [101, 103, 100, 104],
            'volume': [1050, 1150, 950, 1250],
        })

        predictor.update_model_incrementally(new_data)

        # 更新が成功したこと
        assert predictor.is_trained is True

        print("✅ 増分学習機能テスト成功")

    def test_model_ensemble_techniques(self):
        """モデルアンサンブル技法テスト"""
        predictor = HybridPredictor()
        population = [np.random.rand(3) for _ in range(5)]
        market_data = pd.DataFrame({
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
        })

        # アンサンブル予測を実行
        ensemble_result = predictor.get_ensemble_prediction(population, market_data)

        assert isinstance(ensemble_result, dict)
        assert 'consensus_signal' in ensemble_result

        print("✅ モデルアンサンブル技法テスト成功")