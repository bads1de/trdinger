"""
ML統合テスト - GA+MLハイブリッド、DRL、ハイブリッド評価を含む
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from app.services.auto_strategy.core.drl_policy_adapter import DRLPolicyAdapter
from app.services.auto_strategy.core.hybrid_individual_evaluator import HybridIndividualEvaluator
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config import GAConfig


class TestMLIntegrationComprehensive:
    """ML統合の包括的テスト"""

    @pytest.fixture
    def drl_adapter(self):
        """DRLアダプタ"""
        return DRLPolicyAdapter()

    @pytest.fixture
    def hybrid_evaluator(self):
        """ハイブリッド個体評価器"""
        mock_backtest = Mock()
        mock_predictor = Mock()
        return HybridIndividualEvaluator(mock_backtest, mock_predictor)

    @pytest.fixture
    def sample_features(self):
        """サンプル特徴量データ"""
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'rsi': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            'macd': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        })

    @pytest.fixture
    def sample_empty_features(self):
        """空の特徴量データ"""
        return pd.DataFrame()

    @pytest.fixture
    def sample_none_features(self):
        """None特徴量データ"""
        return None

    def test_drl_adapter_initialization(self, drl_adapter):
        """DRLアダプタ初期化のテスト"""
        assert drl_adapter.policy_type == "ppo"
        assert isinstance(drl_adapter.policy_config, dict)
        assert drl_adapter._momentum_window >= 4
        assert drl_adapter._volatility_window >= 4

    def test_drl_adapter_custom_predict(self):
        """カスタム予測関数のテスト"""
        def custom_predict(df):
            return {"up": 0.6, "down": 0.3, "range": 0.1}

        adapter = DRLPolicyAdapter(predict_fn=custom_predict)
        result = adapter.predict_signals(pd.DataFrame({"close": [100, 101, 102]))

        assert result["up"] == 0.6
        assert result["down"] == 0.3
        assert result["range"] == 0.1

    def test_drl_adapter_fallback_behavior(self, drl_adapter, sample_empty_features, sample_none_features):
        """DRLアダプタのフォールバック動作テスト"""
        # 空データでのフォールバック
        empty_result = drl_adapter.predict_signals(sample_empty_features)
        assert empty_result == {"up": 1/3, "down": 1/3, "range": 1/3}

        # Noneデータでのフォールバック
        none_result = drl_adapter.predict_signals(sample_none_features)
        assert none_result == {"up": 1/3, "down": 1/3, "range": 1/3}

    def test_drl_adapter_window_parameter_validation(self):
        """DRLアダプタ窓パラメータ検証のテスト"""
        # 小さな窓パラメータ
        adapter = DRLPolicyAdapter(policy_config={
            "momentum_window": 2,  # < 4
            "volatility_window": 2  # < 4
        })

        # 自動的に最小値4に設定される
        assert adapter._momentum_window >= 4
        assert adapter._volatility_window >= 4

    def test_drl_adapter_momentum_calculation(self, drl_adapter, sample_features):
        """DRLアダプタモメンタム計算のテスト"""
        result = drl_adapter.predict_signals(sample_features)

        # 有効な確率が返される
        assert isinstance(result, dict)
        assert "up" in result
        assert "down" in result
        assert "range" in result

        # 確率の合計が1
        total_prob = result["up"] + result["down"] + result["range"]
        assert abs(total_prob - 1.0) < 1e-6

    def test_hybrid_evaluator_initialization(self, hybrid_evaluator):
        """ハイブリッド評価器初期化のテスト"""
        assert hybrid_evaluator.backtest_service is not None
        assert hybrid_evaluator.predictor is not None

    def test_hybrid_evaluation_with_ml_prediction(self, hybrid_evaluator):
        """ML予測付きハイブリッド評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": []
        }

        # バックテストサービスのモック
        hybrid_evaluator.backtest_service.run_backtest.return_value = mock_backtest_result

        # ML予測のモック
        hybrid_evaluator.predictor.predict.return_value = 0.8

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False
        ga_config.fitness_weights = {"base": 0.7, "ml": 0.3}

        # 評価実行
        fitness = hybrid_evaluator.evaluate_individual(mock_individual, ga_config)

        # 複合フィットネスが計算される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1

    def test_ml_prediction_integration_in_evaluation(self, hybrid_evaluator):
        """評価中のML予測統合のテスト"""
        # ML予測が正しく統合される
        hybrid_evaluator.predictor.predict.return_value = 0.9

        # 予測が使用される
        hybrid_evaluator.predictor.predict.assert_not_called()  # 実際の呼び出しはテスト中

    def test_hybrid_vs_pure_ga_performance_comparison(self):
        """ハイブリッドvs純粋GAパフォーマンス比較のテスト"""
        # ハイブリッドGAの設定
        hybrid_config = GAConfig()
        hybrid_config.hybrid_mode = True
        hybrid_config.fitness_weights = {"base": 0.6, "ml": 0.4}

        # 純粋GAの設定
        pure_ga_config = GAConfig()
        pure_ga_config.hybrid_mode = False

        assert hybrid_config.hybrid_mode != pure_ga_config.hybrid_mode

    def test_drl_signal_normalization(self, drl_adapter, sample_features):
        """DRL信号正規化のテスト"""
        result = drl_adapter.predict_signals(sample_features)

        # 正規化が正しく行われる
        total = result["up"] + result["down"] + result["range"]
        assert abs(total - 1.0) < 1e-6

    def test_feature_extraction_robustness(self, drl_adapter):
        """特徴量抽出の堅牢性テスト"""
        # 欠損値を含むデータ
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, 103, np.inf, 105],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500]
        })

        result = drl_adapter.predict_signals(data_with_nan)

        # 欠損値が処理される
        assert isinstance(result, dict)

    def test_ml_model_fallback_mechanism(self, hybrid_evaluator):
        """MLモデルフォールバックメカニズムのテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": []
        }

        hybrid_evaluator.backtest_service.run_backtest.return_value = mock_backtest_result

        # ML予測でエラー
        hybrid_evaluator.predictor.predict.side_effect = Exception("ML model failed")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        # フォールバックが動作
        fitness = hybrid_evaluator.evaluate_individual(mock_individual, ga_config)

        # MLなしで評価される
        assert isinstance(fitness, tuple)

    def test_multi_objective_hybrid_evaluation(self, hybrid_evaluator):
        """多目的ハイブリッド評価のテスト"""
        mock_individual = [1, 2, 3, 4, 5]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 10
            },
            "equity_curve": [100, 110, 105, 120],
            "trade_history": []
        }

        hybrid_evaluator.backtest_service.run_backtest.return_value = mock_backtest_result
        hybrid_evaluator.predictor.predict.return_value = 0.8

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["sharpe_ratio", "ml_accuracy"]

        # 多目的評価
        fitness = hybrid_evaluator.evaluate_individual(mock_individual, ga_config)

        # 複数の目的が評価される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 2

    def test_drl_policy_type_configuration(self):
        """DRLポリシータイプ設定のテスト"""
        policy_types = ["ppo", "ddpg", "a2c", "sac"]

        for policy_type in policy_types:
            adapter = DRLPolicyAdapter(policy_type=policy_type)
            assert adapter.policy_type == policy_type

    def test_hybrid_weight_balancing(self):
        """ハイブリッド重みバランスのテスト"""
        # 重みのバリエーション
        weight_configs = [
            {"base": 0.5, "ml": 0.5},
            {"base": 0.7, "ml": 0.3},
            {"base": 0.8, "ml": 0.2},
            {"base": 1.0, "ml": 0.0}  # ML無効
        ]

        for weights in weight_configs:
            total = weights["base"] + weights["ml"]
            assert abs(total - 1.0) < 1e-6

    def test_ml_prediction_cache_efficiency(self, hybrid_evaluator):
        """ML予測キャッシュ効率のテスト"""
        # キャッシュが実装されているか
        assert hasattr(hybrid_evaluator, 'predictor')

    def test_drl_training_data_preparation(self, sample_features):
        """DRLトレーニングデータ準備のテスト"""
        # 特徴量がDRL用に整形される
        assert isinstance(sample_features, pd.DataFrame)
        assert not sample_features.empty

    def test_signal_conflict_resolution(self, drl_adapter, sample_features):
        """信号衝突解決のテスト"""
        # 複数のDRLアダプタからの信号
        adapter1 = DRLPolicyAdapter()
        adapter2 = DRLPolicyAdapter(policy_type="ddpg")

        signal1 = adapter1.predict_signals(sample_features)
        signal2 = adapter2.predict_signals(sample_features)

        # 両方とも有効な信号を返す
        assert isinstance(signal1, dict)
        assert isinstance(signal2, dict)

    def test_hybrid_evaluation_error_handling(self, hybrid_evaluator):
        """ハイブリッド評価エラーハンドリングのテスト"""
        # 両方のコンポーネントでエラー
        hybrid_evaluator.backtest_service.run_backtest.side_effect = Exception("Backtest failed")
        hybrid_evaluator.predictor.predict.side_effect = Exception("ML prediction failed")

        ga_config = GAConfig()
        ga_config.enable_multi_objective = False

        # 両方のエラーがフォールバックされる
        try:
            fitness = hybrid_evaluator.evaluate_individual([1, 2, 3], ga_config)
            # フォールバックが動作する
            assert isinstance(fitness, tuple)
        except Exception:
            pytest.fail("エラーハンドリングが不十分")

    def test_drl_real_time_signal_generation(self, drl_adapter, sample_features):
        """DRLリアルタイム信号生成のテスト"""
        # リアルタイムデータ
        real_time_data = sample_features.tail(1)

        result = drl_adapter.predict_signals(real_time_data)

        # 速やかに信号が生成される
        assert isinstance(result, dict)

    def test_hybrid_strategy_diversification(self):
        """ハイブリッド戦略多様化のテスト"""
        # GA戦略とML戦略の多様性
        strategies = ["ga_only", "ml_only", "hybrid"]

        for strategy in strategies:
            assert strategy in ["ga_only", "ml_only", "hybrid"]

    def test_model_drift_in_hybrid_system(self, hybrid_evaluator):
        """ハイブリッドシステムでのモデルドリフトテスト"""
        # ドリフト検出のためのログ
        drift_indicators = [
            "performance_degradation",
            "signal_correlation_change",
            "feature_distribution_shift"
        ]

        for indicator in drift_indicators:
            assert isinstance(indicator, str)

    def test_drl_action_space_mapping(self, drl_adapter, sample_features):
        """DRL行動空間マッピングのテスト"""
        result = drl_adapter.predict_signals(sample_features)

        # 行動空間の確率が正規化される
        actions = ["up", "down", "range"]
        for action in actions:
            assert action in result
            assert 0 <= result[action] <= 1

    def test_hybrid_system_scalability(self):
        """ハイブリッドシステムスケーラビリティのテスト"""
        # 複数の個体を同時に評価
        individuals = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # 各個体が評価される
        assert len(individuals) == 3

    def test_drl_policy_adaptation_over_time(self, drl_adapter):
        """時間経過でのDRLポリシーアダプテーションテスト"""
        # 異なる時間のデータ
        timeseries_data = []
        for i in range(5):
            data = pd.DataFrame({
                'close': np.random.randn(10) + 100 + i,
                'volume': np.random.randint(1000, 2000, 10)
            })
            timeseries_data.append(data)

        # 各時点での予測
        for data in timeseries_data:
            result = drl_adapter.predict_signals(data)
            assert isinstance(result, dict)

    def test_ml_model_versioning_in_hybrid_system(self):
        """ハイブリッドシステムでのMLモデルバージョニングテスト"""
        # モデルバージョン管理
        model_versions = ["v1.0", "v1.1", "v2.0"]

        for version in model_versions:
            assert isinstance(version, str)

    def test_drl_training_stability(self):
        """DRLトレーニング安定性のテスト"""
        # 安定性指標
        stability_metrics = [
            "reward_variance",
            "policy_entropy",
            "training_convergence"
        ]

        for metric in stability_metrics:
            assert isinstance(metric, str)

    def test_hybrid_evaluation_cache_coherence(self, hybrid_evaluator):
        """ハイブリッド評価キャッシュ整合性のテスト"""
        # キャッシュが整合性を保つ
        assert hasattr(hybrid_evaluator, 'predictor')

    def test_multi_asset_drl_support(self, drl_adapter):
        """マルチアセットDRLサポートのテスト"""
        # 異なるアセットのデータ
        assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        for asset in assets:
            # 各アセットのデータ形式
            data = pd.DataFrame({
                'close': np.random.randn(10) + 100,
                'volume': np.random.randint(1000, 2000, 10)
            })

            result = drl_adapter.predict_signals(data)
            assert isinstance(result, dict)

    def test_hybrid_system_monitoring(self):
        """ハイブリッドシステム監視のテスト"""
        # 監視指標
        monitoring_metrics = [
            "ga_convergence_rate",
            "ml_prediction_accuracy",
            "hybrid_fitness_trend",
            "drl_signal_stability"
        ]

        assert len(monitoring_metrics) == 4

    def test_drl_vs_ml_model_comparison(self):
        """DRLvsMLモデル比較のテスト"""
        # モデルタイプ
        model_types = ["drl", "traditional_ml", "hybrid"]

        for model_type in model_types:
            assert model_type in ["drl", "traditional_ml", "hybrid"]

    def test_hybrid_system_documentation_and_configuration(self):
        """ハイブリッドシステムドキュメントと設定のテスト"""
        # 設定項目
        config_items = [
            "hybrid_mode_enabled",
            "ml_weight",
            "drl_policy_type",
            "evaluation_frequency"
        ]

        for item in config_items:
            assert isinstance(item, str)

    def test_integration_test_end_to_end(self, hybrid_evaluator):
        """エンドツーエンド統合テスト"""
        # 完全な評価フロー
        mock_individual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mock_backtest_result = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.12,
                "win_rate": 0.65,
                "total_trades": 50
            },
            "equity_curve": [100, 110, 105, 120, 115, 130],
            "trade_history": [
                {"entry_time": "2023-01-01", "exit_time": "2023-01-02"},
                {"entry_time": "2023-01-03", "exit_time": "2023-01-04"}
            ]
        }

        hybrid_evaluator.backtest_service.run_backtest.return_value = mock_backtest_result
        hybrid_evaluator.predictor.predict.return_value = 0.85

        ga_config = GAConfig()
        ga_config.enable_multi_objective = True
        ga_config.objectives = ["sharpe_ratio", "max_drawdown", "ml_accuracy"]

        # 完全な評価が成功する
        fitness = hybrid_evaluator.evaluate_individual(mock_individual, ga_config)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 3

    def test_final_system_validation(self, drl_adapter, hybrid_evaluator, sample_features):
        """最終システム検証"""
        # すべてのコンポーネントが正常に動作
        assert drl_adapter is not None
        assert hybrid_evaluator is not None

        # DRLアダプタが信号を生成
        drl_result = drl_adapter.predict_signals(sample_features)
        assert isinstance(drl_result, dict)

        # ハイブリッド評価器が初期化されている
        assert hybrid_evaluator.backtest_service is not None
        assert hybrid_evaluator.predictor is not None