"""
HybridPredictorのテストモジュール

MLTrainingServiceを使用したバッチ予測のテスト
TDD: テストファースト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.services.ml.exceptions import MLPredictionError, MLModelError


class TestHybridPredictor:
    """HybridPredictorのテストクラス"""

    @pytest.fixture
    def sample_features_df(self):
        """サンプル特徴量DataFrame"""
        n_samples = 100
        data = {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_ml_training_service(self):
        """Mock MLTrainingService"""
        service = Mock()
        
        # generate_signalsのモック
        service.generate_signals.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }
        
        # is_trainedのモック
        service.trainer = Mock()
        service.trainer.is_trained = True
        service.trainer.model = Mock()
        
        return service

    def test_predictor_initialization(self):
        """
        Predictor初期化テスト
        
        検証項目:
        - 正しく初期化される
        - MLTrainingServiceが設定される
        """
        
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        assert predictor is not None
        assert predictor.trainer_type == "single"
        assert predictor.model_type == "lightgbm"

    def test_predictor_initialization_with_config(self):
        """
        設定付きPredictor初期化テスト
        
        検証項目:
        - single_model_configが正しく設定される
        - 複数モデルタイプをサポート
        """
        
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        config = {
            "model_type": "xgboost",
            "n_estimators": 100,
            "learning_rate": 0.1,
        }
        
        predictor = HybridPredictor(
            trainer_type="single",
            single_model_config=config
        )
        
        assert predictor.single_model_config == config
        assert predictor.model_type == "xgboost"

    @patch("app.services.ml.ml_training_service.MLTrainingService")
    def test_predict_basic(self, mock_service_class, sample_features_df):
        """
        基本的な予測テスト
        
        検証項目:
        - 予測が正しく実行される
        - 確率形式で結果が返される
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        # モックの設定
        mock_service = Mock()
        mock_service.generate_signals.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }
        mock_service_class.return_value = mock_service
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        result = predictor.predict(sample_features_df)
        
        # 予測結果の検証
        assert "up" in result
        assert "down" in result
        assert "range" in result
        assert 0 <= result["up"] <= 1
        assert 0 <= result["down"] <= 1
        assert 0 <= result["range"] <= 1
        assert abs(sum(result.values()) - 1.0) < 0.01  # 合計が約1.0

    @patch("app.services.ml.ml_training_service.MLTrainingService")
    def test_predict_with_multiple_models(self, mock_service_class, sample_features_df):
        """
        複数モデルによる予測テスト
        
        検証項目:
        - 複数モデルの予測が平均化される
        - 各モデルの予測が正しく統合される
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        # 複数モデルの予測結果をモック
        mock_service = Mock()
        mock_service.generate_signals.side_effect = [
            {"up": 0.7, "down": 0.2, "range": 0.1},  # LightGBM
            {"up": 0.5, "down": 0.3, "range": 0.2},  # XGBoost
        ]
        mock_service_class.return_value = mock_service
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_types=["lightgbm", "xgboost"]  # 複数モデル
        )
        
        result = predictor.predict(sample_features_df)
        
        # 平均化された結果を検証
        assert "up" in result
        # (0.7 + 0.5) / 2 = 0.6
        assert abs(result["up"] - 0.6) < 0.01

    def test_predict_untrained_model_error(self, sample_features_df):
        """
        未学習モデルでの予測エラーテスト
        
        検証項目:
        - 未学習モデルで予測するとデフォルト値が返される
        - または適切なエラーメッセージが返される
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        # 未学習の状態で予測
        result = predictor.predict(sample_features_df)
        
        # デフォルト値が返されることを確認
        assert result == {"up": 0.33, "down": 0.33, "range": 0.34}

    def test_predict_with_invalid_features(self):
        """
        無効な特徴量での予測エラーテスト
        
        検証項目:
        - 空のDataFrameでMLPredictionErrorが発生する
        - Noneの特徴量でエラーが発生する
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        # 空のDataFrameでエラー
        with pytest.raises(MLPredictionError):
            predictor.predict(pd.DataFrame())
        
        # NoneでエラーNone
        with pytest.raises(MLPredictionError):
            predictor.predict(None)

    @patch("app.services.ml.ml_training_service.MLTrainingService")
    def test_load_model(self, mock_service_class):
        """
        モデルロードテスト
        
        検証項目:
        - ModelManagerを通じてモデルがロードされる
        - ロード後にis_trainedがTrueになる
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        mock_service = Mock()
        mock_service.load_model.return_value = True
        mock_service_class.return_value = mock_service
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        result = predictor.load_model("models/test_model.pkl")
        
        assert result is True
        assert mock_service.load_model.called

    @patch("app.services.ml.model_manager.ModelManager")
    def test_get_latest_model(self, mock_manager_class):
        """
        最新モデル取得テスト
        
        検証項目:
        - ModelManager.get_latest_modelが呼ばれる
        - 最新モデルが正しくロードされる
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        mock_manager = Mock()
        mock_manager.get_latest_model.return_value = "models/latest_model.pkl"
        mock_manager_class.return_value = mock_manager
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        model_path = predictor.get_latest_model()
        
        assert model_path == "models/latest_model.pkl"
        assert mock_manager.get_latest_model.called

    @patch("app.services.ml.ml_training_service.MLTrainingService")
    def test_predict_with_time_series_cv(
        self, mock_service_class, sample_features_df
    ):
        """
        時系列クロスバリデーションでの予測テスト
        
        検証項目:
        - _time_series_cross_validateが統合される
        - CV結果が予測に反映される
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        mock_service = Mock()
        mock_service.generate_signals.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }
        mock_service_class.return_value = mock_service
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm",
            use_time_series_cv=True
        )
        
        result = predictor.predict(sample_features_df)
        
        assert "up" in result
        assert mock_service.generate_signals.called

    def test_predict_batch(self, sample_features_df):
        """
        バッチ予測テスト
        
        検証項目:
        - 複数の特徴量DataFrameを一度に予測できる
        - 各バッチの予測結果が返される
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        predictor = HybridPredictor(
            trainer_type="single",
            model_type="lightgbm"
        )
        
        # バッチ予測
        features_batch = [sample_features_df.copy() for _ in range(3)]
        results = predictor.predict_batch(features_batch)
        
        assert len(results) == 3
        for result in results:
            assert "up" in result
            assert "down" in result
            assert "range" in result

    @patch("app.services.ml.ml_training_service.MLTrainingService")
    def test_predict_with_ensemble(self, mock_service_class, sample_features_df):
        """
        アンサンブルモデルでの予測テスト
        
        検証項目:
        - ensemble_configが正しく適用される
        - 複数モデルのアンサンブル予測が動作する
        """
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        mock_service = Mock()
        mock_service.generate_signals.return_value = {
            "up": 0.6,
            "down": 0.2,
            "range": 0.2,
        }
        mock_service_class.return_value = mock_service
        
        ensemble_config = {
            "models": ["lightgbm", "xgboost", "randomforest"],
            "voting": "soft",
        }
        
        predictor = HybridPredictor(
            trainer_type="ensemble",
            ensemble_config=ensemble_config
        )
        
        result = predictor.predict(sample_features_df)
        
        assert "up" in result
        assert mock_service.generate_signals.called

    def test_available_models_list(self):
        """
        利用可能なモデル一覧取得テスト
        
        検証項目:
        - get_available_single_models()が正しく動作する
        - LightGBM, XGBoost, CatBoost, RandomForestが含まれる
        """
        
        from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
        
        models = HybridPredictor.get_available_models()
        
        assert "lightgbm" in models
        assert "xgboost" in models
        assert "catboost" in models
        assert "randomforest" in models
