"""
HybridPredictor Tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from backend.app.services.auto_strategy.core.hybrid_predictor import (
    HybridPredictor,
    MLPredictionError,
)


class TestHybridPredictor:
    """HybridPredictorのテスト"""

    @pytest.fixture
    def mock_training_service_cls(self):
        with patch(
            "backend.app.services.auto_strategy.core.hybrid_predictor.HybridPredictor._resolve_training_service_cls"
        ) as mock_cls:
            # Create a mock service class that returns a mock service instance
            service_instance = Mock()
            # Default behavior for service
            service_instance.generate_signals.return_value = {
                "up": 0.8,
                "down": 0.1,
                "range": 0.1,
            }
            service_instance.get_available_single_models.return_value = [
                "lightgbm",
                "rf",
            ]
            service_instance.get_training_status.return_value = {"status": "trained"}
            service_instance.trainer.is_trained = True

            mock_service_class = Mock(return_value=service_instance)
            mock_service_class.get_available_single_models.return_value = [
                "lightgbm",
                "rf",
            ]
            mock_cls.return_value = mock_service_class
            yield mock_service_class

    @pytest.fixture
    def mock_model_manager(self):
        with patch(
            "backend.app.services.auto_strategy.core.hybrid_predictor.HybridPredictor._resolve_model_manager"
        ) as mock_mgr:
            manager = Mock()
            manager.get_latest_model.return_value = "/path/to/model.pkl"
            mock_mgr.return_value = manager
            yield manager

    @pytest.fixture
    def predictor(self, mock_training_service_cls, mock_model_manager):
        return HybridPredictor(
            trainer_type="single",
            model_type="lightgbm",
            single_model_config={"param": 1},
        )

    def test_init_single_model(self, predictor, mock_training_service_cls):
        """単一モデルの初期化"""
        assert len(predictor.services) == 1
        assert predictor.model_type == "lightgbm"
        mock_training_service_cls.assert_called_once()
        # Verify call args
        call_kwargs = mock_training_service_cls.call_args[1]
        assert call_kwargs["trainer_type"] == "single"
        assert call_kwargs["single_model_config"]["model_type"] == "lightgbm"

    def test_init_multi_model(self, mock_training_service_cls, mock_model_manager):
        """複数モデルの初期化"""
        model_types = ["lightgbm", "rf"]
        predictor = HybridPredictor(
            trainer_type="single",  # Conceptually weird but code allows trainer_type to be passed even if overridden by logic?
            # Actually code says: if model_types and len>1: ...
            model_types=model_types,
            single_model_config={"param": 1},
        )

        assert len(predictor.services) == 2
        assert mock_training_service_cls.call_count == 2

    def test_predict_single(self, predictor):
        """単一モデルでの予測"""
        df = pd.DataFrame({"close": [100, 101]})
        result = predictor.predict(df)

        assert "up" in result
        assert "down" in result
        assert "range" in result
        # Expect normalized result
        assert pytest.approx(sum(result.values())) == 1.0

    def test_predict_ensemble_average(
        self, mock_training_service_cls, mock_model_manager
    ):
        """アンサンブル予測（平均化）"""
        # Setup mock to return different values for different instances
        service1 = Mock()
        service1.generate_signals.return_value = {"up": 0.8, "down": 0.1, "range": 0.1}
        service1.trainer.is_trained = True

        service2 = Mock()
        service2.generate_signals.return_value = {"up": 0.4, "down": 0.4, "range": 0.2}
        service2.trainer.is_trained = True

        mock_training_service_cls.side_effect = [service1, service2]

        predictor = HybridPredictor(model_types=["m1", "m2"])

        df = pd.DataFrame({"close": [100]})
        result = predictor.predict(df)

        # Average: up=(0.8+0.4)/2 = 0.6, down=(0.1+0.4)/2 = 0.25, range=(0.1+0.2)/2 = 0.15
        assert result["up"] == pytest.approx(0.6)
        assert result["down"] == pytest.approx(0.25)
        assert result["range"] == pytest.approx(0.15)

    def test_predict_volatility_mode(self, predictor):
        """ボラティリティモードの予測"""
        predictor.service.generate_signals.return_value = {"trend": 0.7, "range": 0.3}

        df = pd.DataFrame({"close": [100]})
        result = predictor.predict(df)

        assert "trend" in result
        assert "range" in result
        assert "up" not in result
        assert pytest.approx(sum(result.values())) == 1.0

    def test_predict_empty_input(self, predictor):
        """空入力のハンドリング"""
        with pytest.raises(MLPredictionError, match="特徴量DataFrameが空"):
            predictor.predict(pd.DataFrame())

    def test_predict_model_error(self, predictor):
        """モデルエラー時のデフォルトフォールバック"""
        predictor.service.generate_signals.side_effect = Exception("Model error")

        with pytest.raises(MLPredictionError, match="予測に失敗しました"):
            predictor.predict(pd.DataFrame({"a": [1]}))

    def test_predict_untrained_model(self, predictor):
        """未学習モデルのハンドリング"""
        predictor.service.trainer.is_trained = False

        # When single model is untrained, it logs warning and returns default
        # BUT code effectively mocks service_is_trained via getattr check on trainer.
        # Let's verify _service_is_trained logic.
        # If is_trained is False, returns default prediction.

        df = pd.DataFrame({"close": [100]})
        result = predictor.predict(df)

        # Default prediction for "direction" (default mode in _default_prediction)
        # {"up": 0.33, "down": 0.33, "range": 0.34} normalized to sum to 1?
        # 0.33+0.33+0.34 = 1.0. Correct.
        assert result["up"] == 0.33
        assert result["down"] == 0.33
        assert result["range"] == 0.34

    def test_predict_batch(self, predictor):
        """バッチ予測"""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        results = predictor.predict_batch([df1, df2])
        assert len(results) == 2
        assert isinstance(results[0], dict)

    def test_load_model(self, predictor):
        """モデルロード"""
        predictor.service.load_model.return_value = True
        assert predictor.load_model("path") is True

        predictor.service.load_model.return_value = False
        assert predictor.load_model("path") is False

    def test_get_available_models(self):
        """利用可能モデルの取得"""
        models = HybridPredictor.get_available_models()
        assert "lightgbm" in models  # As returned by mock

    def test_get_model_info(self, predictor):
        """モデル情報の取得"""
        info = predictor.get_model_info()
        assert info["status"] == "trained"

    def test_normalise_prediction(self):
        """予測値の正規化"""
        # Zero sum case
        pred = {"up": 0.0, "down": 0.0, "range": 0.0}
        norm = HybridPredictor._normalise_prediction(pred)
        assert norm["up"] == pytest.approx(1 / 3)

        # Normal case
        pred = {"up": 10, "down": 10, "range": 0}  # Sum 20
        norm = HybridPredictor._normalise_prediction(pred)
        assert norm["up"] == 0.5
        assert norm["range"] == 0.0

    def test_get_latest_model(self, predictor, mock_model_manager):
        """最新モデルパスの取得"""
        # mock_model_manager is already configured in fixture to return "/path/to/model.pkl"
        assert predictor.get_latest_model() == "/path/to/model.pkl"

        # Test error handling
        mock_model_manager.get_latest_model.side_effect = Exception("DB Error")
        assert predictor.get_latest_model() is None

    def test_is_trained(self, predictor):
        """モデル学習状態のチェック"""
        # Default mock service is trained
        assert predictor.is_trained() is True

        # Set untained
        predictor.service.trainer.is_trained = False
        assert predictor.is_trained() is False

        predictor.service.trainer.is_trained = True
        predictor.service.trainer.model = None
        assert predictor.is_trained() is False

    def test_get_model_info_multi(self, mock_training_service_cls):
        """複数モデル時のモデル情報取得"""
        service1 = Mock()
        service1.get_training_status.return_value = {"status": "trained", "acc": 0.8}
        service2 = Mock()
        service2.get_training_status.return_value = {"status": "trained", "acc": 0.85}

        mock_training_service_cls.side_effect = [service1, service2]

        predictor = HybridPredictor(model_types=["m1", "m2"])

        info = predictor.get_model_info()
        assert info["trainer_type"] == "multi_model"
        assert info["model_count"] == 2
        assert len(info["models"]) == 2
        assert info["models"][0]["acc"] == 0.8
        assert info["models"][1]["acc"] == 0.85
