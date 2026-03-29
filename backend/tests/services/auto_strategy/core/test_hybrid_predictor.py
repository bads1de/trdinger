"""
HybridPredictor Tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from app.services.auto_strategy.core.hybrid.hybrid_predictor import (
    HybridPredictor,
    MLPredictionError,
    RuntimeModelPredictorAdapter,
)


class TestHybridPredictor:
    """HybridPredictorのテスト"""

    @pytest.fixture
    def mock_training_service_cls(self):
        with patch(
            "app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor._resolve_training_service_cls"
        ) as mock_cls:
            # Create a mock service class that returns a mock service instance
            service_instance = Mock()
            service_instance.generate_forecast.return_value = {
                "forecast_log_rv": 0.75,
                "forecast_vol": float(np.exp(0.75)),
                "gate_open": True,
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
            "app.services.auto_strategy.core.hybrid.hybrid_predictor.HybridPredictor._resolve_model_manager"
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
        """単一モデルでの予測（ボラ回帰）"""
        df = pd.DataFrame({"close": [100, 101]})
        result = predictor.predict(df)

        assert "forecast_log_rv" in result
        assert "gate_open" in result

    def test_predict_ensemble_average(
        self, mock_training_service_cls, mock_model_manager
    ):
        """アンサンブル予測（平均化、ボラ回帰）"""
        # Setup mock to return different values for different instances
        service1 = Mock()
        service1.generate_forecast.return_value = {
            "forecast_log_rv": 0.8,
            "forecast_vol": float(np.exp(0.8)),
            "gate_open": True,
        }
        service1.trainer.is_trained = True

        service2 = Mock()
        service2.generate_forecast.return_value = {
            "forecast_log_rv": 0.6,
            "forecast_vol": float(np.exp(0.6)),
            "gate_open": True,
        }
        service2.trainer.is_trained = True

        mock_training_service_cls.side_effect = [service1, service2]

        predictor = HybridPredictor(model_types=["m1", "m2"])

        df = pd.DataFrame({"close": [100]})
        result = predictor.predict(df)

        assert result["forecast_log_rv"] == pytest.approx(0.7)

    def test_predict_forecast_mode(self, predictor):
        """forecast モードの予測"""
        predictor.service.generate_forecast.return_value = {
            "forecast_log_rv": 0.85,
            "forecast_vol": float(np.exp(0.85)),
            "gate_open": True,
        }

        df = pd.DataFrame({"close": [100]})
        result = predictor.predict(df)

        assert "forecast_log_rv" in result
        assert result["gate_open"] is True

    def test_predict_empty_input(self, predictor):
        """空入力のハンドリング"""
        with pytest.raises(MLPredictionError, match="特徴量DataFrameが空"):
            predictor.predict(pd.DataFrame())

    def test_predict_model_error(self, predictor):
        """モデルエラー時は予測失敗として扱う"""
        predictor.service.generate_forecast.side_effect = Exception("Model error")

        with pytest.raises(MLPredictionError, match="予測失敗"):
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

        assert result["forecast_log_rv"] == 0.0
        assert result["gate_open"] is True

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

    def test_load_latest_models_uses_model_type_pattern(
        self, predictor, mock_model_manager
    ):
        """単一モデルでは model_type を使って最新モデルを解決する"""
        predictor.service.load_model.return_value = True

        assert predictor.load_latest_models() is True

        mock_model_manager.get_latest_model.assert_called_with(
            "lightgbm",
            metadata_filters={
                "task_type": "volatility_regression",
                "target_kind": "log_realized_vol",
            },
        )
        predictor.service.load_model.assert_called_once_with("/path/to/model.pkl")

    def test_get_available_models(self):
        """利用可能モデルの取得"""
        models = HybridPredictor.get_available_models()
        assert "lightgbm" in models  # As returned by mock

    def test_get_model_info(self, predictor):
        """モデル情報の取得"""
        info = predictor.get_model_info()
        assert info["status"] == "trained"

    def test_normalise_prediction(self):
        """予測値の正規化（ボラ回帰）"""
        pred = {"forecast_log_rv": 0.75}
        norm = HybridPredictor._normalise_prediction(pred)
        assert norm["forecast_log_rv"] == pytest.approx(0.75)

        pred = {"forecast_log_rv": 1.5}
        norm = HybridPredictor._normalise_prediction(pred)
        assert norm["forecast_log_rv"] == 1.5

        pred = {"forecast_log_rv": -0.1}
        norm = HybridPredictor._normalise_prediction(pred)
        assert norm["forecast_log_rv"] == pytest.approx(-0.1)

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


class TestRuntimeModelPredictorAdapter:
    """保存済みモデルを runtime predictor として扱う薄いアダプタのテスト"""

    def test_predict_uses_loaded_model_artifacts(self):
        """model/scaler/feature_columns から forecast を推論できる"""
        model = Mock()
        model.predict.return_value = np.array([0.6, 0.8])
        model.is_trained = True

        adapter = RuntimeModelPredictorAdapter(
            {
                "model": model,
                "scaler": None,
                "feature_columns": ["close", "volume"],
                "metadata": {
                    "task_type": "volatility_regression",
                    "target_kind": "log_realized_vol",
                    "gate_cutoff_log_rv": 0.7,
                },
            }
        )

        result = adapter.predict(
            pd.DataFrame(
                {
                    "close": [100.0, 101.0],
                    "volume": [10.0, 11.0],
                    "ignored": [1.0, 2.0],
                }
            )
        )

        assert adapter.is_trained() is True
        assert result["forecast_log_rv"] == pytest.approx(0.8)
        assert result["gate_open"] is True
        model.predict.assert_called_once()

    def test_rejects_legacy_model_without_volatility_metadata(self):
        """旧モデル資産は task metadata が無ければ runtime 採用しない"""
        model = Mock()
        model.predict.return_value = np.array([0.8])
        model.is_trained = True

        adapter = RuntimeModelPredictorAdapter(
            {
                "model": model,
                "scaler": None,
                "feature_columns": ["close"],
                "metadata": {},
            }
        )

        result = adapter.predict(pd.DataFrame({"close": [100.0]}))

        assert adapter.is_trained() is False
        assert result["forecast_log_rv"] == 0.0
        assert result["gate_open"] is True
        model.predict.assert_not_called()
