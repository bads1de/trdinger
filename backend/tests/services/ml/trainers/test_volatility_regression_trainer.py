"""
volatility_regression_trainer.py のテスト

app/services/ml/trainers/volatility_regression_trainer.py のテストモジュール
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from app.services.ml.trainers.volatility_regression_trainer import VolatilityRegressionTrainer
from app.utils.error_handler import ModelError


class TestVolatilityRegressionTrainer:
    """VolatilityRegressionTrainer クラスのテスト"""

    def test_initialization_default(self):
        """デフォルト設定で初期化"""
        trainer = VolatilityRegressionTrainer()
        assert trainer.model_type == "lightgbm"
        assert trainer.model_params == {}

    def test_initialization_custom_model_type(self):
        """カスタムモデルタイプで初期化"""
        trainer = VolatilityRegressionTrainer(model_type="xgboost")
        assert trainer.model_type == "xgboost"

    def test_initialization_custom_params(self):
        """カスタムパラメータで初期化"""
        params = {"n_estimators": 100, "learning_rate": 0.1}
        trainer = VolatilityRegressionTrainer(model_params=params)
        assert trainer.model_params == params

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_build_model_lightgbm(self, mock_lgbm):
        """LightGBMモデルの構築"""
        mock_model = MagicMock()
        mock_lgbm.return_value = mock_model
        trainer = VolatilityRegressionTrainer(model_type="lightgbm")

        model = trainer._build_model()

        mock_lgbm.assert_called_once_with(task_type="volatility_regression")
        assert model == mock_model

    @patch("app.services.ml.trainers.volatility_regression_trainer.XGBoostModel")
    def test_build_model_xgboost(self, mock_xgb):
        """XGBoostモデルの構築"""
        mock_model = MagicMock()
        mock_xgb.return_value = mock_model
        trainer = VolatilityRegressionTrainer(model_type="xgboost")

        model = trainer._build_model()

        mock_xgb.assert_called_once_with(task_type="volatility_regression")
        assert model == mock_model

    def test_build_model_unsupported(self):
        """サポートされていないモデルタイプ"""
        trainer = VolatilityRegressionTrainer(model_type="unsupported")

        with pytest.raises(ModelError, match="サポートされていない回帰モデル"):
            trainer._build_model()

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_build_model_with_custom_params(self, mock_lgbm):
        """カスタムパラメータでのモデル構築"""
        mock_model = MagicMock()
        mock_lgbm.return_value = mock_model
        params = {"n_estimators": 100, "learning_rate": 0.1}
        trainer = VolatilityRegressionTrainer(model_params=params)

        model = trainer._build_model()

        mock_lgbm.assert_called_once_with(
            n_estimators=100,
            learning_rate=0.1,
            task_type="volatility_regression",
        )
        assert model == mock_model

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_predict_without_training(self, mock_lgbm):
        """学習前の予測"""
        mock_model = MagicMock()
        mock_lgbm.return_value = mock_model
        trainer = VolatilityRegressionTrainer()
        trainer._model = None

        with pytest.raises(ModelError, match="学習済みモデルがありません"):
            trainer.predict(pd.DataFrame())

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_predict_with_trained_model(self, mock_lgbm):
        """学習済みモデルでの予測"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])
        mock_lgbm.return_value = mock_model
        trainer = VolatilityRegressionTrainer()
        trainer._model = mock_model

        features_df = pd.DataFrame({"feature1": [1, 2, 3]})
        result = trainer.predict(features_df)

        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
        mock_model.predict.assert_called_once_with(features_df)

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_basic(self, mock_lgbm):
        """基本的な学習実装"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1", "feature2"]
        mock_model.last_training_result = {"train_loss": 0.1}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        X_test = pd.DataFrame({"feature1": [5], "feature2": [6]})
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series([0.3])

        result = trainer._train_model_impl(
            X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1
        )

        assert trainer.is_trained is True
        assert trainer.feature_columns == ["feature1", "feature2"]
        assert result["algorithm"] == "lightgbm"
        assert result["train_samples"] == 2
        assert result["test_samples"] == 1
        assert result["feature_count"] == 2

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_without_test_data(self, mock_lgbm):
        """テストデータなしでの学習"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1"]
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = None
        y_train = pd.Series([0.1, 0.2])
        y_test = None

        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)

        assert result["test_samples"] == 0
        # eval_setがNoneであることを確認
        mock_model.fit.assert_called_once()
        call_kwargs = mock_model.fit.call_args.kwargs
        assert call_kwargs["eval_set"] is None

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_with_gate_cutoff(self, mock_lgbm):
        """ゲートカットオフ付きの学習"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1"]
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = pd.DataFrame({"feature1": [3]})
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series([0.3])

        result = trainer._train_model_impl(
            X_train,
            X_test,
            y_train,
            y_test,
            gate_cutoff_log_rv=0.5,
            gate_cutoff_vol=2.0,
        )

        assert result["gate_cutoff_log_rv"] == 0.5
        assert result["gate_cutoff_vol"] == 2.0

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_coerces_invalid_gate_cutoff(self, mock_lgbm):
        """壊れたゲートカットオフ値でも既定値へフォールバックする"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1"]
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = pd.DataFrame({"feature1": [3]})
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series([0.3])

        result = trainer._train_model_impl(
            X_train,
            X_test,
            y_train,
            y_test,
            gate_cutoff_log_rv={"invalid": True},
            gate_cutoff_vol={"invalid": True},
        )

        assert result["gate_cutoff_log_rv"] == 0.0
        assert result["gate_cutoff_vol"] == 1.0

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_empty_test_data(self, mock_lgbm):
        """空のテストデータでの学習"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1"]
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = pd.DataFrame()
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series()

        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)

        assert result["test_samples"] == 0

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_without_feature_columns(self, mock_lgbm):
        """feature_columnsがない場合の学習"""
        mock_model = MagicMock()
        mock_model.feature_columns = None
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = pd.DataFrame({"feature1": [3]})
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series([0.3])

        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)

        assert trainer.feature_columns == ["feature1"]
        assert result["feature_count"] == 1

    @patch("app.services.ml.trainers.volatility_regression_trainer.LightGBMModel")
    def test_train_model_impl_default_gate_cutoff(self, mock_lgbm):
        """デフォルトゲートカットオフの確認"""
        mock_model = MagicMock()
        mock_model.feature_columns = ["feature1"]
        mock_model.last_training_result = {}
        mock_lgbm.return_value = mock_model

        trainer = VolatilityRegressionTrainer()
        X_train = pd.DataFrame({"feature1": [1, 2]})
        X_test = pd.DataFrame({"feature1": [3]})
        y_train = pd.Series([0.1, 0.2])
        y_test = pd.Series([0.3])

        result = trainer._train_model_impl(X_train, X_test, y_train, y_test)

        assert result["gate_cutoff_log_rv"] == 0.0
        assert result["gate_cutoff_vol"] == 1.0
