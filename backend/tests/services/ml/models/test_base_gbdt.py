from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.ml.models.base_gradient_boosting_model import (
    BaseGradientBoostingModel,
)
from app.utils.error_handler import ModelError


# テスト用の具象クラス
class MockGBDT(BaseGradientBoostingModel):
    ALGORITHM_NAME = "mock_gbdt"

    def _create_dataset(self, X, y=None, sample_weight=None):
        return MagicMock()

    def _get_model_params(self, num_classes, **kwargs):
        return {"param": 1}

    def _train_internal(
        self, train_data, valid_data, params, early_stopping_rounds=None, **kwargs
    ):
        m = MagicMock()
        m.best_iteration = 10
        return m

    def _get_prediction_proba(self, data):
        return np.array([0.5])

    def _prepare_input_for_prediction(self, X):
        return X

    def _predict_raw(self, data):
        return np.array([0.7])  # 回帰予測値


class TestBaseGradientBoostingModel:
    @pytest.fixture
    def model(self):
        return MockGBDT()

    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([1.5, 3.0, 2.5, 4.5])  # 回帰用の連続値
        return X, y

    def test_fit_flow(self, model, sample_data):
        """fitメソッドの基本フローと入力整形"""
        X, y = sample_data
        # 1. 正常系
        model.fit(X, y)
        assert model.is_trained is True
        assert model.feature_columns == ["f1", "f2"]
        # 回帰タスクでは classes_ は None
        assert model.classes_ is None

        # 2. NumPy入力
        model.fit(X.values, y.values)
        assert model.is_trained is True

    def test_fit_with_early_stopping_split(self, model, sample_data):
        """早期終了指定時の自動データ分割"""
        # データ数を増やして分割可能にする
        X = pd.DataFrame(np.random.randn(20, 2))
        y = pd.Series(np.random.randn(20))  # 回帰用の連続値

        with patch.object(model, "_train_model_impl") as mock_train:
            model.fit(X, y, early_stopping_rounds=5)
            # 引数を確認
            args, kwargs = mock_train.call_args
            # X_train, X_val, y_train, y_val が渡されているはず
            assert args[0].shape[0] == 16  # 80%
            assert args[1].shape[0] == 4  # 20%

    def test_predict_and_proba(self, model, sample_data):
        """予測インターフェース（回帰）"""
        X, y = sample_data
        model.fit(X, y)

        # 1. predict_proba (回帰では (n, 1) を返す)
        preds = model.predict_proba(X.iloc[:1])
        assert preds.shape == (1, 1)
        assert pytest.approx(float(preds[0, 0])) == 0.7

        # 2. predict (回帰では連続値を返す)
        preds = model.predict(X.iloc[:1])
        assert preds[0] == 0.7

    def test_get_feature_importance(self, model, sample_data):
        """重要度取得の委譲"""
        X, y = sample_data
        model.fit(X, y)

        with patch(
            "app.services.ml.models.base_gradient_boosting_model.get_feature_importance_unified",
            return_value={"f1": 10.0},
        ) as mock_util:
            res = model.get_feature_importance()
            assert res["f1"] == 10.0
            mock_util.assert_called_once()

    def test_fit_error_handling(self, model, sample_data):
        """エラー発生時のModelErrorへのラップ"""
        X, y = sample_data
        with patch.object(
            model, "_train_model_impl", side_effect=ValueError("Internal error")
        ):
            with pytest.raises(ModelError, match="fit失敗"):
                model.fit(X, y)
