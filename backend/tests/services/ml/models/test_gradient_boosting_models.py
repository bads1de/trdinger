from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.ml.models.catboost import CatBoostModel
from app.services.ml.models.lightgbm import LightGBMModel
from app.services.ml.models.xgboost import XGBoostModel


class TestGradientBoostingModels:
    """勾配ブースティングモデルのテストクラス（回帰）"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータを生成"""
        X = pd.DataFrame({"f1": np.random.rand(10), "f2": np.random.rand(10)})
        y = pd.Series(np.random.randn(10))  # 回帰用の連続値
        return X, y

    def test_lightgbm_predict_logic(self, sample_data):
        """BaseGradientBoostingModelを経由したLightGBMModelの予測ロジックをテスト"""
        X, y = sample_data
        model = LightGBMModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()
        model.model.best_iteration = 100

        # 予測出力（回帰値）をモック化
        mock_preds = np.array([0.1, 0.8, 0.4, 0.6, 0.2, 0.9, 0.3, 0.7, 0.1, 0.9])
        model.model.predict.return_value = mock_preds

        # predict_proba のテスト（回帰では (n, 1) を返す）
        probas = model.predict_proba(X)
        assert probas.shape == (10, 1)
        assert np.allclose(probas[:, 0], mock_preds)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        assert np.array_equal(preds, mock_preds)

        # _prepare_input_for_prediction が呼び出されたか検証（引数チェックにより暗黙的に確認）
        # LightGBMModelはXを直接渡す
        model.model.predict.assert_called_with(X, num_iteration=100)

    def test_xgboost_predict_logic(self, sample_data):
        """BaseGradientBoostingModelを経由したXGBoostModelの予測ロジックをテスト"""
        X, y = sample_data
        model = XGBoostModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.feature_names = ["f1", "f2"]
        model.model = MagicMock()

        # 予測出力をモック化（回帰値）
        mock_preds = np.array([0.2, 0.7, 0.3, 0.8, 0.1, 0.95, 0.4, 0.6, 0.15, 0.85])
        model.model.predict.return_value = mock_preds

        # predict_proba のテスト（回帰では (n, 1) を返す）
        probas = model.predict_proba(X)
        assert probas.shape == (10, 1)
        assert np.allclose(probas[:, 0], mock_preds)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        assert np.array_equal(preds, mock_preds)

        # _prepare_input_for_prediction がDMatrixを使用したか検証
        # xgb.DMatrixをモックせずにDMatrixの中身を確認するのは難しいが、
        # model.predictが何かで呼び出されたことは確認できる
        assert model.model.predict.called

    def test_catboost_predict_logic(self, sample_data):
        """CatBoostModelの予測ロジックテスト (BaseGradientBoostingModelを継承)"""
        X, y = sample_data
        model = CatBoostModel()
        model.is_trained = True  # BaseGradientBoostingModelの規約に合わせる
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()

        # predict 出力をモック化（回帰なので1次元配列）
        mock_preds = np.array([0.7, 0.2, 0.6, 0.3, 0.9, 0.4, 0.8, 0.1, 0.75, 0.35])
        model.model.predict.return_value = mock_preds

        # predict_proba のテスト（回帰では (n, 1) を返す）
        probas = model.predict_proba(X)
        assert probas.shape == (10, 1)
        assert np.allclose(probas.flatten(), mock_preds)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        assert np.allclose(preds, mock_preds)
