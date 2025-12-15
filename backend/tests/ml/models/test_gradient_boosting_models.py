import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from backend.app.services.ml.models.lightgbm import LightGBMModel
from backend.app.services.ml.models.xgboost import XGBoostModel
from backend.app.services.ml.models.catboost import CatBoostModel


class TestGradientBoostingModels:
    """勾配ブースティングモデルのテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータを生成"""
        X = pd.DataFrame({"f1": np.random.rand(10), "f2": np.random.rand(10)})
        y = pd.Series(np.random.randint(0, 2, 10))
        return X, y

    def test_lightgbm_predict_logic(self, sample_data):
        """BaseGradientBoostingModelを経由したLightGBMModelの予測ロジックをテスト"""
        X, y = sample_data
        model = LightGBMModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()
        model.model.best_iteration = 100

        # 予測出力（確率）をモック化
        # ケース1: 二値分類 (LightGBMからの1次元配列)
        mock_probas = np.array([0.1, 0.8, 0.4, 0.6, 0.2, 0.9, 0.3, 0.7, 0.1, 0.9])
        model.model.predict.return_value = mock_probas

        # predict_proba のテスト
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.allclose(probas[:, 1], mock_probas)
        assert np.allclose(probas[:, 0], 1 - mock_probas)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = (mock_probas > 0.5).astype(int)
        assert np.array_equal(preds, expected_preds)

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

        # 予測出力をモック化
        # ケース1: 二値分類 (XGBoostからの1次元配列)
        mock_probas = np.array([0.2, 0.7, 0.3, 0.8, 0.1, 0.95, 0.4, 0.6, 0.15, 0.85])
        model.model.predict.return_value = mock_probas

        # predict_proba のテスト
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.allclose(probas[:, 1], mock_probas)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = (mock_probas > 0.5).astype(int)
        assert np.array_equal(preds, expected_preds)

        # _prepare_input_for_prediction がDMatrixを使用したか検証
        # xgb.DMatrixをモックせずにDMatrixの中身を確認するのは難しいが、
        # model.predictが何かで呼び出されたことは確認できる
        assert model.model.predict.called

    def test_multiclass_prediction(self, sample_data):
        """多クラス分類予測ロジックのテスト"""
        X, _ = sample_data
        model = LightGBMModel()
        model.is_trained = True
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()
        model.model.best_iteration = 100

        # 3クラス確率をモック化
        mock_probas = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 3 + [[0.3, 0.3, 0.4]]
        )  # 10サンプル
        model.model.predict.return_value = mock_probas

        # predict_proba のテスト
        probas = model.predict_proba(X)
        assert probas.shape == (10, 3)
        assert np.array_equal(probas, mock_probas)

        # predict のテスト
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = np.argmax(mock_probas, axis=1)
        assert np.array_equal(preds, expected_preds)

    def test_catboost_predict_logic(self, sample_data):
        """CatBoostModelの予測ロジックテスト (BaseGradientBoostingModelを継承)"""
        X, y = sample_data
        model = CatBoostModel()
        model.is_trained = True  # BaseGradientBoostingModelの規約に合わせる
        model.feature_columns = ["f1", "f2"]
        model.model = MagicMock()

        # predict_proba 出力をモック化
        mock_probas = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.75, 0.25],
                [0.35, 0.65],
            ]
        )
        model.model.predict_proba.return_value = mock_probas

        # predict_proba のテスト
        probas = model.predict_proba(X)
        assert probas.shape == (10, 2)
        assert np.array_equal(probas, mock_probas)

        # predict のテスト (predict_probaから派生)
        preds = model.predict(X)
        assert preds.shape == (10,)
        expected_preds = np.argmax(mock_probas, axis=1)
        assert np.array_equal(preds, expected_preds)

