import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.models.catboost import CatBoostModel

class TestCatBoostModel:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
        y = pd.Series([0, 1, 0, 1])
        return X, y

    def test_handle_class_weight(self):
        """CatBoost固有のclass_weight処理"""
        model = CatBoostModel()
        # 1. Balanced
        res = model._handle_class_weight_for_catboost("balanced", {})
        assert res["auto_class_weights"] == "Balanced"
        
        # 2. Dict
        res = model._handle_class_weight_for_catboost({0: 1, 1: 5}, {})
        assert res["class_weights"] == [1, 5]

    def test_create_dataset(self, sample_data):
        """データセット作成のテスト"""
        X, y = sample_data
        model = CatBoostModel()
        
        # CatBoostは (X_values, y_values) のタプルを返す
        ds = model._create_dataset(X, y)
        assert isinstance(ds, tuple)
        assert isinstance(ds[0], np.ndarray)
        assert ds[0].shape == (4, 2)

    def test_fit_and_predict_mock(self, sample_data):
        """CatBoostの学習と予測（内部をモック化）"""
        X, y = sample_data
        model = CatBoostModel(iterations=10)
        
        # cb.CatBoostClassifier をモック化
        with patch('app.services.ml.models.catboost.cb.CatBoostClassifier') as mock_cb:
            mock_instance = mock_cb.return_value
            mock_instance.predict_proba.return_value = np.array([[0.2, 0.8]])
            
            # 学習実行
            model.fit(X, y)
            
            assert model.is_trained is True
            # パラメータが正しく渡されたか
            args, kwargs = mock_cb.call_args
            assert kwargs["iterations"] == 10
            
            # 予測
            probs = model.predict_proba(X.iloc[:1])
            assert probs[0, 1] == 0.8
