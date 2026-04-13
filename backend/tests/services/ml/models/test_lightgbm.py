import pytest
import pandas as pd
import numpy as np
from app.services.ml.models.lightgbm import LightGBMModel
from app.utils.error_handler import ModelError

class TestLightGBMModel:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(np.random.randn(n, 2), columns=["f1", "f2"])
        # 回帰用の連続ターゲット
        y = X["f1"] + X["f2"] + np.random.randn(n) * 0.1
        return X, y

    def test_fit_success(self, sample_data):
        """学習の成功テスト"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)

        # 1. 学習
        model.fit(X, y)
        assert model.is_trained is True
        assert model.model is not None
        # 回帰タスクでは classes_ は None
        assert model.classes_ is None

        # 2. 結果辞書の確認
        assert len(model.feature_columns) == 2

    def test_predict_proba(self, sample_data):
        """予測値の取得テスト（回帰）"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)

        preds = model.predict_proba(X.iloc[:5])
        # 回帰なので (n, 1) の形状
        assert preds.shape == (5, 1)

    def test_predict_continuous(self, sample_data):
        """連続値予測のテスト"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)

        preds = model.predict(X.iloc[:5])
        assert len(preds) == 5
        # 連続値であることを確認
        assert preds.dtype == np.float64

    def test_early_stopping(self, sample_data):
        """Early Stoppingの動作確認"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=100)
        # eval_setなしで early_stopping_rounds を指定すると自動分割される
        model.fit(X, y, early_stopping_rounds=5)

        assert model.is_trained is True
        # best_iteration が記録されているはず
        assert hasattr(model.model, "best_iteration")

    def test_not_trained_error(self, sample_data):
        """未学習時のエラー"""
        X, y = sample_data
        model = LightGBMModel()
        with pytest.raises(ModelError, match="学習済みモデルがありません"):
            model.predict(X)

    def test_get_feature_importance(self, sample_data):
        """特徴量重要度の取得"""
        X, y = sample_data
        model = LightGBMModel()
        model.fit(X, y)

        importance = model.get_feature_importance(top_n=2)
        assert len(importance) <= 2
        assert "f1" in importance or "f2" in importance
