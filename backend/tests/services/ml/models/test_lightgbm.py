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
        # 線形分離可能なターゲット
        y = (X["f1"] + X["f2"] > 0).astype(int)
        return X, y

    def test_fit_success(self, sample_data):
        """学習の成功テスト"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)
        
        # 1. 学習
        result = model.fit(X, y)
        assert model.is_trained is True
        assert model.model is not None
        assert model.classes_ is not None
        
        # 2. 結果辞書の確認
        # BaseGradientBoostingModel._train_model_impl の戻り値を確認するために直接呼ぶか、
        # fit の副作用を確認
        assert len(model.feature_columns) == 2

    def test_predict_proba(self, sample_data):
        """予測確率の取得テスト"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)
        
        probs = model.predict_proba(X.iloc[:5])
        # 二値分類なので (n, 2) の形状
        assert probs.shape == (5, 2)
        assert np.all((probs >= 0) & (probs <= 1))
        # 確率の合計が1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_class(self, sample_data):
        """クラス予測のテスト"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)
        
        preds = model.predict(X.iloc[:5])
        assert len(preds) == 5
        assert set(np.unique(preds)).issubset({0, 1})

    def test_early_stopping(self, sample_data):
        """Early Stoppingの動作確認"""
        X, y = sample_data
        model = LightGBMModel(n_estimators=100)
        # eval_setなしで early_stopping_rounds を指定すると自動分割される
        model.fit(X, y, early_stopping_rounds=5)
        
        assert model.is_trained is True
        # best_iteration が記録されているはず
        assert hasattr(model.model, "best_iteration")

    def test_class_weight(self, sample_data):
        """不均衡データ対応 (class_weight) のテスト"""
        X, y = sample_data
        # 意図的に極端な重みを指定
        model = LightGBMModel()
        model.fit(X, y, class_weight={0: 0.1, 1: 0.9})
        assert model.is_trained is True

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

    def test_multiclass_support(self):
        """多クラス分類のテスト"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 2), columns=["f1", "f2"])
        y = np.random.randint(0, 3, 100) # 3クラス
        
        model = LightGBMModel(n_estimators=10)
        model.fit(X, y)
        
        assert model.is_trained is True
        probs = model.predict_proba(X.iloc[:5])
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
