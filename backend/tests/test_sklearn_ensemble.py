"""
scikit-learnベースのアンサンブル実装のテスト

BaggingClassifierとStackingClassifierの動作確認を行います。
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from backend.app.services.ml.ensemble.bagging import BaggingEnsemble
from backend.app.services.ml.ensemble.stacking import StackingEnsemble


class TestSklearnEnsemble:
    """scikit-learnベースのアンサンブル実装のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成"""
        # 3クラス分類問題を作成
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # DataFrameとSeriesに変換
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        # 訓練・テストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test

    def test_bagging_ensemble_initialization(self):
        """BaggingEnsembleの初期化テスト"""
        config = {
            "n_estimators": 5,
            "bootstrap_fraction": 0.8,
            "base_model_type": "random_forest",
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config)
        
        assert ensemble.n_estimators == 5
        assert ensemble.max_samples == 0.8
        assert ensemble.base_model_type == "random_forest"
        assert ensemble.random_state == 42
        assert ensemble.bagging_classifier is None
        assert not ensemble.is_fitted

    def test_bagging_ensemble_fit_predict(self, sample_data):
        """BaggingEnsembleの学習・予測テスト"""
        X_train, X_test, y_train, y_test = sample_data
        
        config = {
            "n_estimators": 3,
            "bootstrap_fraction": 0.8,
            "base_model_type": "random_forest",
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config)
        
        # 学習
        result = ensemble.fit(X_train, y_train, X_test, y_test)
        
        # 学習結果の確認
        assert ensemble.is_fitted
        assert ensemble.bagging_classifier is not None
        assert "model_type" in result
        assert result["model_type"] == "BaggingClassifier"
        assert "sklearn_implementation" in result
        assert result["sklearn_implementation"] is True
        
        # 予測
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # 予測結果の確認
        assert len(y_pred) == len(X_test)
        assert y_pred_proba.shape == (len(X_test), 3)  # 3クラス分類
        assert np.all(y_pred_proba >= 0) and np.all(y_pred_proba <= 1)
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)  # 確率の合計が1

    def test_stacking_ensemble_initialization(self):
        """StackingEnsembleの初期化テスト"""
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "stack_method": "predict_proba",
            "random_state": 42
        }
        
        ensemble = StackingEnsemble(config)
        
        assert ensemble.base_model_types == ["random_forest", "gradient_boosting"]
        assert ensemble.meta_model_type == "logistic_regression"
        assert ensemble.cv_folds == 3
        assert ensemble.stack_method == "predict_proba"
        assert ensemble.random_state == 42
        assert ensemble.stacking_classifier is None
        assert not ensemble.is_fitted

    def test_stacking_ensemble_fit_predict(self, sample_data):
        """StackingEnsembleの学習・予測テスト"""
        X_train, X_test, y_train, y_test = sample_data
        
        config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "stack_method": "predict_proba",
            "random_state": 42
        }
        
        ensemble = StackingEnsemble(config)
        
        # 学習
        result = ensemble.fit(X_train, y_train, X_test, y_test)
        
        # 学習結果の確認
        assert ensemble.is_fitted
        assert ensemble.stacking_classifier is not None
        assert "model_type" in result
        assert result["model_type"] == "StackingClassifier"
        assert "sklearn_implementation" in result
        assert result["sklearn_implementation"] is True
        
        # 予測
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # 予測結果の確認
        assert len(y_pred) == len(X_test)
        assert y_pred_proba.shape == (len(X_test), 3)  # 3クラス分類
        assert np.all(y_pred_proba >= 0) and np.all(y_pred_proba <= 1)
        assert np.allclose(y_pred_proba.sum(axis=1), 1.0)  # 確率の合計が1

    def test_meta_model_creation(self):
        """メタモデル作成のテスト"""
        config = {
            "base_models": ["random_forest"],
            "meta_model": "logistic_regression",
            "random_state": 42
        }
        
        ensemble = StackingEnsemble(config)
        meta_model = ensemble._create_meta_model()
        
        # LogisticRegressionが作成されることを確認
        from sklearn.linear_model import LogisticRegression
        assert isinstance(meta_model, LogisticRegression)

    def test_feature_importance_bagging(self, sample_data):
        """BaggingEnsembleの特徴量重要度テスト"""
        X_train, X_test, y_train, y_test = sample_data
        
        config = {
            "n_estimators": 3,
            "base_model_type": "random_forest",
            "random_state": 42
        }
        
        ensemble = BaggingEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        # 特徴量重要度の取得
        importance = ensemble.get_feature_importance()
        
        if importance is not None:  # RandomForestは特徴量重要度を持つ
            assert isinstance(importance, dict)
            assert len(importance) == len(X_train.columns)
            assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_feature_importance_stacking(self, sample_data):
        """StackingEnsembleの特徴量重要度テスト"""
        X_train, X_test, y_train, y_test = sample_data
        
        config = {
            "base_models": ["random_forest"],
            "meta_model": "random_forest",  # 特徴量重要度を持つメタモデル
            "cv_folds": 3,
            "random_state": 42
        }
        
        ensemble = StackingEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        # 特徴量重要度の取得（メタモデルの重要度）
        importance = ensemble.get_feature_importance()
        
        if importance is not None:
            assert isinstance(importance, dict)
            # ベースモデルの数だけ重要度がある
            assert len(importance) == len(config["base_models"])

    def test_error_handling_unfitted_model(self, sample_data):
        """未学習モデルでのエラーハンドリングテスト"""
        X_train, X_test, y_train, y_test = sample_data
        
        # BaggingEnsemble
        bagging_config = {"n_estimators": 3, "base_model_type": "random_forest"}
        bagging_ensemble = BaggingEnsemble(bagging_config)
        
        with pytest.raises(Exception):  # UnifiedModelError
            bagging_ensemble.predict(X_test)
        
        with pytest.raises(Exception):  # UnifiedModelError
            bagging_ensemble.predict_proba(X_test)
        
        # StackingEnsemble
        stacking_config = {"base_models": ["random_forest"], "meta_model": "logistic_regression"}
        stacking_ensemble = StackingEnsemble(stacking_config)
        
        with pytest.raises(Exception):  # UnifiedModelError
            stacking_ensemble.predict(X_test)
        
        with pytest.raises(Exception):  # UnifiedModelError
            stacking_ensemble.predict_proba(X_test)

    def test_performance_comparison(self, sample_data):
        """性能比較テスト（参考）"""
        X_train, X_test, y_train, y_test = sample_data
        
        # BaggingEnsemble
        bagging_config = {
            "n_estimators": 5,
            "base_model_type": "random_forest",
            "random_state": 42
        }
        bagging_ensemble = BaggingEnsemble(bagging_config)
        bagging_result = bagging_ensemble.fit(X_train, y_train, X_test, y_test)
        
        # StackingEnsemble
        stacking_config = {
            "base_models": ["random_forest", "gradient_boosting"],
            "meta_model": "logistic_regression",
            "cv_folds": 3,
            "random_state": 42
        }
        stacking_ensemble = StackingEnsemble(stacking_config)
        stacking_result = stacking_ensemble.fit(X_train, y_train, X_test, y_test)
        
        # 両方とも学習が完了していることを確認
        assert bagging_ensemble.is_fitted
        assert stacking_ensemble.is_fitted
        
        # 精度が合理的な範囲にあることを確認（0.3以上）
        if "accuracy" in bagging_result:
            assert bagging_result["accuracy"] > 0.3
        if "accuracy" in stacking_result:
            assert stacking_result["accuracy"] > 0.3
        
        print(f"Bagging accuracy: {bagging_result.get('accuracy', 'N/A')}")
        print(f"Stacking accuracy: {stacking_result.get('accuracy', 'N/A')}")


if __name__ == "__main__":
    # 簡単な動作確認
    pytest.main([__file__, "-v"])
