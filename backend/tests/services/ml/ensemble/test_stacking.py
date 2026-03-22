import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open

import pytest

from app.services.ml.ensemble.stacking import StackingEnsemble


class TestStackingEnsemble:
    """StackingEnsemble のテスト (ファイル操作をモック化)"""

    @pytest.fixture
    def config(self):
        return {
            "base_models": ["lightgbm", "xgboost"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "cv_strategy": "kfold",
            "passthrough": False,
        }

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 40
        dates = pd.date_range("2023-01-01", periods=n)
        X = pd.DataFrame(np.random.randn(n, 2), index=dates, columns=["f1", "f2"])
        y = pd.Series(np.random.randint(0, 2, n), index=dates)
        return X, y

    def test_fit_success(self, config, sample_data):
        """学習の成功フローテスト（ベースモデルをモック化）"""
        X, y = sample_data
        ensemble = StackingEnsemble(config)

        # ベースモデルの生成をモック化
        with patch.object(ensemble, "_create_base_model") as mock_create:
            mock_m = MagicMock()
            mock_m.predict_proba.return_value = np.zeros((len(X) // 2, 2))
            mock_create.return_value = mock_m

            with patch("app.services.ml.ensemble.stacking.cross_val_predict") as mock_cvp:
                # OOF予測として 0.5 の配列を返す
                mock_cvp.return_value = np.full((len(X), 2), 0.5)

                ensemble.fit(X, y, X_test=X, y_test=y)

                assert ensemble.is_fitted is True
                assert ensemble._fitted_meta_model is not None
                assert ensemble.oof_predictions is not None

    def test_predict_proba(self, config, sample_data):
        """予測確率の取得テスト"""
        X, y = sample_data
        ensemble = StackingEnsemble(config)

        # 最小限の状態をセットアップ
        ensemble.is_fitted = True
        mock_base = MagicMock()
        mock_base.predict_proba.return_value = np.full((5, 2), 0.7)
        ensemble._fitted_base_models = {"m1": mock_base}

        mock_meta = MagicMock()
        mock_meta.predict_proba.return_value = np.array([[0.2, 0.8]])
        ensemble._fitted_meta_model = mock_meta

        probs = ensemble.predict_proba(X.iloc[:5])
        assert probs.shape == (1, 2)
        assert probs[0, 1] == 0.8

    def test_get_feature_importance(self, config):
        """重要度取得のテスト"""
        ensemble = StackingEnsemble(config)
        ensemble.is_fitted = True

        from collections import OrderedDict

        ensemble._fitted_base_models = OrderedDict(
            [("lightgbm", MagicMock()), ("xgboost", MagicMock())]
        )

        mock_meta = MagicMock(spec=["coef_"])
        mock_meta.coef_ = np.array([[0.5, 1.5]])
        ensemble._fitted_meta_model = mock_meta

        importance = ensemble.get_feature_importance()

        assert "lightgbm" in importance
        assert "xgboost" in importance
        assert importance["lightgbm"] == 0.5
        assert importance["xgboost"] == 1.5

    def test_clear_training_data(self, config):
        """メモリ解放処理のテスト"""
        ensemble = StackingEnsemble(config)
        ensemble.is_fitted = True
        ensemble.X_train_original = pd.DataFrame([1])
        ensemble.oof_predictions = np.array([0.5])

        ensemble.clear_training_data()
        assert ensemble.X_train_original is None
        assert ensemble.oof_predictions is None

    def test_create_cv_splitter(self, config, sample_data):
        """CV分割器の作成テスト"""
        X, y = sample_data
        ensemble = StackingEnsemble(config)

        # 1. KFold
        ensemble.config["cv_strategy"] = "kfold"
        with patch("app.services.ml.ensemble.stacking.KFold") as mock_cv:
            ensemble._create_cv_splitter(X)
            assert mock_cv.called

        # 2. StratifiedKFold
        ensemble.config["cv_strategy"] = "stratified_kfold"
        with patch("app.services.ml.ensemble.stacking.StratifiedKFold") as mock_cv:
            ensemble._create_cv_splitter(X)
            assert mock_cv.called

        # 3. PurgedKFold
        ensemble.config["cv_strategy"] = "purged_kfold"
        with patch(
            "app.services.ml.ensemble.stacking.get_t1_series", return_value=pd.Series()
        ):
            with patch(
                "app.services.ml.ensemble.stacking.PurgedKFold"
            ) as mock_cv:
                ensemble._create_cv_splitter(X)
                assert mock_cv.called

    def test_load_models_new_format(self, config):
        """新形式モデル読み込み (完全モック)"""
        ensemble = StackingEnsemble(config)
        base_path = "/mock/model"
        
        model_data = {
            "model": {
                "fitted_base_models": {"lgb": MagicMock()},
                "fitted_meta_model": MagicMock(),
                "base_model_types": ["lgb"],
                "meta_model_type": "logistic_regression",
                "config": config,
                "feature_columns": ["f1", "f2"],
                "passthrough": False,
                "cv_folds": 2,
                "stack_method": "predict_proba",
                "is_fitted": True,
            },
            "metadata": {"passthrough": True},
            "feature_columns": ["f1", "f2"],
        }

        # 内部で使われる glob, os.path, joblib をグローバルにパッチ
        # 関数内での import joblib に対応するため
        with patch("joblib.load", return_value=model_data), \
             patch("glob.glob", return_value=["/mock/model.joblib"]), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.abspath", return_value="/mock/model.joblib"), \
             patch("os.path.getmtime", return_value=123.4), \
             patch("builtins.open", mock_open(read_data='{}')):
            
            success = ensemble.load_models(base_path)
            assert success is True
            assert ensemble.is_fitted is True
            assert ensemble.passthrough is True

    def test_train_meta_model(self, sample_data):
        """メタモデル学習の単体テスト"""
        X, y = sample_data
        config = {
            "base_models": ["lightgbm"],
            "meta_model": "logistic_regression",
            "cv_folds": 2,
            "cv_strategy": "kfold",
            "passthrough": False,
        }
        ensemble = StackingEnsemble(config)
        oof_preds = pd.DataFrame(
            {"lightgbm": np.linspace(0.1, 0.9, len(X))}, index=X.index
        )

        with patch("app.services.ml.ensemble.stacking.cross_val_predict") as mock_cvp:
            mock_cvp.return_value = np.zeros((len(X), 2))
            res = ensemble._train_meta_model(oof_preds, X, y)
            assert res.shape == (len(X), 2)

    def test_evaluate_ensemble(self, config, sample_data):
        """評価指標生成のテスト"""
        X, y = sample_data
        ensemble = StackingEnsemble(config)
        ensemble.is_fitted = True

        with patch.object(ensemble, "predict_proba"), patch.object(
            ensemble, "_evaluate_predictions", return_value={"accuracy": 0.8}
        ):
            res = ensemble._evaluate_ensemble(X, y)
            assert res["accuracy"] == 0.8

    def test_cleanup(self, config):
        """クリーンアップ処理のテスト"""
        ensemble = StackingEnsemble(config)
        ensemble.is_fitted = True
        ensemble._fitted_base_models = {"m": MagicMock()}

        ensemble.cleanup()
        assert ensemble.is_fitted is False
        assert not ensemble._fitted_base_models
