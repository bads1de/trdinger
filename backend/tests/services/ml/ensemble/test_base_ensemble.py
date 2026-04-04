import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.services.ml.common.exceptions import MLModelError
from app.services.ml.ensemble.base_ensemble import BaseEnsemble


# テスト用の具象クラス
class MockEnsemble(BaseEnsemble):
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.is_fitted = True
        self.feature_columns = X_train.columns.tolist()
        return {"status": "success"}

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


class TestBaseEnsemble:
    """BaseEnsemble のテスト (ファイル操作をモック化)"""

    @pytest.fixture
    def ensemble(self):
        return MockEnsemble(config={"method": "test"})

    def test_create_base_model(self, ensemble):
        """ベースモデルの動的作成"""
        models = ["lightgbm", "xgboost", "catboost", "logistic_regression"]
        for m in models:
            obj = ensemble._create_base_model(m)
            assert obj is not None

        with pytest.raises(MLModelError, match="Unsupported model type"):
            ensemble._create_base_model("invalid")

    def test_evaluate_predictions(self, ensemble):
        """予測評価の委譲"""
        y_true = pd.Series([0, 1])
        y_pred = np.array([0, 1])

        with patch(
            "app.services.ml.ensemble.base_ensemble.evaluate_model_predictions",
            return_value={"acc": 1.0},
        ) as mock_eval:
            res = ensemble._evaluate_predictions(y_true, y_pred)
            assert res["acc"] == 1.0
            mock_eval.assert_called_once()

    def test_get_feature_importance_aggregation(self, ensemble):
        """複数ベースモデルからの重要度集約テスト"""
        ensemble.is_fitted = True
        ensemble.feature_columns = ["f1", "f2"]

        m1 = MagicMock()
        m2 = MagicMock()
        ensemble.base_models = [m1, m2]

        # 共通ユーティリティの戻り値をモック
        with patch(
            "app.services.ml.ensemble.base_ensemble.get_feature_importance_unified",
            side_effect=[{"f1": 10.0, "f2": 20.0}, {"f1": 30.0, "f2": 40.0}],
        ):
            importance = ensemble.get_feature_importance()
            # 平均: f1=(10+30)/2 = 20, f2=(20+40)/2 = 30
            assert importance["f1"] == 20.0
            assert importance["f2"] == 30.0

    def test_get_feature_importance_uses_fitted_models(self, ensemble):
        """学習済みベースモデルから重要度を集約するテスト"""
        ensemble.is_fitted = True
        ensemble.feature_columns = ["f1", "f2"]

        m1 = MagicMock()
        m2 = MagicMock()
        ensemble._fitted_base_models = {"m1": m1, "m2": m2}

        with patch(
            "app.services.ml.ensemble.base_ensemble.get_feature_importance_unified",
            side_effect=[{"f1": 10.0, "f2": 20.0}, {"f1": 30.0, "f2": 40.0}],
        ):
            importance = ensemble.get_feature_importance()
            assert importance["f1"] == 20.0
            assert importance["f2"] == 30.0

    def test_save_models_legacy_fallback(self, ensemble):
        """従来形式（レガシー）での保存フォールバック (mock path)"""
        # 実ファイルパスを使わず、文字列のみを指定
        base_path = "/mock/path/model"
        with patch("joblib.dump") as mock_dump:
            paths = ensemble.save_models(base_path)
            assert len(paths) == 1
            assert "legacy" in paths[0]
            assert mock_dump.called

    def test_load_models_failure(self, ensemble):
        """ファイルが見つからない場合の読み込み失敗"""
        # 関数内 import のため、グローバルな glob.glob をパッチ
        with patch("glob.glob", return_value=[]):
            assert ensemble.load_models("/non/existent/path") is False
            
    def test_load_models_success_legacy(self, ensemble):
        """旧形式モデルの読み込み成功テスト"""
        with patch("glob.glob") as mock_glob, \
             patch("joblib.load") as mock_load:
            mock_glob.return_value = ["/mock/model_stacking_classifier_v1.pkl"]
            mock_load.return_value = {
                "ensemble_classifier": MagicMock(),
                "config": {"test": True},
                "feature_columns": ["f1"],
                "is_fitted": True
            }
            
            success = ensemble.load_models("/mock/model")
            assert success is True
            assert ensemble.is_fitted is True
            assert ensemble.config["test"] is True
