from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.utils.error_handler import ModelError


class TestEnsembleTrainer:
    @pytest.fixture
    def config(self):
        return {
            "method": "stacking",
            "models": ["lightgbm", "xgboost"],
            "strict_error_mode": True,
            "stacking_params": {
                "cv_folds": 2,
                "meta_model_type": "logistic_regression",
            },
        }

    @pytest.fixture
    def trainer(self, config):
        return EnsembleTrainer(ensemble_config=config)

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=20)
        X = pd.DataFrame(np.random.randn(20, 2), index=dates, columns=["f1", "f2"])
        y = pd.Series(np.random.randint(0, 2, 20), index=dates)
        return X, y

    def test_init_modes(self):
        """初期化モードのテスト (アンサンブル vs 単一モデル)"""
        # 1. アンサンブルモード
        t1 = EnsembleTrainer({"models": ["lgb", "xgb"]})
        assert t1.is_single_model is False
        assert t1.model_type == "EnsembleModel"

        # 2. 単一モデルモード
        t2 = EnsembleTrainer({"models": ["lightgbm"]})
        assert t2.is_single_model is True
        assert t2.model_type == "lightgbm"

    def test_extract_optimized_parameters(self, trainer):
        """最適化パラメータの抽出テスト"""
        training_params = {
            "lgb_learning_rate": 0.1,
            "xgb_max_depth": 5,
            "stacking_cv": 3,
            "stacking_meta_C": 1.0,
            "other_param": "val",
        }
        params = trainer._extract_optimized_parameters(training_params)

        assert params["base_models"]["lightgbm"]["learning_rate"] == 0.1
        assert params["base_models"]["xgboost"]["max_depth"] == 5
        assert params["stacking"]["cv"] == 3
        assert params["stacking"]["meta_model_params"]["C"] == 1.0

    def test_train_model_impl_success(self, trainer, sample_data):
        """アンサンブル学習の実行テスト"""
        X, y = sample_data

        # StackingEnsemble をモック化
        with patch(
            "app.services.ml.ensemble.ensemble_trainer.StackingEnsemble"
        ) as mock_stacking:
            mock_instance = mock_stacking.return_value
            mock_instance.fit.return_value = {"status": "success", "score": 0.9}
            mock_instance.predict_proba.return_value = np.zeros((len(X), 2))
            mock_instance.get_feature_importance.return_value = {"f1": 0.5}
            mock_instance.get_oof_predictions.return_value = np.zeros(len(X))
            mock_instance.get_oof_base_model_predictions.return_value = pd.DataFrame()
            mock_instance.get_X_train_original.return_value = X
            mock_instance.get_y_train_original.return_value = y

            # メタラベリングも学習されるパスを通す
            trainer.ensemble_config["meta_labeling_params"] = {"enabled": True}

            result = trainer._train_model_impl(X, X, y, y)

            assert trainer.is_trained is True
            assert result["ensemble_method"] == "stacking"
            assert "feature_importance" in result

    def test_train_model_impl_unsupported_method(self, trainer):
        """サポートされていないアンサンブル手法のエラーテスト"""
        trainer.ensemble_method = "unsupported"
        X, y = pd.DataFrame({"f": [1]}), pd.Series([1])
        with pytest.raises(ModelError, match="サポートされていないアンサンブル手法"):
            trainer._train_model_impl(X, X, y, y)

    def test_predict_proba_and_error(self, trainer):
        """predict_proba の正常動作と未学習時のエラー"""
        # 未学習
        with pytest.raises(ModelError):
            trainer.predict_proba(pd.DataFrame())

        # 正常
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        assert trainer.predict_proba(pd.DataFrame({"f": [1]})).shape == (1, 2)

    def test_predict_without_meta_labeling(self, trainer, sample_data):
        """メタラベリングなしの予測"""
        X, _ = sample_data
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        trainer.meta_labeling_service = None

        res = trainer.predict(X.iloc[:1])
        # メタラベリングなしの場合は確率配列が返る
        assert res.shape == (1, 2)
        assert res[0, 1] == 0.7

    def test_predict_with_meta_labeling_full(self, trainer):
        """メタラベリングありの予測 (正常系)"""
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        trainer.ensemble_model.predict_base_models_proba.return_value = pd.DataFrame(
            {"l": [0.8]}
        )

        trainer.meta_labeling_service = MagicMock(is_trained=True)
        trainer.meta_labeling_service.predict.return_value = pd.Series([1])

        res = trainer.predict(pd.DataFrame({"f": [1]}))
        assert res[0] == 1

    def test_predict_meta_labeling_tolerant_mode(self, trainer):
        """メタラベリング失敗時のフォールバック (strict_error_mode = False)"""
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        # ベースモデルの確率取得で失敗させる
        trainer.ensemble_model.predict_base_models_proba.side_effect = Exception("Fail")

        trainer.meta_labeling_service = MagicMock(is_trained=True)
        trainer.strict_error_mode = False
        res = trainer.predict(pd.DataFrame({"f": [1]}))
        # フォールバックで 0 が返ることを期待
        assert (res == 0).all()
        assert len(res) == 1

    def test_get_metadata_with_ml_error(self, trainer):
        """メタデータ取得時のエラーハンドリング"""
        trainer.ensemble_model = MagicMock(best_algorithm="lgb")
        trainer.meta_labeling_service = MagicMock(is_trained=True)
        # 保存エラー
        with patch(
            "app.services.ml.models.model_manager.model_manager.save_model",
            side_effect=Exception("Err"),
        ):
            meta = trainer._get_model_specific_metadata("m")
            assert "meta_model_path" not in meta

    def test_save_model_uses_ensemble_model(self, trainer):
        """保存時に ensemble_model をそのまま渡すことを確認する"""
        trainer.is_trained = True
        trainer.ensemble_model = MagicMock()
        trainer.feature_columns = ["f1", "f2"]
        trainer._model = None

        with patch(
            "app.services.ml.trainers.base_ml_trainer.model_manager.save_model",
            return_value="/saved/path",
        ) as mock_save:
            path = trainer.save_model("ensemble_model")

        assert path == "/saved/path"
        assert mock_save.call_args.kwargs["model"] is trainer.ensemble_model

    def test_load_model_success(self, trainer):
        """モデル読み込みのテスト"""
        with patch(
            "app.services.ml.trainers.base_ml_trainer.BaseMLTrainer.load_model",
            return_value=True,
        ):
            trainer._model = MagicMock()
            trainer.current_model_metadata = {
                "ensemble_config": {"method": "stacking"},
                "ensemble_method": "stacking",
                "meta_model_path": None,
            }
            # メタモデル読み込みもパッチ
            with patch(
                "app.services.ml.models.model_manager.model_manager.load_model",
                return_value=MagicMock(),
            ):
                assert trainer.load_model("/path") is True
                assert trainer.ensemble_model is not None

    def test_cleanup_models(self, trainer):
        """クリーンアップのテスト"""
        trainer.ensemble_model = MagicMock()
        trainer._cleanup_models("low")
        assert trainer.ensemble_model is None
