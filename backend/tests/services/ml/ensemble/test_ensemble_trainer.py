import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.utils.error_handler import ModelError

class TestEnsembleTrainer:
    @pytest.fixture
    def config(self):
        return {
            "method": "stacking",
            "models": ["lightgbm", "xgboost"],
            "stacking_params": {"cv_folds": 2}
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

    def test_init_single_model(self):
        """単一モデルモードでの初期化"""
        config = {"models": ["lightgbm"]}
        trainer = EnsembleTrainer(config)
        assert trainer.is_single_model is True
        assert trainer.model_type == "lightgbm"

    def test_init_ensemble_model(self, config):
        """アンサンブルモードでの初期化"""
        trainer = EnsembleTrainer(config)
        assert trainer.is_single_model is False
        assert trainer.model_type == "EnsembleModel"

    def test_extract_optimized_parameters(self, trainer):
        """最適化パラメータの分離テスト"""
        training_params = {
            "lgb_learning_rate": 0.1,
            "xgb_max_depth": 5,
            "stacking_cv_folds": 3,
            "stacking_meta_model_type": "lightgbm",
            "other_param": "val"
        }
        params = trainer._extract_optimized_parameters(training_params)
        
        assert params["base_models"]["lightgbm"]["learning_rate"] == 0.1
        assert params["base_models"]["xgboost"]["max_depth"] == 5
        assert params["stacking"]["cv_folds"] == 3
        assert params["stacking"]["meta_model_params"]["model_type"] == "lightgbm"

    def test_train_model_impl_success(self, trainer, sample_data):
        """アンサンブル学習の実行テスト"""
        X, y = sample_data
        
        # StackingEnsemble をモック化
        with patch('app.services.ml.ensemble.ensemble_trainer.StackingEnsemble') as mock_stacking:
            mock_instance = mock_stacking.return_value
            mock_instance.fit.return_value = {"status": "success"}
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

    def test_predict_without_meta_labeling(self, trainer, sample_data):
        """メタラベリングなしの予測"""
        X, y = sample_data
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        res = trainer.predict(X.iloc[:1])
        # メタラベリングなしの場合は確率配列が返る
        assert res.shape == (1, 2)
        assert res[0, 1] == 0.7

    def test_predict_with_meta_labeling(self, trainer, sample_data):
        """メタラベリングありの予測"""
        X, y = sample_data
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        trainer.ensemble_model.predict_base_models_proba.return_value = pd.DataFrame({"m1": [0.7]})
        
        trainer.meta_labeling_service = MagicMock()
        trainer.meta_labeling_service.is_trained = True
        # メタモデルは 0 または 1 を返す
        trainer.meta_labeling_service.predict.return_value = pd.Series([1])
        
        res = trainer.predict(X.iloc[:1])
        # メタラベリングありの場合はクラス（0/1）が返る
        assert len(res) == 1
        assert res[0] == 1

    def test_predict_not_fitted_error(self, trainer, sample_data):
        """未学習時の予測エラー"""
        X, y = sample_data
        with pytest.raises(ModelError, match="学習済みアンサンブルモデルがありません"):
            trainer.predict(X)

    def test_load_model_success(self, trainer):
        """モデル読み込みのテスト"""
        with patch('app.services.ml.trainers.base_ml_trainer.BaseMLTrainer.load_model', return_value=True):
            trainer._model = MagicMock()
            trainer.metadata = {
                "ensemble_config": {"method": "stacking"},
                "meta_model_path": None
            }
            assert trainer.load_model("/path") is True
            assert trainer.ensemble_model is not None
