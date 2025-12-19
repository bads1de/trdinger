import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer
from app.utils.error_handler import ModelError

class TestEnsembleTrainerUnit:
    @pytest.fixture
    def ensemble_config(self):
        return {
            "models": ["lightgbm", "xgboost"],
            "method": "stacking",
            "strict_error_mode": True,
            "stacking_params": {"meta_model_type": "logistic_regression"}
        }

    @pytest.fixture
    def trainer(self, ensemble_config):
        return EnsembleTrainer(ensemble_config)

    def test_init_modes(self):
        # 1. アンサンブルモード
        t1 = EnsembleTrainer({"models": ["lgb", "xgb"]})
        assert t1.is_single_model is False
        
        # 2. 単一モデルモード
        t2 = EnsembleTrainer({"models": ["lightgbm"]})
        assert t2.is_single_model is True

    def test_extract_optimized_parameters(self, trainer):
        raw_params = {
            "lgb_learning_rate": 0.1,
            "xgb_max_depth": 5,
            "stacking_cv": 3,
            "stacking_meta_C": 1.0
        }
        extracted = trainer._extract_optimized_parameters(raw_params)
        assert extracted["base_models"]["lightgbm"]["learning_rate"] == 0.1
        assert extracted["stacking"]["meta_model_params"]["C"] == 1.0

    def test_predict_proba_and_error(self, trainer):
        # 未学習
        with pytest.raises(ModelError):
            trainer.predict_proba(pd.DataFrame())
            
        # 正常
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        assert trainer.predict_proba(pd.DataFrame({"f": [1]})).shape == (1, 2)

    def test_predict_with_meta_labeling_full(self, trainer):
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        trainer.ensemble_model.predict_base_models_proba.return_value = pd.DataFrame({"l": [0.8]})
        
        trainer.meta_labeling_service = MagicMock(is_trained=True)
        trainer.meta_labeling_service.predict.return_value = pd.Series([1])
        
        res = trainer.predict(pd.DataFrame({"f": [1]}))
        assert res[0] == 1

    def test_predict_no_meta_labeling(self, trainer):
        trainer.ensemble_model = MagicMock(is_fitted=True)
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        trainer.meta_labeling_service = None
        res = trainer.predict(pd.DataFrame({"f": [1]}))
        assert res[0][1] == 0.7

    def test_predict_meta_labeling_tolerant_mode(self, trainer):
        # strict_error_mode = False 時のフォールバック (0を返す)
        trainer.ensemble_model = MagicMock(is_fitted=True)
        # predict_proba が正しい形状の配列を返すようにする
        trainer.ensemble_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        # ベースモデルの確率取得で失敗させる
        trainer.ensemble_model.predict_base_models_proba.side_effect = Exception("Fail")
        
        trainer.meta_labeling_service = MagicMock(is_trained=True)
        trainer.strict_error_mode = False
        res = trainer.predict(pd.DataFrame({"f": [1]}))
        assert (res == 0).all()
        assert len(res) == 1


    def test_train_model_impl_success_and_skip(self, trainer):
        X, y = pd.DataFrame({"f": [1]*10}), pd.Series([1]*10)
        
        # 1. StackingEnsemble モック
        with patch("app.services.ml.ensemble.ensemble_trainer.StackingEnsemble") as mock_stack_cls:
            mock_stack = mock_stack_cls.return_value
            mock_stack.fit.return_value = {"score": 0.9}
            mock_stack.predict_proba.return_value = np.zeros((10, 2))
            mock_stack.get_feature_importance.return_value = {"f": 1.0}
            # メタラベリングデータをワザと欠落させる
            mock_stack.get_oof_predictions.return_value = None
            
            # 実行 (同期メソッド)
            result = trainer._train_model_impl(X, X, y, y)
            assert result["score"] == 0.9
            assert trainer.meta_labeling_service is None # データ欠落でスキップされた

    def test_train_model_impl_unsupported_method(self, trainer):
        trainer.ensemble_method = "unsupported"
        X, y = pd.DataFrame({"f": [1]}), pd.Series([1])
        with pytest.raises(ModelError, match="サポートされていないアンサンブル手法"):
            trainer._train_model_impl(X, X, y, y)

    def test_get_metadata_with_ml_error(self, trainer):
        trainer.ensemble_model = MagicMock(best_algorithm="lgb")
        trainer.meta_labeling_service = MagicMock(is_trained=True)
        # 保存エラー
        with patch("app.services.ml.models.model_manager.model_manager.save_model", side_effect=Exception("Err")):
            meta = trainer._get_model_specific_metadata("m")
            assert "meta_model_path" not in meta

    def test_load_model_meta_fail(self, trainer):
        # 親クラスの load_model をパッチ
        with patch("app.services.ml.trainers.base_ml_trainer.BaseMLTrainer.load_model", return_value=True):
            trainer.metadata = {"meta_model_path": "/path"}
            # メタモデルのロード失敗
            with patch("app.services.ml.models.model_manager.model_manager.load_model", return_value=None):
                trainer.meta_labeling_service = None # クリーンな状態
                assert trainer.load_model("dummy") is True
                assert trainer.meta_labeling_service is None

    def test_cleanup_models(self, trainer):
        trainer.ensemble_model = MagicMock()
        trainer._cleanup_models("low")
        assert trainer.ensemble_model is None