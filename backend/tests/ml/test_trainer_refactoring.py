import pytest
from unittest.mock import MagicMock, patch
from backend.app.services.ml.ensemble.ensemble_trainer import EnsembleTrainer


class TestTrainerRefactoring:
    """トレーナーリファクタリングのテストクラス"""

    @pytest.fixture
    def mock_model_manager(self):
        # BaseMLTrainerで使用される箇所をパッチ
        with patch("backend.app.services.ml.base_ml_trainer.model_manager") as mock:
            yield mock

    @pytest.fixture
    def mock_single_model_save(self, mock_model_manager):
        return mock_model_manager

    @pytest.fixture
    def mock_ensemble_model_save(self, mock_model_manager):
        return mock_model_manager

    def test_single_model_save_model_duplication(self, mock_single_model_save):
        """リファクタリング後のEnsembleTrainer（単一モデルモード）のsave_modelテスト"""
        # 単一モデルモードでEnsembleTrainerを使用
        config = {"method": "stacking", "models": ["lightgbm"]}
        trainer = EnsembleTrainer(ensemble_config=config)
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.model = MagicMock()
        trainer._model = trainer.ensemble_model

        # feature_importance属性が存在しないようにする
        if hasattr(trainer._model, "feature_importance"):
            del trainer._model.feature_importance

        # get_feature_importanceをモック
        trainer._model.get_feature_importance.return_value = {
            "f1": 0.5,
            "f2": 0.5,
        }

        mock_single_model_save.save_model.return_value = "path/to/model"

        path = trainer.save_model("test_model")

        assert path == "path/to/model"
        args, kwargs = mock_single_model_save.save_model.call_args
        metadata = kwargs["metadata"]
        # EnsembleTrainer（単一モデルモード）はmodel_typeを設定する
        assert "model_type" in metadata
        assert metadata["is_trained"] is True
        assert "feature_count" in metadata

    def test_ensemble_model_save_model_duplication(self, mock_ensemble_model_save):
        """リファクタリング後のEnsembleTrainer.save_modelテスト"""
        config = {"method": "stacking", "models": ["lightgbm", "xgboost"]}
        trainer = EnsembleTrainer(ensemble_config=config)
        trainer.is_trained = True
        trainer.feature_columns = ["f1", "f2"]
        trainer.ensemble_model = MagicMock()
        trainer.ensemble_model.is_fitted = True
        trainer.ensemble_model.best_algorithm = "lgbm"
        # _modelを設定
        trainer._model = trainer.ensemble_model

        # feature_importance属性が存在しないようにして、get_feature_importanceメソッドにフォールバックさせる
        if hasattr(trainer.ensemble_model, "feature_importance"):
            del trainer.ensemble_model.feature_importance

        # get_feature_importanceをモック
        trainer.ensemble_model.get_feature_importance.return_value = {
            "f1": 0.6,
            "f2": 0.4,
        }

        mock_ensemble_model_save.save_model.return_value = "path/to/ensemble_pickle"

        path = trainer.save_model("test_ensemble")

        assert path == "path/to/ensemble_pickle"

        # save_modelが正しいメタデータで呼び出されたか検証
        args, kwargs = mock_ensemble_model_save.save_model.call_args
        metadata = kwargs["metadata"]
        # BaseMLTrainerはmodel_typeを__class__.__name__に設定するが、EnsembleTrainerはメタデータ経由でオーバーライドする
        assert metadata["model_type"] == "lgbm"
        assert metadata["is_trained"] is True
        assert "feature_count" in metadata
        assert "feature_importance" in metadata

        # 渡されたモデルオブジェクトがトレーナー自身であることを検証（BaseMLTrainerは'self'を保存）
        assert kwargs["model"] == trainer.ensemble_model

    def test_get_feature_importance_duplication(self):
        """get_feature_importanceの重複テスト"""
        # 単一モデルモード（EnsembleTrainerを使用）
        config_single = {"method": "stacking", "models": ["lightgbm"]}
        trainer = EnsembleTrainer(ensemble_config=config_single)
        trainer.is_trained = True
        trainer.ensemble_model = MagicMock()
        trainer._model = trainer.ensemble_model  # _modelを設定

        # feature_importance属性が存在しないようにする
        if hasattr(trainer._model, "feature_importance"):
            del trainer._model.feature_importance

        # BaseMLTrainer.get_feature_importanceのロジック用モック
        # hasattr(self._model, "feature_importance") または "get_feature_importance" をチェックする
        trainer._model.get_feature_importance.return_value = {
            "f1": 0.8,
            "f2": 0.2,
        }
        trainer.feature_columns = ["f1", "f2"]

        fi = trainer.get_feature_importance()
        assert fi == {"f1": 0.8, "f2": 0.2}

        # アンサンブルモデル
        config_ensemble = {"method": "stacking", "models": ["lightgbm", "xgboost"]}
        trainer_ens = EnsembleTrainer(ensemble_config=config_ensemble)
        trainer_ens.is_trained = True
        trainer_ens.ensemble_model = MagicMock()
        trainer_ens._model = trainer_ens.ensemble_model  # _modelを設定

        # feature_importance属性が存在しないようにする
        if hasattr(trainer_ens.ensemble_model, "feature_importance"):
            del trainer_ens.ensemble_model.feature_importance

        trainer_ens.ensemble_model.get_feature_importance.return_value = {
            "f1": 0.7,
            "f2": 0.3,
        }
        trainer_ens.feature_columns = ["f1", "f2"]

        fi_ens = trainer_ens.get_feature_importance()
        assert fi_ens == {"f1": 0.7, "f2": 0.3}

