"""
オーケストレーション共通ユーティリティのテスト
"""

from unittest.mock import Mock, patch

from app.services.ml.orchestration.orchestration_utils import (
    get_latest_model_with_info,
    get_model_info_with_defaults,
    load_model_metadata_safely,
)


class TestLoadModelMetadataSafely:
    """load_model_metadata_safely関数のテスト"""

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_load_model_metadata_successfully(self, mock_model_manager):
        """正常にモデルメタデータを読み込むケース"""
        # Arrange
        model_path = "/path/to/model.pkl"
        expected_metadata = {
            "model_type": "LightGBM",
            "feature_count": 10,
            "accuracy": 0.85,
        }
        mock_model_manager.load_metadata_only.return_value = {
            "model": Mock(),
            "metadata": expected_metadata,
        }

        # Act
        result = load_model_metadata_safely(model_path)

        # Assert
        assert result is not None
        assert result["metadata"] == expected_metadata
        mock_model_manager.load_metadata_only.assert_called_once_with(model_path)

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_load_model_metadata_no_metadata_key(self, mock_model_manager):
        """メタデータキーが存在しない場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.load_metadata_only.return_value = {
            "model": Mock(),
        }

        # Act
        result = load_model_metadata_safely(model_path)

        # Assert
        assert result is None

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_load_model_metadata_load_returns_none(self, mock_model_manager):
        """load_modelがNoneを返す場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.load_metadata_only.return_value = None

        # Act
        result = load_model_metadata_safely(model_path)

        # Assert
        assert result is None

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    @patch("app.services.ml.orchestration.orchestration_utils.logger")
    def test_load_model_metadata_exception(self, mock_logger, mock_model_manager):
        """例外が発生した場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.load_metadata_only.side_effect = Exception("Load error")

        # Act
        result = load_model_metadata_safely(model_path)

        # Assert
        assert result is None
        mock_logger.warning.assert_called_once()


class TestGetLatestModelWithInfo:
    """get_latest_model_with_info関数のテスト"""

    @patch("app.services.ml.orchestration.orchestration_utils.os.path.exists")
    @patch("app.services.ml.orchestration.orchestration_utils.os.path.getmtime")
    @patch("app.services.ml.orchestration.orchestration_utils.os.path.getsize")
    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_get_latest_model_with_full_info(
        self, mock_model_manager, mock_getsize, mock_getmtime, mock_exists
    ):
        """完全な情報を持つ最新モデルを取得する"""
        # Arrange
        model_path = "/path/to/latest_model.pkl"
        mock_model_manager.get_latest_model.return_value = model_path
        mock_exists.return_value = True
        mock_getmtime.return_value = 1700000000.0
        mock_getsize.return_value = 1024 * 1024 * 5  # 5MB

        metadata = {
            "model_type": "LightGBM",
            "feature_count": 20,
            "training_samples": 1000,
            "accuracy": 0.90,
        }
        mock_model_manager.load_metadata_only.return_value = {
            "model": Mock(),
            "metadata": metadata,
        }
        mock_model_manager.extract_model_performance_metrics.return_value = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.86,
        }

        # Act
        result = get_latest_model_with_info()

        # Assert
        assert result is not None
        assert result["path"] == model_path
        assert result["metadata"] == metadata
        assert "metrics" in result
        assert result["metrics"]["accuracy"] == 0.90
        assert "file_info" in result
        assert result["file_info"]["size_mb"] == 5.0

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_get_latest_model_not_found(self, mock_model_manager):
        """最新モデルが見つからない場合"""
        # Arrange
        mock_model_manager.get_latest_model.return_value = None

        # Act
        result = get_latest_model_with_info()

        # Assert
        assert result is None

    @patch("app.services.ml.orchestration.orchestration_utils.os.path.exists")
    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_get_latest_model_file_not_exists(self, mock_model_manager, mock_exists):
        """モデルファイルが存在しない場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.get_latest_model.return_value = model_path
        mock_exists.return_value = False

        # Act
        result = get_latest_model_with_info()

        # Assert
        assert result is None

    @patch("app.services.ml.orchestration.orchestration_utils.os.path.exists")
    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_get_latest_model_no_metadata(self, mock_model_manager, mock_exists):
        """メタデータが存在しない場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.get_latest_model.return_value = model_path
        mock_exists.return_value = True
        mock_model_manager.load_metadata_only.return_value = {"model": Mock()}

        # Act
        result = get_latest_model_with_info()

        # Assert
        assert result is None

    @patch("app.services.ml.orchestration.orchestration_utils.os.path.exists")
    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    @patch("app.services.ml.orchestration.orchestration_utils.logger")
    def test_get_latest_model_with_exception(
        self, mock_logger, mock_model_manager, mock_exists
    ):
        """例外が発生した場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.get_latest_model.return_value = model_path
        mock_exists.return_value = True
        mock_model_manager.load_metadata_only.side_effect = Exception(
            "Unexpected error"
        )

        # Act
        result = get_latest_model_with_info()

        # Assert
        assert result is None
        mock_logger.warning.assert_called_once()

    @patch("app.services.ml.orchestration.orchestration_utils.os.path.exists")
    @patch("app.services.ml.orchestration.orchestration_utils.os.path.getmtime")
    @patch("app.services.ml.orchestration.orchestration_utils.os.path.getsize")
    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_get_latest_model_with_pattern(
        self, mock_model_manager, mock_getsize, mock_getmtime, mock_exists
    ):
        """パターンを指定して最新モデルを取得"""
        # Arrange
        pattern = "ensemble_*"
        model_path = "/path/to/ensemble_model.pkl"
        mock_model_manager.get_latest_model.return_value = model_path
        mock_exists.return_value = True
        mock_getmtime.return_value = 1700000000.0
        mock_getsize.return_value = 1024 * 1024 * 3  # 3MB

        metadata = {"model_type": "Ensemble"}
        mock_model_manager.load_metadata_only.return_value = {
            "model": Mock(),
            "metadata": metadata,
        }
        mock_model_manager.extract_model_performance_metrics.return_value = {
            "accuracy": 0.92,
        }

        # Act
        result = get_latest_model_with_info(model_name_pattern=pattern)

        # Assert
        assert result is not None
        assert result["path"] == model_path
        mock_model_manager.get_latest_model.assert_called_once_with(pattern)


class TestGetModelInfoWithDefaults:
    """get_model_info_with_defaults関数のテスト"""

    def test_with_valid_model_info(self):
        """有効なモデル情報がある場合、適切に変換される"""
        # Arrange
        from datetime import datetime

        model_info = {
            "path": "/path/to/model.pkl",
            "metadata": {
                "model_type": "LightGBM",
                "feature_count": 20,
                "training_samples": 1000,
                "test_samples": 200,
                "num_classes": 2,
                "best_iteration": 100,
                "train_test_split": 0.8,
                "random_state": 42,
                "feature_importance": {"feature1": 0.5, "feature2": 0.3},
                "classification_report": {"precision": 0.9},
            },
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.88,
                "recall": 0.85,
                "f1_score": 0.86,
            },
            "file_info": {
                "size_mb": 5.0,
                "modified_at": datetime(2024, 1, 1, 12, 0, 0),
            },
        }

        # Act
        result = get_model_info_with_defaults(model_info)

        # Assert
        assert result["model_type"] == "LightGBM"
        assert result["feature_count"] == 20
        assert result["training_samples"] == 1000
        assert result["test_samples"] == 200
        assert result["accuracy"] == 0.90
        assert result["precision"] == 0.88
        assert result["recall"] == 0.85
        assert result["f1_score"] == 0.86
        assert result["file_size_mb"] == 5.0
        assert result["last_updated"] == "2024-01-01T12:00:00"
        assert result["num_classes"] == 2
        assert result["best_iteration"] == 100
        assert result["train_test_split"] == 0.8
        assert result["random_state"] == 42
        assert result["feature_importance"] == {"feature1": 0.5, "feature2": 0.3}
        assert result["classification_report"] == {"precision": 0.9}

    @patch("app.services.ml.orchestration.orchestration_utils.get_default_metrics")
    def test_with_none_model_info(self, mock_get_default_metrics):
        """モデル情報がNoneの場合、デフォルト値が返される"""
        # Arrange
        mock_get_default_metrics.return_value = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        # Act
        result = get_model_info_with_defaults(None)

        # Assert
        assert result["model_type"] == "No Model"
        assert result["feature_count"] == 0
        assert result["training_samples"] == 0
        assert result["last_updated"] == "未学習"
        assert result["file_size_mb"] == 0.0
        assert result["accuracy"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1_score"] == 0.0

    def test_with_partial_metadata(self):
        """一部のメタデータが欠けている場合、デフォルト値が使用される"""
        # Arrange
        from datetime import datetime

        model_info = {
            "path": "/path/to/model.pkl",
            "metadata": {
                "model_type": "XGBoost",
                # feature_count, training_samplesなど一部が欠けている
            },
            "metrics": {
                "accuracy": 0.85,
            },
            "file_info": {
                "size_mb": 3.0,
                "modified_at": datetime(2024, 2, 1, 10, 30, 0),
            },
        }

        # Act
        result = get_model_info_with_defaults(model_info)

        # Assert
        assert result["model_type"] == "XGBoost"
        assert result["feature_count"] == 0  # デフォルト値
        assert result["training_samples"] == 0  # デフォルト値
        assert result["test_samples"] == 0  # デフォルト値
        assert result["accuracy"] == 0.85
        assert result["file_size_mb"] == 3.0
        assert result["num_classes"] == 2  # デフォルト値
        assert result["best_iteration"] == 0  # デフォルト値


