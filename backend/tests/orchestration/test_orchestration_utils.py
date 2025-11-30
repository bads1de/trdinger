"""
オーケストレーション共通ユーティリティのテスト
"""

from unittest.mock import Mock, patch

from app.services.ml.orchestration.orchestration_utils import (
    get_latest_model_with_info,
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
        mock_model_manager.load_model.return_value = {
            "model": Mock(),
            "metadata": expected_metadata,
        }

        # Act
        result = load_model_metadata_safely(model_path)

        # Assert
        assert result is not None
        assert result["metadata"] == expected_metadata
        mock_model_manager.load_model.assert_called_once_with(model_path)

    @patch("app.services.ml.orchestration.orchestration_utils.model_manager")
    def test_load_model_metadata_no_metadata_key(self, mock_model_manager):
        """メタデータキーが存在しない場合"""
        # Arrange
        model_path = "/path/to/model.pkl"
        mock_model_manager.load_model.return_value = {
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
        mock_model_manager.load_model.return_value = None

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
        mock_model_manager.load_model.side_effect = Exception("Load error")

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
        mock_model_manager.load_model.return_value = {
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
        mock_model_manager.load_model.return_value = {"model": Mock()}

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
        mock_model_manager.load_model.side_effect = Exception("Unexpected error")

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
        mock_model_manager.load_model.return_value = {
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
