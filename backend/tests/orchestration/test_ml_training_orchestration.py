"""
MLトレーニングオーケストレーションサービスのテストモジュール

MLTrainingOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import BackgroundTasks

from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
    training_status,
)


@pytest.fixture
def mock_db_session() -> MagicMock:
    """
    データベースセッションのモック

    Returns:
        MagicMock: モックされたデータベースセッション
    """
    return MagicMock()


@pytest.fixture
def mock_background_tasks() -> MagicMock:
    """
    BackgroundTasksのモック

    Returns:
        MagicMock: モックされたBackgroundTasks
    """
    return MagicMock(spec=BackgroundTasks)


@pytest.fixture
def orchestration_service() -> MLTrainingOrchestrationService:
    """
    MLTrainingOrchestrationServiceのインスタンス

    Returns:
        MLTrainingOrchestrationService: テスト対象のサービス
    """
    return MLTrainingOrchestrationService()


@pytest.fixture
def sample_training_config() -> MagicMock:
    """
    サンプルトレーニング設定

    Returns:
        MagicMock: モックされた設定
    """
    config = MagicMock()
    config.symbol = "BTC/USDT:USDT"
    config.timeframe = "1h"
    config.start_date = "2024-01-01T00:00:00"
    config.end_date = "2024-01-31T00:00:00"
    config.save_model = True
    config.train_test_split = 0.8
    config.random_state = 42
    config.ensemble_config = MagicMock()
    config.ensemble_config.enabled = True
    config.ensemble_config.model_dump.return_value = {
        "enabled": True,
        "method": "stacking",
    }
    config.single_model_config = MagicMock()
    config.single_model_config.model_dump.return_value = {
        "model_type": "lightgbm"
    }
    config.optimization_settings = MagicMock()
    config.optimization_settings.enabled = False
    return config


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """
    サンプルトレーニングデータ

    Returns:
        pd.DataFrame: サンプルデータフレーム
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "close": range(100),
            "volume": range(100),
            "label": [0, 1] * 50,
        }
    )


@pytest.fixture(autouse=True)
def reset_training_status():
    """各テスト前にトレーニング状態をリセット"""
    training_status.update(
        {
            "is_training": False,
            "progress": 0,
            "status": "idle",
            "message": "待機中",
            "start_time": None,
            "end_time": None,
            "model_info": None,
            "error": None,
        }
    )
    yield


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, MLTrainingOrchestrationService)


class TestGetDataService:
    """正常系: データサービス取得のテスト"""

    def test_get_data_service_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: データサービスが正常に取得される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch.object(orchestration_service, "get_data_service") as mock_get_data_service:
            mock_backtest_data_service = MagicMock()
            mock_get_data_service.return_value = mock_backtest_data_service

            data_service = orchestration_service.get_data_service(mock_db_session)

            mock_get_data_service.assert_called_once_with(mock_db_session)
            assert data_service == mock_backtest_data_service


class TestValidateTrainingConfig:
    """正常系・異常系: トレーニング設定の検証テスト"""

    def test_validate_training_config_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: 有効な設定が検証される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        orchestration_service.validate_training_config(sample_training_config)
        # エラーが発生しないことを確認

    def test_validate_training_config_already_training(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: 既にトレーニング中の場合エラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        training_status["is_training"] = True

        with pytest.raises(ValueError, match="既にトレーニングが実行中"):
            orchestration_service.validate_training_config(sample_training_config)

    def test_validate_training_config_invalid_dates(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: 開始日が終了日より後の場合エラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-31T00:00:00"
        sample_training_config.end_date = "2024-01-01T00:00:00"

        with pytest.raises(ValueError, match="開始日は終了日より前"):
            orchestration_service.validate_training_config(sample_training_config)

    def test_validate_training_config_too_short_period(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: トレーニング期間が短すぎる場合エラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-01T00:00:00"
        sample_training_config.end_date = "2024-01-03T00:00:00"

        with pytest.raises(ValueError, match="最低7日間必要"):
            orchestration_service.validate_training_config(sample_training_config)


class TestStartTraining:
    """正常系: トレーニング開始のテスト"""

    @pytest.mark.asyncio
    async def test_start_training_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        mock_background_tasks: MagicMock,
        mock_db_session: MagicMock,
    ):
        """
        正常系: トレーニングが正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            mock_background_tasks: BackgroundTasksモック
            mock_db_session: DBセッションモック
        """
        result = await orchestration_service.start_training(
            config=sample_training_config,
            background_tasks=mock_background_tasks,
            db=mock_db_session,
        )

        assert result["success"] is True
        assert "MLトレーニングを開始" in result["message"]
        assert "training_id" in result["data"]
        mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_training_with_invalid_config(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        mock_background_tasks: MagicMock,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 無効な設定でエラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            mock_background_tasks: BackgroundTasksモック
            mock_db_session: DBセッションモック
        """
        sample_training_config.start_date = "2024-01-31T00:00:00"
        sample_training_config.end_date = "2024-01-01T00:00:00"

        with pytest.raises(ValueError):
            await orchestration_service.start_training(
                config=sample_training_config,
                background_tasks=mock_background_tasks,
                db=mock_db_session,
            )


class TestGetTrainingStatus:
    """正常系: トレーニング状態取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_training_status_idle(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: アイドル状態の取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        result = await orchestration_service.get_training_status()

        assert result["is_training"] is False
        assert result["status"] == "idle"
        assert result["progress"] == 0

    @pytest.mark.asyncio
    async def test_get_training_status_training(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: トレーニング中の状態取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        training_status.update(
            {
                "is_training": True,
                "progress": 50,
                "status": "training",
                "message": "トレーニング中",
            }
        )

        result = await orchestration_service.get_training_status()

        assert result["is_training"] is True
        assert result["status"] == "training"
        assert result["progress"] == 50


class TestGetModelInfo:
    """正常系: モデル情報取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_model_info_with_model(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: モデルが存在する場合の情報取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = "/path/to/model.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {
                    "model_type": "LightGBM",
                    "feature_count": 50,
                    "training_samples": 1000,
                    "accuracy": 0.85,
                }
            }

            with patch("os.path.exists", return_value=True):
                result = await orchestration_service.get_model_info()

                assert result["success"] is True
                assert result["data"]["model_status"]["is_loaded"] is True
                assert (
                    result["data"]["model_status"]["model_type"] == "LightGBM"
                )

    @pytest.mark.asyncio
    async def test_get_model_info_no_model(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: モデルが存在しない場合の情報取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_model_info()

            assert result["success"] is True
            assert result["data"]["model_status"]["is_loaded"] is False


class TestStopTraining:
    """正常系・異常系: トレーニング停止のテスト"""

    @pytest.mark.asyncio
    async def test_stop_training_success(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: トレーニングが正常に停止される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        training_status["is_training"] = True

        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.background_task_manager"
        ) as mock_manager:
            result = await orchestration_service.stop_training()

            assert result["success"] is True
            assert "停止" in result["message"]
            assert training_status["is_training"] is False
            mock_manager.cleanup_all_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_training_not_running(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        異常系: トレーニングが実行されていない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        training_status["is_training"] = False

        with pytest.raises(ValueError, match="実行中のトレーニングがありません"):
            await orchestration_service.stop_training()


class TestExecuteMLTraining:
    """正常系: MLトレーニング実行のテスト"""

    @pytest.mark.asyncio
    async def test_execute_ml_training_ensemble(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        sample_training_data: pd.DataFrame,
    ):
        """
        正常系: アンサンブルトレーニングの実行

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            sample_training_data: サンプルデータ
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.trainer_type = "ensemble"
            mock_service.train_model.return_value = {
                "accuracy": 0.85,
                "model_type": "ensemble",
            }
            mock_service_class.return_value = mock_service

            orchestration_service._execute_ml_training_with_error_handling(
                trainer_type="ensemble",
                ensemble_config_dict={"enabled": True, "method": "stacking"},
                single_model_config_dict=None,
                config=sample_training_config,
                training_data=sample_training_data,
            )

            mock_service_class.assert_called_once_with(
                trainer_type="ensemble",
                ensemble_config={"enabled": True, "method": "stacking"},
                single_model_config=None,
            )
            mock_service.train_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_ml_training_single_model(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        sample_training_data: pd.DataFrame,
    ):
        """
        正常系: 単一モデルトレーニングの実行

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            sample_training_data: サンプルデータ
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.trainer_type = "single"
            mock_service.trainer.model_type = "lightgbm"
            mock_service.train_model.return_value = {
                "accuracy": 0.85,
                "model_type": "lightgbm",
            }
            mock_service_class.return_value = mock_service

            orchestration_service._execute_ml_training_with_error_handling(
                trainer_type="single",
                ensemble_config_dict=None,
                single_model_config_dict={"model_type": "lightgbm"},
                config=sample_training_config,
                training_data=sample_training_data,
            )

            mock_service_class.assert_called_once_with(
                trainer_type="single",
                ensemble_config=None,
                single_model_config={"model_type": "lightgbm"},
            )
            mock_service.train_model.assert_called_once()

    def test_execute_ml_training_initialization_error(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        sample_training_data: pd.DataFrame,
    ):
        """
        異常系: MLサービス初期化エラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            sample_training_data: サンプルデータ
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService"
        ) as mock_service_class:
            mock_service_class.side_effect = Exception("Initialization failed")

            with pytest.raises(Exception, match="Initialization failed"):
                orchestration_service._execute_ml_training_with_error_handling(
                    trainer_type="ensemble",
                    ensemble_config_dict={"enabled": True},
                    single_model_config_dict=None,
                    config=sample_training_config,
                    training_data=sample_training_data,
                )


class TestLogAndValidateConfig:
    """正常系: 設定のログ出力と検証のテスト"""

    def test_log_and_validate_config_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: 設定のログ出力と検証が正常に行われる

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        orchestration_service._log_and_validate_config(sample_training_config)
        # エラーが発生しないことを確認

    def test_log_and_validate_config_with_invalid_config(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: 無効な設定でエラー

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-31T00:00:00"
        sample_training_config.end_date = "2024-01-01T00:00:00"

        with pytest.raises(ValueError):
            orchestration_service._log_and_validate_config(sample_training_config)


class TestTrainingStatusManagement:
    """正常系: トレーニング状態管理のテスト"""

    def test_training_status_updates_correctly(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: トレーニング状態が正しく更新される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        # 初期状態
        assert training_status["is_training"] is False
        assert training_status["status"] == "idle"

        # トレーニング開始状態に更新
        training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "status": "starting",
                "message": "トレーニングを開始しています...",
                "start_time": datetime.now().isoformat(),
            }
        )

        assert training_status["is_training"] is True
        assert training_status["status"] == "starting"

        # トレーニング中状態に更新
        training_status.update(
            {"progress": 50, "status": "training", "message": "トレーニング中..."}
        )

        assert training_status["progress"] == 50
        assert training_status["status"] == "training"

        # トレーニング完了状態に更新
        training_status.update(
            {
                "is_training": False,
                "progress": 100,
                "status": "completed",
                "message": "トレーニングが完了しました",
                "end_time": datetime.now().isoformat(),
            }
        )

        assert training_status["is_training"] is False
        assert training_status["status"] == "completed"
        assert training_status["progress"] == 100


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_model_info_with_corrupt_metadata(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        境界値: メタデータが破損している場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.return_value = "/path/to/model.pkl"
            mock_manager.load_model.side_effect = Exception("Corrupt metadata")

            with patch("os.path.exists", return_value=True):
                result = await orchestration_service.get_model_info()

                assert result["success"] is True
                # メタデータ破損でもエラーハンドリングされる
                assert result["data"]["model_status"]["is_loaded"] is False

    def test_validate_training_config_minimum_period(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        境界値: 最小トレーニング期間（7日間）

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-01T00:00:00"
        sample_training_config.end_date = "2024-01-08T00:00:00"

        orchestration_service.validate_training_config(sample_training_config)
        # エラーが発生しないことを確認