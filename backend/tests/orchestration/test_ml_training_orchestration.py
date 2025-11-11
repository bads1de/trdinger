"""
MLトレーニングオーケストレーションサービスのテストモジュール

MLTrainingOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ml.orchestration.ml_training_orchestration_service import (
    MLTrainingOrchestrationService,
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
        MagicMock: モックされたバックグラウンドタスク
    """
    mock = MagicMock()
    mock.add_task = MagicMock()
    return mock


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
        MagicMock: トレーニング設定のモック
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
    config.ensemble_config.model_dump.return_value = {"enabled": True}
    config.single_model_config = None
    config.optimization_settings = None
    return config


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


class TestValidateTrainingConfig:
    """正常系: トレーニング設定検証のテスト"""

    def test_validate_training_config_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: 有効な設定がバリデーションを通過する

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": False},
        ):
            # 例外が発生しないことを確認
            orchestration_service.validate_training_config(sample_training_config)

    def test_validate_training_config_already_training(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: 既にトレーニング中の場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": True},
        ):
            with pytest.raises(ValueError, match="既にトレーニングが実行中"):
                orchestration_service.validate_training_config(sample_training_config)

    def test_validate_training_config_invalid_dates(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: 開始日が終了日より後の場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-02-01T00:00:00"
        sample_training_config.end_date = "2024-01-01T00:00:00"

        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": False},
        ):
            with pytest.raises(ValueError, match="開始日は終了日より前"):
                orchestration_service.validate_training_config(sample_training_config)

    def test_validate_training_config_short_period(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        異常系: トレーニング期間が短すぎる場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-01T00:00:00"
        sample_training_config.end_date = "2024-01-03T00:00:00"

        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": False},
        ):
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
            mock_background_tasks: バックグラウンドタスクモック
            mock_db_session: DBセッションモック
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"is_training": False},
            ),
            patch.object(
                orchestration_service, "_log_and_validate_config"
            ) as mock_validate,
        ):
            result = await orchestration_service.start_training(
                config=sample_training_config,
                background_tasks=mock_background_tasks,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert "開始" in result["message"]
            mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_training_validation_error(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        mock_background_tasks: MagicMock,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 設定検証エラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            mock_background_tasks: バックグラウンドタスクモック
            mock_db_session: DBセッションモック
        """
        with patch.object(
            orchestration_service, "_log_and_validate_config"
        ) as mock_validate:
            mock_validate.side_effect = ValueError("Invalid config")

            with pytest.raises(ValueError, match="Invalid config"):
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
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {
                "is_training": False,
                "status": "idle",
                "progress": 0,
                "message": "待機中",
            },
        ):
            result = await orchestration_service.get_training_status()

            assert result["is_training"] is False
            assert result["status"] == "idle"

    @pytest.mark.asyncio
    async def test_get_training_status_training(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: トレーニング中の状態取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {
                "is_training": True,
                "status": "training",
                "progress": 50,
                "message": "トレーニング中...",
            },
        ):
            result = await orchestration_service.get_training_status()

            assert result["is_training"] is True
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
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"model_info": None},
            ),
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_manager.load_model.return_value = {
                "metadata": {
                    "model_type": "LightGBM",
                    "feature_count": 50,
                    "accuracy": 0.85,
                }
            }

            result = await orchestration_service.get_model_info()

            assert result["success"] is True
            assert result["data"]["model_status"]["is_loaded"] is True

    @pytest.mark.asyncio
    async def test_get_model_info_no_model(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        エッジケース: モデルが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"model_info": None},
            ),
        ):
            mock_manager.get_latest_model.return_value = None

            result = await orchestration_service.get_model_info()

            assert result["success"] is True
            assert result["data"]["model_status"]["is_loaded"] is False


class TestStopTraining:
    """正常系: トレーニング停止のテスト"""

    @pytest.mark.asyncio
    async def test_stop_training_success(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        正常系: トレーニングが正常に停止される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"is_training": True},
            ) as mock_status,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.background_task_manager"
            ) as mock_task_manager,
        ):
            result = await orchestration_service.stop_training()

            assert result["success"] is True
            mock_task_manager.cleanup_all_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_training_not_running(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        異常系: トレーニングが実行中でない場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": False},
        ):
            with pytest.raises(ValueError, match="実行中のトレーニングがありません"):
                await orchestration_service.stop_training()


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
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.OHLCVRepository"
            ),
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.OpenInterestRepository"
            ),
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.FundingRateRepository"
            ),
            patch(
                "app.services.backtest.backtest_data_service.BacktestDataService"
            ) as mock_data_service_class,
        ):
            result = orchestration_service.get_data_service(db=mock_db_session)

            assert result is not None
            mock_data_service_class.assert_called_once()


class TestLogAndValidateConfig:
    """正常系: 設定ログ出力と検証のテスト"""

    def test_log_and_validate_config_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: 設定のログ出力と検証が正常に実行される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"is_training": False},
            ),
            patch.object(
                orchestration_service, "validate_training_config"
            ) as mock_validate,
        ):
            orchestration_service._log_and_validate_config(sample_training_config)

            mock_validate.assert_called_once_with(sample_training_config)

    def test_log_and_validate_config_with_single_model(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: 単一モデル設定でログ出力と検証が実行される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.ensemble_config.enabled = False
        sample_training_config.ensemble_config.model_dump.return_value = {
            "enabled": False
        }
        sample_training_config.single_model_config = MagicMock()
        sample_training_config.single_model_config.model_type = "lightgbm"
        sample_training_config.single_model_config.model_dump.return_value = {
            "model_type": "lightgbm"
        }

        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"is_training": False},
            ),
            patch.object(orchestration_service, "validate_training_config"),
        ):
            orchestration_service._log_and_validate_config(sample_training_config)


class TestExecuteMLTraining:
    """正常系: MLトレーニング実行のテスト"""

    def test_execute_ml_training_success(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        正常系: MLトレーニングが正常に実行される

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"is_training": False},
            ) as mock_status,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.MLTrainingService"
            ) as mock_service_class,
        ):
            mock_service = MagicMock()
            mock_service.train_model.return_value = {"accuracy": 0.85}
            mock_service_class.return_value = mock_service

            training_data = {"X": [], "y": []}

            orchestration_service._execute_ml_training_with_error_handling(
                trainer_type="ensemble",
                ensemble_config_dict={"enabled": True},
                single_model_config_dict=None,
                config=sample_training_config,
                training_data=training_data,
            )

            mock_service.train_model.assert_called_once()


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_start_training_with_exception(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
        mock_background_tasks: MagicMock,
        mock_db_session: MagicMock,
    ):
        """
        異常系: トレーニング開始中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
            mock_background_tasks: バックグラウンドタスクモック
            mock_db_session: DBセッションモック
        """
        with patch.object(
            orchestration_service, "_log_and_validate_config"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Unexpected error")

            with pytest.raises(Exception, match="Unexpected error"):
                await orchestration_service.start_training(
                    config=sample_training_config,
                    background_tasks=mock_background_tasks,
                    db=mock_db_session,
                )

    @pytest.mark.asyncio
    async def test_get_model_info_with_exception(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        異常系: モデル情報取得中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
        ) as mock_manager:
            mock_manager.get_latest_model.side_effect = Exception("Manager error")

            with pytest.raises(Exception, match="Manager error"):
                await orchestration_service.get_model_info()


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_training_status_with_error_state(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        境界値: エラー状態のステータス取得

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {
                "is_training": False,
                "status": "error",
                "error": "Training failed",
                "progress": 50,
            },
        ):
            result = await orchestration_service.get_training_status()

            assert result["is_training"] is False
            assert result["status"] == "error"
            assert "error" in result

    def test_validate_training_config_exactly_7_days(
        self,
        orchestration_service: MLTrainingOrchestrationService,
        sample_training_config: MagicMock,
    ):
        """
        境界値: ちょうど7日間のトレーニング期間

        Args:
            orchestration_service: オーケストレーションサービス
            sample_training_config: サンプル設定
        """
        sample_training_config.start_date = "2024-01-01T00:00:00"
        sample_training_config.end_date = "2024-01-08T00:00:00"

        with patch(
            "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
            {"is_training": False},
        ):
            # 例外が発生しないことを確認
            orchestration_service.validate_training_config(sample_training_config)

    @pytest.mark.asyncio
    async def test_get_model_info_with_metadata_error(
        self, orchestration_service: MLTrainingOrchestrationService
    ):
        """
        境界値: メタデータ読み込みエラーの場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with (
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.model_manager"
            ) as mock_manager,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.os.path.exists"
            ) as mock_exists,
            patch(
                "app.services.ml.orchestration.ml_training_orchestration_service.training_status",
                {"model_info": None},
            ),
        ):
            mock_manager.get_latest_model.return_value = "/models/test.pkl"
            mock_exists.return_value = True
            mock_manager.load_model.side_effect = Exception("Load error")

            result = await orchestration_service.get_model_info()

            assert result["success"] is True
            assert result["data"]["model_status"]["is_loaded"] is False
