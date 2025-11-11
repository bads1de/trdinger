"""
データ収集オーケストレーションサービスのテストモジュール

DataCollectionOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
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
def orchestration_service() -> DataCollectionOrchestrationService:
    """
    DataCollectionOrchestrationServiceのインスタンス

    Returns:
        DataCollectionOrchestrationService: テスト対象のサービス
    """
    return DataCollectionOrchestrationService()


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, DataCollectionOrchestrationService)
        assert orchestration_service.historical_service is not None


class TestValidateSymbolAndTimeframe:
    """正常系: シンボルと時間軸のバリデーションテスト"""

    def test_validate_symbol_and_timeframe_success(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        正常系: 有効なシンボルと時間軸がバリデーションを通過する

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h", "4h", "1d"]

            result = orchestration_service.validate_symbol_and_timeframe("BTC", "1h")

            assert result == "BTC/USDT:USDT"

    def test_validate_symbol_and_timeframe_invalid_symbol(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        異常系: 無効なシンボルでValueErrorが発生する

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]

            with pytest.raises(ValueError, match="サポートされていないシンボル"):
                orchestration_service.validate_symbol_and_timeframe("INVALID", "1h")

    def test_validate_symbol_and_timeframe_invalid_timeframe(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        異常系: 無効な時間軸でValueErrorが発生する

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h", "4h"]

            with pytest.raises(ValueError, match="無効な時間軸"):
                orchestration_service.validate_symbol_and_timeframe("BTC", "5m")


class TestStartHistoricalDataCollection:
    """正常系: 履歴データ収集開始のテスト"""

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_new_data(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 新規データ収集が正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: バックグラウンドタスクモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert result["status"] == "started"
            mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_existing_data(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 既存データがある場合はスキップされる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: バックグラウンドタスクモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 100
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=False,
            )

            assert result["success"] is True
            assert result["status"] == "exists"
            mock_background_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_force_update(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 強制更新モードで既存データが削除され再収集される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: バックグラウンドタスクモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 100
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.return_value = 100
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=True,
            )

            assert result["success"] is True
            assert "強制更新" in result["message"]
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_called_once()
            mock_background_tasks.add_task.assert_called_once()


class TestBulkIncrementalUpdate:
    """正常系: 一括差分更新のテスト"""

    @pytest.mark.asyncio
    async def test_execute_bulk_incremental_update_success(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: 一括差分更新が正常に実行される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ),
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.FundingRateRepository"
            ),
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OpenInterestRepository"
            ),
        ):
            orchestration_service.historical_service = AsyncMock()
            orchestration_service.historical_service.collect_bulk_incremental_data = (
                AsyncMock(
                    return_value={
                        "ohlcv_updated": 10,
                        "funding_rate_updated": 5,
                        "open_interest_updated": 8,
                    }
                )
            )

            result = await orchestration_service.execute_bulk_incremental_update(
                symbol="BTC/USDT:USDT", db=mock_db_session
            )

            assert result["success"] is True
            assert "完了" in result["message"]


class TestGetCollectionStatus:
    """正常系: データ収集状況確認のテスト"""


class TestBulkDataCollection:
    """正常系: 一括データ収集のテスト"""

    @pytest.mark.asyncio
    async def test_start_bulk_historical_data_collection_success(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 一括履歴データ収集が正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: バックグラウンドタスクモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_bulk_historical_data_collection(
                background_tasks=mock_background_tasks, db=mock_db_session
            )

            assert result["success"] is True
            assert result["status"] == "started"
            assert "collection_tasks" in result["data"]


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_bulk_incremental_update_with_exception(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 一括差分更新でエラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_repo_class:
            mock_repo_class.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                await orchestration_service.execute_bulk_incremental_update(
                    symbol="BTC/USDT:USDT", db=mock_db_session
                )


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_with_start_date(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        境界値: 開始日付を指定した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: バックグラウンドタスクモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                start_date="2024-01-01",
            )

            assert result["success"] is True
            assert result["status"] == "started"
