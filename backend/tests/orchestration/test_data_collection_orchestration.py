"""
データ収集オーケストレーションサービスのテストモジュール

DataCollectionOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.config.unified_config import unified_config
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
        MagicMock: モックされたBackgroundTasks
    """
    return MagicMock(spec=BackgroundTasks)


@pytest.fixture
def orchestration_service() -> DataCollectionOrchestrationService:
    """
    DataCollectionOrchestrationServiceのインスタンス

    Returns:
        DataCollectionOrchestrationService: テスト対象のサービス
    """
    return DataCollectionOrchestrationService()


@pytest.fixture
def mock_ohlcv_repository() -> MagicMock:
    """
    OHLCVRepositoryのモック

    Returns:
        MagicMock: モックされたリポジトリ
    """
    return MagicMock()


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
    """正常系・異常系: シンボルと時間軸のバリデーションテスト"""

    def test_validate_symbol_and_timeframe_success(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        正常系: 有効なシンボルと時間軸が検証される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h", "4h", "1d"]

            result = orchestration_service.validate_symbol_and_timeframe(
                "BTC/USDT:USDT", "1h"
            )

            assert result == "BTC/USDT:USDT"

    def test_validate_symbol_and_timeframe_with_mapping(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        正常系: シンボルマッピングが適用される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        # unified_config.market をモックする
        with patch.object(unified_config, "market") as mock_market_config:
            mock_market_config.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_market_config.supported_symbols = ["BTC/USDT:USDT"]
            mock_market_config.supported_timeframes = ["1h"]

            result = orchestration_service.validate_symbol_and_timeframe("BTC", "1h")
            assert result == "BTC/USDT:USDT"

            assert result == "BTC/USDT:USDT"

    def test_validate_symbol_and_timeframe_invalid_symbol(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        異常系: サポートされていないシンボルでValueError

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            with pytest.raises(ValueError, match="サポートされていないシンボル"):
                orchestration_service.validate_symbol_and_timeframe(
                    "INVALID/SYMBOL", "1h"
                )

    def test_validate_symbol_and_timeframe_invalid_timeframe(
        self, orchestration_service: DataCollectionOrchestrationService
    ):
        """
        異常系: サポートされていない時間軸でValueError

        Args:
            orchestration_service: オーケストレーションサービス
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            with pytest.raises(ValueError, match="無効な時間軸"):
                orchestration_service.validate_symbol_and_timeframe(
                    "BTC/USDT:USDT", "5m"
                )


class TestStartHistoricalDataCollection:
    """正常系: 履歴データ収集開始のテスト"""

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_success(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 履歴データ収集が正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert result["status"] == "started"
            assert "履歴データ収集を開始" in result["message"]
            mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_data_exists(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: データが既に存在する場合はスキップ

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 1000
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=False,
            )

            assert result["success"] is True
            assert result["status"] == "exists"
            assert "既に存在" in result["message"]
            mock_background_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_force_update(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 強制更新モードでデータを削除して再収集

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 1000
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.return_value = 1000
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_historical_data_collection(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=True,
            )

            assert result["success"] is True
            assert result["status"] == "started"
            assert "強制更新モード" in result["message"]
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_called_once()
            mock_background_tasks.add_task.assert_called_once()


class TestExecuteBulkIncrementalUpdate:
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
        mock_result = {
            "data": {
                "ohlcv_updates": 100,
                "funding_rate_updates": 50,
                "open_interest_updates": 75,
            }
        }

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
            orchestration_service.historical_service.collect_bulk_incremental_data = (
                AsyncMock(return_value=mock_result)
            )

            result = await orchestration_service.execute_bulk_incremental_update(
                symbol="BTC/USDT:USDT", db=mock_db_session
            )

            assert result["success"] is True
            assert "一括差分更新が完了" in result["message"]
            assert result["data"] == mock_result

    @pytest.mark.asyncio
    async def test_execute_bulk_incremental_update_error(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 一括差分更新でエラーが発生

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
            orchestration_service.historical_service.collect_bulk_incremental_data = (
                AsyncMock(side_effect=Exception("Update failed"))
            )

            with pytest.raises(Exception, match="Update failed"):
                await orchestration_service.execute_bulk_incremental_update(
                    symbol="BTC/USDT:USDT", db=mock_db_session
                )


class TestStartBitcoinFullDataCollection:
    """正常系: ビットコイン全時間軸データ収集のテスト"""

    @pytest.mark.asyncio
    async def test_start_bitcoin_full_data_collection_success(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: ビットコイン全時間軸データ収集が正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        result = await orchestration_service.start_bitcoin_full_data_collection(
            background_tasks=mock_background_tasks, db=mock_db_session
        )

        assert result["success"] is True
        assert result["status"] == "started"
        assert "ビットコインの全時間軸データ収集を開始" in result["message"]
        # タイムフレームはunified_configから取得
        expected_timeframes = unified_config.market.supported_timeframes
        assert result["data"]["timeframes"] == expected_timeframes
        assert mock_background_tasks.add_task.call_count == len(expected_timeframes)


class TestGetCollectionStatus:
    """正常系: データ収集状況確認のテスト"""

    @pytest.mark.asyncio
    async def test_get_collection_status_data_exists(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: データが存在する場合の状況確認

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        from datetime import datetime

        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 1000
            mock_repo.get_latest_timestamp.return_value = datetime(2024, 1, 31)
            mock_repo.get_oldest_timestamp.return_value = datetime(2024, 1, 1)
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_collection_status(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                auto_fetch=False,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["data_count"] == 1000
            assert result["data"]["status"] == "data_exists"

    @pytest.mark.asyncio
    async def test_get_collection_status_no_data_auto_fetch(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: データが存在せず自動フェッチが有効

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_collection_status(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                auto_fetch=True,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert result["status"] == "auto_fetch_started"
            assert "自動収集を開始" in result["message"]

    @pytest.mark.asyncio
    async def test_get_collection_status_no_data_no_auto_fetch(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: データが存在せず自動フェッチが無効

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with (
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
            ) as mock_config,
            patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
            ) as mock_repo_class,
        ):
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_collection_status(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=mock_background_tasks,
                auto_fetch=False,
                db=mock_db_session,
            )

            assert result["success"] is True
            assert result["status"] == "no_data"
            assert "データが存在しません" in result["message"]
            assert "suggestion" in result["data"]


class TestStartBulkHistoricalDataCollection:
    """正常系: 一括履歴データ収集のテスト"""

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
            mock_background_tasks: BackgroundTasksモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_bulk_historical_data_collection(
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=False,
            )

            assert result["success"] is True
            assert result["status"] == "started"
            assert "collection_tasks" in result["data"]
            assert result["data"]["force_update"] is False

    @pytest.mark.asyncio
    async def test_start_bulk_historical_data_collection_force_update(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 強制更新モードでの一括履歴データ収集

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 1000
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.return_value = 1000
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_bulk_historical_data_collection(
                background_tasks=mock_background_tasks,
                db=mock_db_session,
                force_update=True,
            )

            assert result["success"] is True
            assert result["data"]["force_update"] is True
            assert "強制更新モード" in result["message"]


class TestStartAllDataBulkCollection:
    """正常系: 全データ一括収集のテスト"""

    @pytest.mark.asyncio
    async def test_start_all_data_bulk_collection_success(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        正常系: 全データ一括収集が正常に開始される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_data_count.return_value = 0
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.start_all_data_bulk_collection(
                background_tasks=mock_background_tasks, db=mock_db_session
            )

            assert result["success"] is True
            assert result["status"] == "started"
            assert "全データ一括収集を開始" in result["message"]
            # タイムフレーム数はunified_configから取得
            expected_count = len(unified_config.market.supported_timeframes)
            assert result["data"]["collection_tasks"] == expected_count


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_start_historical_data_collection_invalid_symbol(
        self,
        orchestration_service: DataCollectionOrchestrationService,
        mock_db_session: MagicMock,
        mock_background_tasks: MagicMock,
    ):
        """
        異常系: 無効なシンボルでエラーが発生

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            mock_background_tasks: BackgroundTasksモック
        """
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.unified_config"
        ) as mock_config:
            mock_config.market.symbol_mapping = {}
            mock_config.market.supported_symbols = ["BTC/USDT:USDT"]
            mock_config.market.supported_timeframes = ["1h"]

            with pytest.raises(ValueError):
                await orchestration_service.start_historical_data_collection(
                    symbol="INVALID/SYMBOL",
                    timeframe="1h",
                    background_tasks=mock_background_tasks,
                    db=mock_db_session,
                )
