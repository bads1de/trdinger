"""
ロング/ショート比率オーケストレーションサービスのテストモジュール

LongShortRatioOrchestrationServiceの全機能をテストします:
- サービス初期化
- データ取得 (get_long_short_ratio_data)
- データ収集 (collect_long_short_ratio_data) - 差分/全件
- 一括収集 (collect_bulk_long_short_ratio_data)
- エラーハンドリング
- 境界値ケース
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.orchestration.long_short_ratio_orchestration_service import (
    LongShortRatioOrchestrationService,
)


@pytest.fixture
def mock_db_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_bybit_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def orchestration_service(
    mock_bybit_service: AsyncMock,
) -> LongShortRatioOrchestrationService:
    return LongShortRatioOrchestrationService(bybit_service=mock_bybit_service)


@pytest.fixture
def sample_lsr_data() -> List[MagicMock]:
    """サンプルLSRレコード"""
    records = []
    for i, (ts, buy, sell) in enumerate(
        [
            (datetime(2024, 1, 1, 0, 0), 0.6, 0.4),
            (datetime(2024, 1, 1, 1, 0), 0.55, 0.45),
            (datetime(2024, 1, 1, 2, 0), 0.5, 0.5),
        ]
    ):
        record = MagicMock()
        record.timestamp = ts
        record.buy_ratio = buy
        record.sell_ratio = sell
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# サービス初期化
# ---------------------------------------------------------------------------

class TestServiceInitialization:
    """サービス初期化テスト"""

    def test_service_creation(self, orchestration_service: LongShortRatioOrchestrationService):
        assert orchestration_service is not None
        assert isinstance(orchestration_service, LongShortRatioOrchestrationService)
        assert orchestration_service.bybit_service is not None


# ---------------------------------------------------------------------------
# get_long_short_ratio_data
# ---------------------------------------------------------------------------

class TestGetLongShortRatioData:
    """データ取得テスト"""

    @pytest.mark.asyncio
    async def test_get_data_success(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
        sample_lsr_data: List[MagicMock],
    ):
        """正常系: LSRデータが取得できる"""
        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_long_short_ratio_data.return_value = sample_lsr_data
            mock_repo_cls.return_value = mock_repo

            result = await orchestration_service.get_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                limit=100,
                start_date=None,
                end_date=None,
                db_session=mock_db_session,
            )

            assert len(result) == 3
            mock_repo.get_long_short_ratio_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_with_date_range(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """正常系: 日付範囲を指定して取得できる"""
        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_long_short_ratio_data.return_value = []
            mock_repo_cls.return_value = mock_repo

            result = await orchestration_service.get_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                limit=100,
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00",
                db_session=mock_db_session,
            )

            assert result == []
            call_kwargs = mock_repo.get_long_short_ratio_data.call_args.kwargs
            assert call_kwargs["start_time"] is not None
            assert call_kwargs["end_time"] is not None

    @pytest.mark.asyncio
    async def test_get_data_empty_result(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """エッジケース: データが存在しない場合は空リストを返す"""
        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get_long_short_ratio_data.return_value = []
            mock_repo_cls.return_value = mock_repo

            result = await orchestration_service.get_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                limit=100,
                start_date=None,
                end_date=None,
                db_session=mock_db_session,
            )

            assert result == []


# ---------------------------------------------------------------------------
# collect_long_short_ratio_data
# ---------------------------------------------------------------------------

class TestCollectLongShortRatioData:
    """データ収集テスト"""

    @pytest.mark.asyncio
    async def test_collect_incremental_success(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """正常系: 差分更新でデータ収集が成功する"""
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(return_value={"saved_count": 12})
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 12

    @pytest.mark.asyncio
    async def test_collect_fetch_all_success(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """正常系: 全件取得(fetch_all=True)でデータ収集が成功する"""
        orchestration_service.bybit_service.collect_historical_long_short_ratio_data = (
            AsyncMock(return_value=50)
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=True,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 50
            orchestration_service.bybit_service.collect_historical_long_short_ratio_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_incremental_error_returns_error_response(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """異常系: 差分更新でエラーが発生した場合はエラーレスポンスを返す"""
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(side_effect=Exception("API error"))
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["success"] is False
            assert "API error" in result["message"]

    @pytest.mark.asyncio
    async def test_collect_fetch_all_error_returns_error_response(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """異常系: 全件取得でエラーが発生した場合はエラーレスポンスを返す"""
        orchestration_service.bybit_service.collect_historical_long_short_ratio_data = (
            AsyncMock(side_effect=Exception("Network timeout"))
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=True,
                db_session=mock_db_session,
            )

            assert result["success"] is False
            assert "Network timeout" in result["message"]

    @pytest.mark.asyncio
    async def test_collect_returns_zero_count_on_empty(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """エッジケース: データ0件でも正常レスポンスを返す"""
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(return_value={"saved_count": 0})
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_long_short_ratio_data(
                symbol="BTC/USDT:USDT",
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 0


# ---------------------------------------------------------------------------
# collect_bulk_long_short_ratio_data
# ---------------------------------------------------------------------------

class TestCollectBulkLongShortRatioData:
    """一括データ収集テスト"""

    @pytest.mark.asyncio
    async def test_bulk_collect_success(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """正常系: 複数シンボルの一括収集が成功する"""
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(return_value={"saved_count": 10})
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_bulk_long_short_ratio_data(
                symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["total_count"] == 20  # 2 symbols * 10

    @pytest.mark.asyncio
    async def test_bulk_collect_with_partial_failure(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """異常系: 一部シンボルでエラーが発生しても他は継続する"""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"saved_count": 10}
            raise Exception("API error for second symbol")

        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(side_effect=side_effect)
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_bulk_long_short_ratio_data(
                symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            # 1件目は成功、2件目はエラーで total_count は 10
            assert result["success"] is True
            assert result["data"]["total_count"] == 10

    @pytest.mark.asyncio
    async def test_bulk_collect_empty_symbols(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """境界値: 空のシンボルリストでは0件を返す"""
        result = await orchestration_service.collect_bulk_long_short_ratio_data(
            symbols=[],
            period="1h",
            fetch_all=False,
            db_session=mock_db_session,
        )

        assert result["success"] is True
        assert result["data"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_collect_single_symbol(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """境界値: シンボル1件のみでも正しく動作する"""
        orchestration_service.bybit_service.collect_historical_long_short_ratio_data = (
            AsyncMock(return_value=25)
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_bulk_long_short_ratio_data(
                symbols=["BTC/USDT:USDT"],
                period="1h",
                fetch_all=True,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["total_count"] == 25

    @pytest.mark.asyncio
    async def test_bulk_collect_all_symbols_fail(
        self,
        orchestration_service: LongShortRatioOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """異常系: 全シンボルでエラーが発生した場合"""
        orchestration_service.bybit_service.fetch_incremental_long_short_ratio_data = (
            AsyncMock(side_effect=Exception("Total failure"))
        )

        with patch(
            "app.services.data_collection.orchestration.long_short_ratio_orchestration_service.LongShortRatioRepository"
        ):
            result = await orchestration_service.collect_bulk_long_short_ratio_data(
                symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
                period="1h",
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["total_count"] == 0
