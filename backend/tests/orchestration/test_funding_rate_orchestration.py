"""
ファンディングレートオーケストレーションサービスのテストモジュール

FundingRateOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.orchestration.funding_rate_orchestration_service import (
    FundingRateOrchestrationService,
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
def mock_bybit_service() -> AsyncMock:
    """
    BybitFundingRateServiceのモック

    Returns:
        AsyncMock: モックされたBybitサービス
    """
    return AsyncMock()


@pytest.fixture
def orchestration_service(
    mock_bybit_service: AsyncMock,
) -> FundingRateOrchestrationService:
    """
    FundingRateOrchestrationServiceのインスタンス

    Args:
        mock_bybit_service: モックされたBybitサービス

    Returns:
        FundingRateOrchestrationService: テスト対象のサービス
    """
    return FundingRateOrchestrationService(bybit_service=mock_bybit_service)


@pytest.fixture
def sample_funding_rate_data() -> List[Dict[str, Any]]:
    """
    サンプルファンディングレートデータ

    Returns:
        List[Dict[str, Any]]: ファンディングレートデータのリスト
    """
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "funding_rate": 0.0001,
            "timestamp": datetime(2024, 1, 1, 0, 0, 0),
        },
        {
            "symbol": "BTC/USDT:USDT",
            "funding_rate": 0.00015,
            "timestamp": datetime(2024, 1, 1, 8, 0, 0),
        },
    ]


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: FundingRateOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, FundingRateOrchestrationService)
        assert orchestration_service.bybit_service is not None


class TestParseDatetime:
    """正常系: 日付パーステスト"""

    def test_parse_datetime_success(
        self, orchestration_service: FundingRateOrchestrationService
    ):
        """
        正常系: ISO形式の日付文字列が正常にパースされる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        date_str = "2024-01-01T00:00:00"
        result = orchestration_service._parse_datetime(date_str)

        assert result is not None
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_parse_datetime_with_z(
        self, orchestration_service: FundingRateOrchestrationService
    ):
        """
        正常系: Z付きの日付文字列が正常にパースされる

        Args:
            orchestration_service: オーケストレーションサービス
        """
        date_str = "2024-01-01T00:00:00Z"
        result = orchestration_service._parse_datetime(date_str)

        assert result is not None
        assert isinstance(result, datetime)

    def test_parse_datetime_none(
        self, orchestration_service: FundingRateOrchestrationService
    ):
        """
        境界値: Noneが渡された場合

        Args:
            orchestration_service: オーケストレーションサービス
        """
        result = orchestration_service._parse_datetime(None)
        assert result is None

    def test_parse_datetime_invalid_format(
        self, orchestration_service: FundingRateOrchestrationService
    ):
        """
        異常系: 無効な形式の日付文字列

        Args:
            orchestration_service: オーケストレーションサービス
        """
        result = orchestration_service._parse_datetime("invalid-date")
        assert result is None


class TestGetFundingRateData:
    """正常系: ファンディングレートデータ取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_funding_rate_data_success(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
        sample_funding_rate_data: List[Dict[str, Any]],
    ):
        """
        正常系: ファンディングレートデータが正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_funding_rate_data: サンプルデータ
        """
        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_funding_rate_data.return_value = sample_funding_rate_data
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                start_date=None,
                end_date=None,
                db_session=mock_db_session,
            )

            assert len(result) == 2
            mock_repo.get_funding_rate_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_funding_rate_data_with_dates(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: 日付範囲を指定してデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_funding_rate_data.return_value = []
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00",
                db_session=mock_db_session,
            )

            assert result == []


class TestCollectFundingRateData:
    """正常系: ファンディングレートデータ収集のテスト"""

    @pytest.mark.asyncio
    async def test_collect_funding_rate_data_success(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
        sample_funding_rate_data: List[Dict[str, Any]],
    ):
        """
        正常系: ファンディングレートデータが正常に収集される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_funding_rate_data: サンプルデータ
        """
        orchestration_service.bybit_service.fetch_funding_rate_history = AsyncMock(
            return_value=sample_funding_rate_data
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.insert_funding_rate_data.return_value = 2
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.collect_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["status"] == "success"
            assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_collect_funding_rate_data_fetch_all(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
        sample_funding_rate_data: List[Dict[str, Any]],
    ):
        """
        正常系: 全データ収集モードで正常に動作する

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_funding_rate_data: サンプルデータ
        """
        orchestration_service.bybit_service.fetch_all_funding_rate_history = AsyncMock(
            return_value=sample_funding_rate_data
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.insert_funding_rate_data.return_value = 2
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.collect_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                fetch_all=True,
                db_session=mock_db_session,
            )

            assert result["status"] == "success"
            assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_collect_funding_rate_data_no_data(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        エッジケース: データが見つからない場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_funding_rate_history = AsyncMock(
            return_value=[]
        )

        result = await orchestration_service.collect_funding_rate_data(
            symbol="BTC/USDT:USDT",
            limit=100,
            fetch_all=False,
            db_session=mock_db_session,
        )

        assert result["status"] == "success"
        assert result["count"] == 0
        assert "データなし" in result["message"]


class TestCollectBulkFundingRateData:
    """正常系: 一括データ収集のテスト"""

    @pytest.mark.asyncio
    async def test_collect_bulk_funding_rate_data_success(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
        sample_funding_rate_data: List[Dict[str, Any]],
    ):
        """
        正常系: 複数シンボルのデータが正常に収集される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_funding_rate_data: サンプルデータ
        """
        orchestration_service.bybit_service.fetch_all_funding_rate_history = AsyncMock(
            return_value=sample_funding_rate_data
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.insert_funding_rate_data.return_value = 2
            mock_repo_class.return_value = mock_repo

            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            result = await orchestration_service.collect_bulk_funding_rate_data(
                symbols=symbols, db_session=mock_db_session
            )

            assert result["status"] == "success"
            assert result["total_count"] == 4  # 2 symbols * 2 data each

    @pytest.mark.asyncio
    async def test_collect_bulk_funding_rate_data_with_errors(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 一部のシンボルでエラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_all_funding_rate_history = AsyncMock(
            side_effect=Exception("API error")
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ):
            symbols = ["BTC/USDT:USDT"]
            result = await orchestration_service.collect_bulk_funding_rate_data(
                symbols=symbols, db_session=mock_db_session
            )

            assert result["status"] == "success"
            assert result["total_count"] == 0


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_collect_funding_rate_data_with_exception(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: データ収集中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_funding_rate_history = AsyncMock(
            side_effect=Exception("Network error")
        )

        with pytest.raises(Exception, match="Network error"):
            await orchestration_service.collect_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                fetch_all=False,
                db_session=mock_db_session,
            )


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_get_funding_rate_data_with_zero_limit(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: limit=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_funding_rate_data.return_value = []
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=0,
                start_date=None,
                end_date=None,
                db_session=mock_db_session,
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_collect_bulk_funding_rate_data_empty_symbols(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: 空のシンボルリスト

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        result = await orchestration_service.collect_bulk_funding_rate_data(
            symbols=[], db_session=mock_db_session
        )

        assert result["status"] == "success"
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_collect_funding_rate_data_large_limit(
        self,
        orchestration_service: FundingRateOrchestrationService,
        mock_db_session: MagicMock,
        sample_funding_rate_data: List[Dict[str, Any]],
    ):
        """
        境界値: 非常に大きなlimit値

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_funding_rate_data: サンプルデータ
        """
        orchestration_service.bybit_service.fetch_funding_rate_history = AsyncMock(
            return_value=sample_funding_rate_data
        )

        with patch(
            "app.services.data_collection.orchestration.funding_rate_orchestration_service.FundingRateRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.insert_funding_rate_data.return_value = 2
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.collect_funding_rate_data(
                symbol="BTC/USDT:USDT",
                limit=999999,
                fetch_all=False,
                db_session=mock_db_session,
            )

            assert result["status"] == "success"
