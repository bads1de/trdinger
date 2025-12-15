"""
オープンインタレストオーケストレーションサービスのテストモジュール

OpenInterestOrchestrationServiceの正常系、異常系、エッジケースをテストします。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.data_collection.orchestration.open_interest_orchestration_service import (
    OpenInterestOrchestrationService,
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
    BybitOpenInterestServiceのモック

    Returns:
        AsyncMock: モックされたBybitサービス
    """
    return AsyncMock()


@pytest.fixture
def orchestration_service(
    mock_bybit_service: AsyncMock,
) -> OpenInterestOrchestrationService:
    """
    OpenInterestOrchestrationServiceのインスタンス

    Args:
        mock_bybit_service: モックされたBybitサービス

    Returns:
        OpenInterestOrchestrationService: テスト対象のサービス
    """
    return OpenInterestOrchestrationService(bybit_service=mock_bybit_service)


@pytest.fixture
def sample_open_interest_data() -> List[Dict[str, Any]]:
    """
    サンプルオープンインタレストデータ

    Returns:
        List[Dict[str, Any]]: オープンインタレストデータのリスト
    """
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "open_interest_value": 1000000.0,
            "data_timestamp": datetime(2024, 1, 1, 0, 0, 0),
            "timestamp": datetime(2024, 1, 1, 0, 0, 0),
        },
        {
            "symbol": "BTC/USDT:USDT",
            "open_interest_value": 1100000.0,
            "data_timestamp": datetime(2024, 1, 1, 1, 0, 0),
            "timestamp": datetime(2024, 1, 1, 1, 0, 0),
        },
    ]


class TestServiceInitialization:
    """正常系: サービスの初期化テスト"""

    def test_service_creation(
        self, orchestration_service: OpenInterestOrchestrationService
    ):
        """
        正常系: サービスが正常に初期化される

        Args:
            orchestration_service: オーケストレーションサービス
        """
        assert orchestration_service is not None
        assert isinstance(orchestration_service, OpenInterestOrchestrationService)
        assert orchestration_service.bybit_service is not None


class TestCollectOpenInterestData:
    """正常系: オープンインタレストデータ収集のテスト"""

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_success(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: オープンインタレストデータが正常に収集される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        # モックの設定
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={"success": True, "saved_count": 10}
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.collect_open_interest_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                fetch_all=False,
                db_session=mock_db_session,
            )

            # レスポンス形式変更に伴う修正
            assert result["success"] is True
            assert result["data"]["saved_count"] == 10

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_fetch_all(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: 全データ収集モードで正常に動作する

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={"success": True, "saved_count": 100}
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ):
            result = await orchestration_service.collect_open_interest_data(
                symbol="BTC/USDT:USDT",
                fetch_all=True,
                db_session=mock_db_session,
            )

            # レスポンス形式変更に伴う修正
            assert result["success"] is True
            assert result["data"]["saved_count"] == 100

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_failure(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: データ収集が失敗した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={
                "success": False,
                "error": "API error",
                "saved_count": 0,
            }
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ):
            result = await orchestration_service.collect_open_interest_data(
                symbol="BTC/USDT:USDT",
                db_session=mock_db_session,
            )

            assert result["success"] is False
            assert "error" in result["details"]


class TestGetOpenInterestData:
    """正常系: オープンインタレストデータ取得のテスト"""

    @pytest.mark.asyncio
    async def test_get_open_interest_data_success(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
        sample_open_interest_data: List[Dict[str, Any]],
    ):
        """
        正常系: オープンインタレストデータが正常に取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
            sample_open_interest_data: サンプルデータ
        """
        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            # リポジトリが返すオブジェクトのモック作成
            mock_records = []
            for data in sample_open_interest_data:
                record = MagicMock()
                record.symbol = data["symbol"]
                record.open_interest_value = data["open_interest_value"]
                record.data_timestamp = data["data_timestamp"]
                record.timestamp = data["timestamp"]
                mock_records.append(record)
            
            mock_repo.get_open_interest_data.return_value = mock_records
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_open_interest_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 2

    @pytest.mark.asyncio
    async def test_get_open_interest_data_with_dates(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: 日付範囲を指定してデータが取得できる

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_open_interest_data.return_value = []
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_open_interest_data(
                symbol="BTC/USDT:USDT",
                limit=100,
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00",
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 0


class TestCollectBulkOpenInterestData:
    """正常系: 一括データ収集のテスト"""

    @pytest.mark.asyncio
    async def test_collect_bulk_open_interest_data_success(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: 複数シンボルのデータが正常に収集される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={"success": True, "saved_count": 10}
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ):
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            result = await orchestration_service.collect_bulk_open_interest_data(
                symbols=symbols, db_session=mock_db_session
            )

            # レスポンス形式変更に伴う修正
            assert result["success"] is True
            assert result["data"]["total_saved"] == 20  # 2 symbols * 10 each

    @pytest.mark.asyncio
    async def test_collect_bulk_open_interest_data_with_errors(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: 一部のシンボルでエラーが発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            side_effect=Exception("API error")
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ):
            symbols = ["BTC/USDT:USDT"]
            result = await orchestration_service.collect_bulk_open_interest_data(
                symbols=symbols, db_session=mock_db_session
            )

            # レスポンス形式変更に伴う修正
            assert result["success"] is True
            assert result["data"]["total_saved"] == 0
            assert len(result["data"]["failed_symbols"]) == 1


class TestErrorHandling:
    """異常系: エラーハンドリングのテスト"""

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_with_exception(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: データ収集中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            side_effect=Exception("Service error")
        )

        result = await orchestration_service.collect_open_interest_data(
            symbol="BTC/USDT:USDT",
            db_session=mock_db_session,
        )

        assert result["success"] is False
        assert "error" in result["details"]

    @pytest.mark.asyncio
    async def test_get_open_interest_data_with_exception(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        異常系: データ取得中に例外が発生した場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo_class.side_effect = Exception("Repository error")

            result = await orchestration_service.get_open_interest_data(
                symbol="BTC/USDT:USDT",
                db_session=mock_db_session,
            )

            assert result["success"] is False
            assert "error" in result["details"]


class TestEdgeCases:
    """境界値テスト"""

    @pytest.mark.asyncio
    async def test_collect_open_interest_data_with_zero_limit(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: limit=0の場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        orchestration_service.bybit_service.fetch_and_save_open_interest_data = AsyncMock(
            return_value={"success": True, "saved_count": 0}
        )

        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ):
            result = await orchestration_service.collect_open_interest_data(
                symbol="BTC/USDT:USDT",
                limit=0,
                db_session=mock_db_session,
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_collect_bulk_open_interest_data_empty_symbols(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: 空のシンボルリスト

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        result = await orchestration_service.collect_bulk_open_interest_data(
            symbols=[], db_session=mock_db_session
        )

        # レスポンス形式変更に伴う修正
        assert result["success"] is True
        assert result["data"]["total_saved"] == 0

    @pytest.mark.asyncio
    async def test_get_open_interest_data_empty_result(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        境界値: データが存在しない場合

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_open_interest_data.return_value = []
            mock_repo_class.return_value = mock_repo

            result = await orchestration_service.get_open_interest_data(
                symbol="BTC/USDT:USDT",
                db_session=mock_db_session,
            )

            assert result["success"] is True
            assert result["data"]["count"] == 0


class TestSymbolNormalization:
    """正常系: シンボル正規化のテスト"""

    @pytest.mark.asyncio
    async def test_symbol_normalization(
        self,
        orchestration_service: OpenInterestOrchestrationService,
        mock_db_session: MagicMock,
    ):
        """
        正常系: シンボルが正規化される

        Args:
            orchestration_service: オーケストレーションサービス
            mock_db_session: DBセッションモック
        """
        with patch(
            "app.services.data_collection.orchestration.open_interest_orchestration_service.OpenInterestRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_open_interest_data.return_value = []
            mock_repo_class.return_value = mock_repo

            # シンボル形式の違いをテスト
            symbols = ["BTC/USDT:USDT", "ETH/USD", "SOL/USDT:USDT"]
            for symbol in symbols:
                result = await orchestration_service.get_open_interest_data(
                    symbol=symbol,
                    db_session=mock_db_session,
                )
                assert result["success"] is True

