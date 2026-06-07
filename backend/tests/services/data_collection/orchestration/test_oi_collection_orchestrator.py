"""
OICollectionOrchestrator のユニットテスト

Open Interest 収集オーケストレーターの正常系・異常系・エッジケースを網羅的にテストします。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.services.data_collection.orchestration.oi_collection_orchestrator import (
    OICollectionOrchestrator,
)


class TestOICollectionOrchestratorStart:
    """OICollectionOrchestrator.start_historical_oi_collection のテスト"""

    @pytest.fixture
    def orchestrator(self):
        return OICollectionOrchestrator()

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def mock_data_validator(self):
        return MagicMock()

    @pytest.fixture
    def mock_background_tasks(self):
        return MagicMock(spec=BackgroundTasks)

    @pytest.mark.asyncio
    async def test_start_collection_uses_data_validator_when_provided(
        self, orchestrator, mock_db, mock_data_validator, mock_background_tasks
    ):
        """data_validator が提供された場合の動作テスト"""
        mock_data_validator.validate_symbol_and_timeframe.return_value = "BTC/USDT:USDT"

        result = await orchestrator.start_historical_oi_collection(
            symbol="BTC/USDT",
            interval="1h",
            background_tasks=mock_background_tasks,
            db=mock_db,
            data_validator=mock_data_validator,
        )

        assert result["success"] is True
        assert "BTC/USDT:USDT" in result["message"]
        assert result["status"] == "started"
        mock_data_validator.validate_symbol_and_timeframe.assert_called_once_with(
            "BTC/USDT", "1h"
        )
        mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_collection_falls_back_to_unified_config(
        self, orchestrator, mock_db, mock_background_tasks
    ):
        """data_validator が None の場合に unified_config を使うテスト"""
        result = await orchestrator.start_historical_oi_collection(
            symbol="BTC/USDT:USDT",
            interval="1h",
            background_tasks=mock_background_tasks,
            db=mock_db,
            data_validator=None,
        )

        assert result["success"] is True
        assert result["status"] == "started"
        mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_collection_normalizes_symbol_via_mapping(
        self, orchestrator, mock_db, mock_background_tasks
    ):
        """シンボルマッピングで正規化されるテスト"""
        result = await orchestrator.start_historical_oi_collection(
            symbol="BTCUSDT",
            interval="1h",
            background_tasks=mock_background_tasks,
            db=mock_db,
            data_validator=None,
        )

        assert result["success"] is True
        # 正規化されたシンボルで背景タスクが登録される
        call_args = mock_background_tasks.add_task.call_args
        normalized_symbol = call_args.args[1]
        assert normalized_symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_start_collection_rejects_unsupported_symbol(
        self, orchestrator, mock_db, mock_background_tasks
    ):
        """サポートされていないシンボルで HTTPException が発生"""
        with pytest.raises(HTTPException) as exc_info:
            await orchestrator.start_historical_oi_collection(
                symbol="INVALID/UNKNOWN",
                interval="1h",
                background_tasks=mock_background_tasks,
                db=mock_db,
                data_validator=None,
            )
        assert exc_info.value.status_code == 500
        assert "サポートされていないシンボル" in str(exc_info.value.detail)
        mock_background_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_collection_rejects_invalid_timeframe(
        self, orchestrator, mock_db, mock_background_tasks
    ):
        """無効な時間軸で HTTPException が発生"""
        with pytest.raises(HTTPException) as exc_info:
            await orchestrator.start_historical_oi_collection(
                symbol="BTC/USDT:USDT",
                interval="99x",
                background_tasks=mock_background_tasks,
                db=mock_db,
                data_validator=None,
            )
        assert exc_info.value.status_code == 500
        assert "無効な時間軸" in str(exc_info.value.detail)
        mock_background_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_collection_propagates_validator_exception(
        self, orchestrator, mock_db, mock_data_validator, mock_background_tasks
    ):
        """data_validator が例外を投げた場合に HTTPException として伝搬される"""
        mock_data_validator.validate_symbol_and_timeframe.side_effect = ValueError(
            "validator error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await orchestrator.start_historical_oi_collection(
                symbol="BTC/USDT:USDT",
                interval="1h",
                background_tasks=mock_background_tasks,
                db=mock_db,
                data_validator=mock_data_validator,
            )
        assert exc_info.value.status_code == 500
        assert "validator error" in str(exc_info.value.detail)
        mock_background_tasks.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_collection_message_contains_normalized_symbol(
        self, orchestrator, mock_db, mock_data_validator, mock_background_tasks
    ):
        """レスポンスメッセージに正規化済みシンボルが含まれる"""
        mock_data_validator.validate_symbol_and_timeframe.return_value = "ETH/USDT:USDT"

        result = await orchestrator.start_historical_oi_collection(
            symbol="ETHUSDT",
            interval="15m",
            background_tasks=mock_background_tasks,
            db=mock_db,
            data_validator=mock_data_validator,
        )

        assert "ETH/USDT:USDT" in result["message"]
        assert "15m" in result["message"]

    @pytest.mark.asyncio
    async def test_start_collection_passes_db_to_background(
        self, orchestrator, mock_db, mock_data_validator, mock_background_tasks
    ):
        """DB セッションがバックグラウンドタスクに渡される"""
        mock_data_validator.validate_symbol_and_timeframe.return_value = "BTC/USDT:USDT"

        await orchestrator.start_historical_oi_collection(
            symbol="BTC/USDT",
            interval="1h",
            background_tasks=mock_background_tasks,
            db=mock_db,
            data_validator=mock_data_validator,
        )

        call_args = mock_background_tasks.add_task.call_args
        assert call_args.args[3] is mock_db


class TestOICollectionBackground:
    """OICollectionOrchestrator._collect_historical_oi_background のテスト"""

    @pytest.fixture
    def orchestrator(self):
        return OICollectionOrchestrator()

    @pytest.mark.asyncio
    async def test_background_collection_succeeds_and_deletes_existing(
        self, orchestrator
    ):
        """バックグラウンド収集が成功し既存データを削除する"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 5
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    return_value={"success": True, "saved_count": 100}
                )

                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

        # 削除とコミットが呼ばれた
        mock_db.query.return_value.delete.assert_called_once()
        mock_db.commit.assert_called()
        # サービス取得が正しい引数で呼ばれた
        call_kwargs = mock_service.fetch_and_save_open_interest_data.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC/USDT:USDT"
        assert call_kwargs["fetch_all"] is True
        assert call_kwargs["interval"] == "1h"

    @pytest.mark.asyncio
    async def test_background_collection_skips_delete_when_no_existing_data(
        self, orchestrator
    ):
        """既存データがない場合、削除はスキップされる"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    return_value={"success": True, "saved_count": 0}
                )

                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

        # 削除は呼ばれない
        mock_db.query.return_value.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_collection_rolls_back_on_delete_error(self, orchestrator):
        """削除時のエラーでロールバックされる"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 10
        mock_query.delete.side_effect = Exception("delete failed")
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    return_value={"success": True, "saved_count": 0}
                )

                # エラーでも collect 自体は継続する
                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

        mock_db.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_background_collection_logs_service_failure(self, orchestrator):
        """サービス呼び出しが失敗した場合でも例外を外に出さない"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    return_value={"success": False, "message": "API down"}
                )

                # 例外が発生しない
                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

    @pytest.mark.asyncio
    async def test_background_collection_closes_db_in_finally(self, orchestrator):
        """finally 節で db.close() が呼ばれる"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    side_effect=Exception("boom")
                )

                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_collection_closes_db_on_success(self, orchestrator):
        """正常終了時にも db.close() が呼ばれる"""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_db.query.return_value = mock_query

        with patch(
            "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
        ) as mock_service_cls:
            with patch(
                "app.services.data_collection.orchestration.oi_collection_orchestrator.OpenInterestRepository"
            ):
                mock_service = mock_service_cls.return_value
                mock_service.fetch_and_save_open_interest_data = AsyncMock(
                    return_value={"success": True, "saved_count": 50}
                )

                await orchestrator._collect_historical_oi_background(
                    "BTC/USDT:USDT", "1h", mock_db
                )

        mock_db.close.assert_called_once()
