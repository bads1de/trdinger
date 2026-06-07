"""
BulkDataOrchestrator の拡張テスト

``app.services.data_collection.orchestration.bulk_data_orchestrator.BulkDataOrchestrator`` の
4 つのオーケストレーション系メソッド（``execute_bulk_incremental_update``,
``start_bitcoin_full_data_collection``, ``start_bulk_historical_data_collection``,
``start_all_data_bulk_collection``）と内部の ``_collect_all_data_background`` を検証します。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.services.data_collection.orchestration import (
    bulk_data_orchestrator as orch_module,
)
from app.services.data_collection.orchestration.bulk_data_orchestrator import (
    BulkDataOrchestrator,
)


class TestBulkDataOrchestratorInit:
    """``__init__`` の挙動"""

    def test_initializes_historical_service(self) -> None:
        orch = BulkDataOrchestrator()
        assert orch.historical_service is not None


class TestExecuteBulkIncrementalUpdate:
    """``execute_bulk_incremental_update`` のテスト"""

    @pytest.mark.asyncio
    async def test_returns_api_response_with_result(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        fake_result = {"data": {"summary": "ok"}, "timeframes": ["1h"]}

        with patch.object(
            orch.historical_service,
            "collect_bulk_incremental_data",
            new=AsyncMock(return_value=fake_result),
        ):
            response = await orch.execute_bulk_incremental_update("BTC/USDT:USDT", db)

        assert response["success"] is True
        assert "一括差分更新が完了" in response["message"]
        assert response["data"] == fake_result

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_bulk_incremental_data",
            new=AsyncMock(side_effect=RuntimeError("api fail")),
        ):
            with pytest.raises(RuntimeError, match="api fail"):
                await orch.execute_bulk_incremental_update("BTC/USDT:USDT", db)


class TestStartBitcoinFullDataCollection:
    """``start_bitcoin_full_data_collection`` のテスト"""

    @pytest.mark.asyncio
    async def test_schedules_collection_for_each_timeframe(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()
        historical_orchestrator = MagicMock()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h", "1d"],
        ):
            response = await orch.start_bitcoin_full_data_collection(
                background_tasks=background_tasks,
                db=db,
                historical_orchestrator=historical_orchestrator,
            )

        assert response["success"] is True
        assert response["data"]["timeframes"] == ["1h", "1d"]
        # バックグラウンドタスクが追加されたか
        assert len(background_tasks.tasks) == 2
        for task in background_tasks.tasks:
            assert task.func == historical_orchestrator._collect_historical_background

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = MagicMock()
        historical_orchestrator = MagicMock()

        class FailingList(list):
            def __iter__(self):
                raise RuntimeError("config fail")

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=FailingList(),
        ):
            with pytest.raises(RuntimeError, match="config fail"):
                await orch.start_bitcoin_full_data_collection(
                    background_tasks=background_tasks,
                    db=db,
                    historical_orchestrator=historical_orchestrator,
                )


class TestStartBulkHistoricalDataCollection:
    """``start_bulk_historical_data_collection`` のテスト"""

    @pytest.mark.asyncio
    async def test_skips_timeframes_with_existing_data(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()
        historical_orchestrator = MagicMock()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h", "1d"],
        ):
            with patch(
                "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                # 1h にはデータがある、1d には無い
                mock_repo.get_data_count.side_effect = [100, 0]
                response = await orch.start_bulk_historical_data_collection(
                    background_tasks=background_tasks,
                    db=db,
                    force_update=False,
                    start_date="2024-01-01",
                    historical_orchestrator=historical_orchestrator,
                )

        # 1d のみタスクがスケジュールされる
        assert len(background_tasks.tasks) == 1
        assert response["data"]["collection_tasks"] == 1
        assert response["data"]["force_update"] is False
        assert response["data"]["start_date"] == "2024-01-01"
        # clear_ohlcv は呼ばれない
        mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_update_clears_existing_data(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()
        historical_orchestrator = MagicMock()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h"],
        ):
            with patch(
                "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 50
                mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.return_value = 50
                response = await orch.start_bulk_historical_data_collection(
                    background_tasks=background_tasks,
                    db=db,
                    force_update=True,
                    historical_orchestrator=historical_orchestrator,
                )

        # 強制更新の場合は既存データを削除
        mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_called_once_with(
            "BTC/USDT:USDT", "1h"
        )
        assert response["data"]["force_update"] is True
        assert "強制更新モード" in response["message"]
        assert len(background_tasks.tasks) == 1

    @pytest.mark.asyncio
    async def test_uses_default_start_date_when_none(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()
        historical_orchestrator = MagicMock()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h"],
        ):
            with patch(
                "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0
                response = await orch.start_bulk_historical_data_collection(
                    background_tasks=background_tasks,
                    db=db,
                    force_update=False,
                    start_date=None,
                    historical_orchestrator=historical_orchestrator,
                )

        # start_date=None 時はデフォルト
        assert response["data"]["start_date"] == "2020-03-25"

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = MagicMock()
        historical_orchestrator = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_data_count.side_effect = RuntimeError("db fail")
            with pytest.raises(RuntimeError, match="db fail"):
                await orch.start_bulk_historical_data_collection(
                    background_tasks=background_tasks,
                    db=db,
                    force_update=False,
                    historical_orchestrator=historical_orchestrator,
                )


class TestStartAllDataBulkCollection:
    """``start_all_data_bulk_collection`` のテスト"""

    @pytest.mark.asyncio
    async def test_schedules_only_for_missing_data(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h", "1d"],
        ):
            with patch(
                "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                # 1h データなし、1d データあり
                mock_repo.get_data_count.side_effect = [0, 100]
                response = await orch.start_all_data_bulk_collection(
                    background_tasks=background_tasks, db=db
                )

        # 1h のみタスク追加
        assert len(background_tasks.tasks) == 1
        assert response["data"]["collection_tasks"] == 1
        assert "1件のタスク" in response["message"]

    @pytest.mark.asyncio
    async def test_no_tasks_when_all_data_exists(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = BackgroundTasks()

        with patch.object(
            orch_module.unified_config.market,
            "supported_timeframes",
            new=["1h"],
        ):
            with patch(
                "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 200
                response = await orch.start_all_data_bulk_collection(
                    background_tasks=background_tasks, db=db
                )

        assert len(background_tasks.tasks) == 0
        assert response["data"]["collection_tasks"] == 0

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()
        background_tasks = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.bulk_data_orchestrator.OHLCVRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_data_count.side_effect = RuntimeError("db fail")
            with pytest.raises(RuntimeError, match="db fail"):
                await orch.start_all_data_bulk_collection(
                    background_tasks=background_tasks, db=db
                )


class TestCollectAllDataBackground:
    """``_collect_all_data_background`` のテスト"""

    @pytest.mark.asyncio
    async def test_runs_three_collections_and_closes_db(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        # OHLCV 成功
        ohlcv_result = 10
        mock_ohlcv = AsyncMock(return_value=ohlcv_result)
        with patch.object(
            orch.historical_service, "collect_historical_data", new=mock_ohlcv
        ):
            # FR 成功
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_cls:
                mock_fr = mock_fr_cls.return_value
                mock_fr.fetch_and_save_funding_rate_data = AsyncMock(
                    return_value={"success": True, "saved_count": 5}
                )
                # OI 成功
                with patch(
                    "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
                ) as mock_oi_cls:
                    mock_oi = mock_oi_cls.return_value
                    mock_oi.fetch_and_save_open_interest_data = AsyncMock(
                        return_value={"success": True, "saved_count": 3}
                    )

                    await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        # OHLCV が呼ばれた
        mock_ohlcv.assert_awaited_once()
        # FR と OI が呼ばれた
        mock_fr.fetch_and_save_funding_rate_data.assert_awaited_once()
        mock_oi.fetch_and_save_open_interest_data.assert_awaited_once()
        # db.close が finally で呼ばれる
        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_early_when_ohlcv_fails(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(return_value=None),
        ):
            await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        # OHLCV が None を返したので FR/OI は呼ばれない
        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_when_fr_fails(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(return_value=5),
        ):
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_cls:
                mock_fr = mock_fr_cls.return_value
                mock_fr.fetch_and_save_funding_rate_data = AsyncMock(
                    side_effect=RuntimeError("fr fail")
                )
                with patch(
                    "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
                ) as mock_oi_cls:
                    mock_oi = mock_oi_cls.return_value
                    mock_oi.fetch_and_save_open_interest_data = AsyncMock(
                        return_value={"success": True, "saved_count": 3}
                    )

                    # エラーが起きても継続
                    await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        mock_oi.fetch_and_save_open_interest_data.assert_awaited_once()
        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_when_oi_fails(self) -> None:
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(return_value=5),
        ):
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_cls:
                mock_fr = mock_fr_cls.return_value
                mock_fr.fetch_and_save_funding_rate_data = AsyncMock(
                    return_value={"success": True, "saved_count": 5}
                )
                with patch(
                    "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
                ) as mock_oi_cls:
                    mock_oi = mock_oi_cls.return_value
                    mock_oi.fetch_and_save_open_interest_data = AsyncMock(
                        side_effect=RuntimeError("oi fail")
                    )

                    await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_fr_returns_failure_dict(self) -> None:
        """FR が success=False を返しても例外扱いにならない"""
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(return_value=5),
        ):
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_cls:
                mock_fr = mock_fr_cls.return_value
                mock_fr.fetch_and_save_funding_rate_data = AsyncMock(
                    return_value={"success": False, "message": "rate limit"}
                )
                with patch(
                    "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
                ) as mock_oi_cls:
                    mock_oi = mock_oi_cls.return_value
                    mock_oi.fetch_and_save_open_interest_data = AsyncMock(
                        return_value={"success": True, "saved_count": 3}
                    )

                    await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_oi_returns_failure_dict(self) -> None:
        """OI が success=False を返しても例外扱いにならない"""
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(return_value=5),
        ):
            with patch(
                "app.services.data_collection.bybit.funding_rate_service.BybitFundingRateService"
            ) as mock_fr_cls:
                mock_fr = mock_fr_cls.return_value
                mock_fr.fetch_and_save_funding_rate_data = AsyncMock(
                    return_value={"success": True, "saved_count": 5}
                )
                with patch(
                    "app.services.data_collection.bybit.open_interest_service.BybitOpenInterestService"
                ) as mock_oi_cls:
                    mock_oi = mock_oi_cls.return_value
                    mock_oi.fetch_and_save_open_interest_data = AsyncMock(
                        return_value={"success": False, "message": "fail"}
                    )

                    await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_db_on_top_level_exception(self) -> None:
        """OHLCV 収集で例外が出ても finally で db.close される"""
        orch = BulkDataOrchestrator()
        db = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data",
            new=AsyncMock(side_effect=RuntimeError("ohlcv fail")),
        ):
            await orch._collect_all_data_background("BTC/USDT:USDT", "1h", db)

        db.close.assert_called_once()
