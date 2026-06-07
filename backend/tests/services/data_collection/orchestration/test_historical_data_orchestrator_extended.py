"""
HistoricalDataOrchestrator の拡張テスト

``start_historical_data_collection`` の全分岐と
``_collect_historical_background`` を検証します。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.services.data_collection.orchestration.historical_data_orchestrator import (
    HistoricalDataOrchestrator,
)


def _make_validator(
    normalized_symbol: str = "BTC/USDT:USDT",
) -> MagicMock:
    validator = MagicMock()
    validator.validate_symbol_and_timeframe.return_value = normalized_symbol
    return validator


class TestHistoricalDataOrchestratorInit:
    """``__init__`` のテスト"""

    def test_initializes_historical_service(self) -> None:
        orch = HistoricalDataOrchestrator()
        assert orch.historical_service is not None


class TestStartHistoricalDataCollection:
    """``start_historical_data_collection`` のテスト"""

    @pytest.mark.asyncio
    async def test_returns_exists_when_data_present(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()
        validator = _make_validator()

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_data_count.return_value = 100
            response = await orch.start_historical_data_collection(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=bg,
                db=db,
                data_validator=validator,
            )

        assert response["success"] is True
        assert response["status"] == "exists"
        # バックグラウンドタスク追加されない
        assert len(bg.tasks) == 0

    @pytest.mark.asyncio
    async def test_uses_data_validator_for_normalization(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()
        validator = _make_validator(normalized_symbol="XBT/USDT")

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_data_count.return_value = 0
            response = await orch.start_historical_data_collection(
                symbol="BTC",
                timeframe="1h",
                background_tasks=bg,
                db=db,
                data_validator=validator,
            )

        validator.validate_symbol_and_timeframe.assert_called_once_with("BTC", "1h")
        # バックグラウンドタスク追加
        assert len(bg.tasks) == 1
        assert response["status"] == "started"

    @pytest.mark.asyncio
    async def test_force_update_clears_existing_data(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()
        validator = _make_validator()

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            # 既存データあり + force_update
            mock_repo.get_data_count.return_value = 50
            mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.return_value = 50
            response = await orch.start_historical_data_collection(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                background_tasks=bg,
                db=db,
                force_update=True,
                data_validator=validator,
            )

        mock_repo.clear_ohlcv_data_by_symbol_and_timeframe.assert_called_once_with(
            "BTC/USDT:USDT", "1h"
        )
        assert response["status"] == "started"
        assert "強制更新モード" in response["message"]
        assert len(bg.tasks) == 1

    @pytest.mark.asyncio
    async def test_default_validator_rejects_unsupported_symbol(self) -> None:
        """data_validator=None でサポート外シンボルは ValueError"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = MagicMock()

        with patch("app.config.unified_config.unified_config.market") as mock_market:
            mock_market.symbol_mapping = {"BTC/USDT:USDT": "BTC/USDT:USDT"}
            mock_market.supported_symbols = ["ETH/USDT:USDT"]
            mock_market.supported_timeframes = ["1h"]

            with pytest.raises(ValueError, match="サポートされていないシンボル"):
                await orch.start_historical_data_collection(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    background_tasks=bg,
                    db=db,
                )

    @pytest.mark.asyncio
    async def test_default_validator_rejects_unsupported_timeframe(self) -> None:
        """data_validator=None でサポート外時間軸は ValueError"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = MagicMock()

        with patch("app.config.unified_config.unified_config.market") as mock_market:
            mock_market.symbol_mapping = {"BTC/USDT:USDT": "BTC/USDT:USDT"}
            mock_market.supported_symbols = ["BTC/USDT:USDT"]
            mock_market.supported_timeframes = ["1h"]

            with pytest.raises(ValueError, match="無効な時間軸"):
                await orch.start_historical_data_collection(
                    symbol="BTC/USDT:USDT",
                    timeframe="99x",
                    background_tasks=bg,
                    db=db,
                )

    @pytest.mark.asyncio
    async def test_default_validator_passes_with_supported_values(self) -> None:
        """data_validator=None でもシンボルが supported なら通る"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()

        with patch("app.config.unified_config.unified_config.market") as mock_market:
            mock_market.symbol_mapping = {"BTC/USDT:USDT": "BTC/USDT:USDT"}
            mock_market.supported_symbols = ["BTC/USDT:USDT"]
            mock_market.supported_timeframes = ["1h"]

            with patch(
                "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0
                response = await orch.start_historical_data_collection(
                    symbol="BTC/USDT:USDT",
                    timeframe="1h",
                    background_tasks=bg,
                    db=db,
                )

        assert response["status"] == "started"
        assert len(bg.tasks) == 1

    @pytest.mark.asyncio
    async def test_uses_ohlcv_repository_class_override(self) -> None:
        """ohlcv_repository_class 引数で repository class を差し替え"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()
        validator = _make_validator()
        custom_class = MagicMock()
        # custom_class(db) で返るインスタンスの get_data_count を設定
        custom_class.return_value.get_data_count.return_value = 0

        await orch.start_historical_data_collection(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            background_tasks=bg,
            db=db,
            data_validator=validator,
            ohlcv_repository_class=custom_class,
        )

        # カスタムクラスが呼ばれる
        custom_class.assert_called_once_with(db)
        custom_class.return_value.get_data_count.assert_called_once_with(
            "BTC/USDT:USDT", "1h"
        )

    @pytest.mark.asyncio
    async def test_default_validator_normalizes_symbol(self) -> None:
        """data_validator=None で symbol_mapping による正規化が効く"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        bg = BackgroundTasks()

        with patch("app.config.unified_config.unified_config.market") as mock_market:
            mock_market.symbol_mapping = {"BTC": "BTC/USDT:USDT"}
            mock_market.supported_symbols = ["BTC/USDT:USDT"]
            mock_market.supported_timeframes = ["1h"]

            with patch(
                "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
            ) as mock_repo_cls:
                mock_repo = mock_repo_cls.return_value
                mock_repo.get_data_count.return_value = 0
                await orch.start_historical_data_collection(
                    symbol="BTC",
                    timeframe="1h",
                    background_tasks=bg,
                    db=db,
                )
                # 正規化されたシンボルで count
                mock_repo.get_data_count.assert_called_once_with("BTC/USDT:USDT", "1h")


class TestCollectHistoricalBackground:
    """``_collect_historical_background`` のテスト"""

    @pytest.mark.asyncio
    async def test_runs_collection_and_succeeds(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ):
            with patch.object(
                orch.historical_service,
                "collect_historical_data_with_start_date",
                new=AsyncMock(return_value=10),
            ):
                await orch._collect_historical_background("BTC/USDT:USDT", "1h", db)

        # 何も raise しなければ OK

    @pytest.mark.asyncio
    async def test_logs_failure_when_result_none(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ):
            with patch.object(
                orch.historical_service,
                "collect_historical_data_with_start_date",
                new=AsyncMock(return_value=None),
            ):
                await orch._collect_historical_background("BTC/USDT:USDT", "1h", db)

        # result is None のときは logger.error 呼ばれる
        # 例外は投げない

    @pytest.mark.asyncio
    async def test_logs_error_on_exception(self) -> None:
        orch = HistoricalDataOrchestrator()
        db = MagicMock()

        with patch(
            "app.services.data_collection.orchestration.historical_data_orchestrator.OHLCVRepository"
        ):
            with patch.object(
                orch.historical_service,
                "collect_historical_data_with_start_date",
                new=AsyncMock(side_effect=RuntimeError("api fail")),
            ):
                # 例外は raise しない、ログだけ
                await orch._collect_historical_background("BTC/USDT:USDT", "1h", db)

    @pytest.mark.asyncio
    async def test_uses_ohlcv_repository_class_override(self) -> None:
        """ohlcv_repository_class 引数の挙動"""
        orch = HistoricalDataOrchestrator()
        db = MagicMock()
        custom_class = MagicMock()

        with patch.object(
            orch.historical_service,
            "collect_historical_data_with_start_date",
            new=AsyncMock(return_value=5),
        ):
            await orch._collect_historical_background(
                "BTC/USDT:USDT", "1h", db, ohlcv_repository_class=custom_class
            )

        custom_class.assert_called_once_with(db)
