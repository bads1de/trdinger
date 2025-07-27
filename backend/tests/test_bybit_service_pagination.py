"""
Bybitサービスのページネーション共通化のテスト

提案6の実装をテストします。
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.core.services.data_collection.bybit.bybit_service import BybitService
from app.core.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.core.services.data_collection.bybit.open_interest_service import (
    BybitOpenInterestService,
)


class TestBybitService:
    """BybitService基底クラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.service = BybitService()

    def test_get_interval_milliseconds(self):
        """間隔文字列のミリ秒変換テスト"""
        assert self.service._get_interval_milliseconds("5min") == 5 * 60 * 1000
        assert self.service._get_interval_milliseconds("15min") == 15 * 60 * 1000
        assert self.service._get_interval_milliseconds("30min") == 30 * 60 * 1000
        assert self.service._get_interval_milliseconds("1h") == 60 * 60 * 1000
        assert self.service._get_interval_milliseconds("4h") == 4 * 60 * 60 * 1000
        assert self.service._get_interval_milliseconds("1d") == 24 * 60 * 60 * 1000
        # デフォルト値のテスト
        assert self.service._get_interval_milliseconds("unknown") == 60 * 60 * 1000

    def test_convert_to_api_symbol(self):
        """APIシンボル変換テスト"""
        assert self.service._convert_to_api_symbol("BTC/USDT:USDT") == "BTCUSDT"
        assert self.service._convert_to_api_symbol("ETH/USDT:USDT") == "ETHUSDT"
        assert self.service._convert_to_api_symbol("BTC/USD:USD") == "BTCUSD"

    def test_process_page_data_no_duplicates(self):
        """重複なしのページデータ処理テスト"""
        page_data = [
            {"timestamp": 1000, "value": 1},
            {"timestamp": 2000, "value": 2},
        ]
        all_data = []

        result = self.service._process_page_data(page_data, all_data, None, 1)

        assert result == page_data
        assert len(result) == 2

    def test_process_page_data_with_duplicates(self):
        """重複ありのページデータ処理テスト"""
        page_data = [
            {"timestamp": 1000, "value": 1},
            {"timestamp": 2000, "value": 2},
            {"timestamp": 3000, "value": 3},
        ]
        all_data = [
            {"timestamp": 2000, "value": 2},  # 重複データ
        ]

        result = self.service._process_page_data(page_data, all_data, None, 1)

        # 重複を除いた2件が返される
        assert len(result) == 2
        assert {"timestamp": 1000, "value": 1} in result
        assert {"timestamp": 3000, "value": 3} in result
        assert {"timestamp": 2000, "value": 2} not in result

    def test_process_page_data_incremental_update(self):
        """差分更新のページデータ処理テスト"""
        page_data = [
            {"timestamp": 1000, "value": 1},  # 既存より古い
            {"timestamp": 2000, "value": 2},  # 既存より新しい
            {"timestamp": 3000, "value": 3},  # 既存より新しい
        ]
        all_data = []
        latest_existing_timestamp = 1500  # 1500より古いデータのみ取得

        result = self.service._process_page_data(
            page_data, all_data, latest_existing_timestamp, 1
        )

        # 1500より古いデータのみ返される
        assert len(result) == 1
        assert result[0]["timestamp"] == 1000

    def test_process_page_data_incremental_update_no_new_data(self):
        """差分更新で新規データなしのテスト"""
        page_data = [
            {"timestamp": 2000, "value": 2},  # 既存より新しい
            {"timestamp": 3000, "value": 3},  # 既存より新しい
        ]
        all_data = []
        latest_existing_timestamp = 1500  # 1500より古いデータのみ取得

        result = self.service._process_page_data(
            page_data, all_data, latest_existing_timestamp, 1
        )

        # 新規データなしの場合はNoneが返される
        assert result is None


class TestBybitFundingRateService:
    """BybitFundingRateServiceのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.service = BybitFundingRateService()

    @pytest.mark.asyncio
    async def test_fetch_all_funding_rate_history_uses_until_strategy(self):
        """全期間ファンディングレート取得でuntil戦略を使用することをテスト"""
        with (
            patch.object(
                self.service, "_get_latest_timestamp_from_db", return_value=None
            ),
            patch.object(
                self.service, "_fetch_paginated_data", return_value=[]
            ) as mock_paginated,
        ):

            await self.service.fetch_all_funding_rate_history("BTC/USDT")

            # _fetch_paginated_dataが正しいパラメータで呼ばれることを確認
            mock_paginated.assert_called_once()
            args, kwargs = mock_paginated.call_args
            assert kwargs["pagination_strategy"] == "until"
            assert kwargs["symbol"] == "BTC/USDT:USDT"


class TestBybitOpenInterestService:
    """BybitOpenInterestServiceのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.service = BybitOpenInterestService()

    @pytest.mark.asyncio
    async def test_fetch_all_open_interest_history_uses_time_range_strategy(self):
        """全期間オープンインタレスト取得でtime_range戦略を使用することをテスト"""
        with (
            patch.object(
                self.service, "_get_latest_timestamp_from_db", return_value=None
            ),
            patch.object(
                self.service, "_fetch_paginated_data", return_value=[]
            ) as mock_paginated,
        ):

            await self.service.fetch_all_open_interest_history("BTC/USDT", "1h")

            # _fetch_paginated_dataが正しいパラメータで呼ばれることを確認
            mock_paginated.assert_called_once()
            args, kwargs = mock_paginated.call_args
            assert kwargs["pagination_strategy"] == "time_range"
            assert kwargs["symbol"] == "BTC/USDT:USDT"
            assert kwargs["interval"] == "1h"


class TestPaginationIntegration:
    """ページネーション統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.service = BybitService()

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_invalid_strategy(self):
        """無効なページネーション戦略のテスト"""
        with pytest.raises(ValueError, match="未対応のページネーション戦略"):
            await self.service._fetch_paginated_data(
                fetch_func=Mock(),
                symbol="BTC/USDT:USDT",
                pagination_strategy="invalid_strategy",
            )

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_until_strategy(self):
        """until戦略のページネーションテスト"""
        mock_fetch_func = AsyncMock()
        mock_fetch_func.return_value = [
            {"timestamp": 1000, "value": 1},
            {"timestamp": 2000, "value": 2},
        ]

        with patch.object(
            self.service, "_handle_ccxt_errors", side_effect=mock_fetch_func
        ):
            result = await self.service._fetch_paginated_data(
                fetch_func=mock_fetch_func,
                symbol="BTC/USDT:USDT",
                page_limit=2,
                max_pages=1,
                pagination_strategy="until",
            )

            assert len(result) == 2
            assert result[0]["timestamp"] == 2000  # 新しい順にソート
            assert result[1]["timestamp"] == 1000

    @pytest.mark.asyncio
    async def test_fetch_paginated_data_time_range_strategy(self):
        """time_range戦略のページネーションテスト"""
        mock_fetch_func = AsyncMock()
        mock_fetch_func.return_value = [
            {"timestamp": 1000, "value": 1},
            {"timestamp": 2000, "value": 2},
        ]

        with patch.object(
            self.service, "_handle_ccxt_errors", side_effect=mock_fetch_func
        ):
            result = await self.service._fetch_paginated_data(
                fetch_func=mock_fetch_func,
                symbol="BTC/USDT:USDT",
                page_limit=2,
                max_pages=1,
                pagination_strategy="time_range",
                interval="1h",
            )

            assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
