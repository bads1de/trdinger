"""
Bybitサービスの統合テスト

実際のサービスクラスが正しく動作することを確認します。
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.services.data_collection.bybit.open_interest_service import (
    BybitOpenInterestService,
)


class TestBybitServiceIntegration:
    """Bybitサービス統合テスト"""

    @pytest.mark.asyncio
    async def test_funding_rate_service_initialization(self):
        """ファンディングレートサービスの初期化テスト"""
        service = BybitFundingRateService()

        # 基底クラスのメソッドが利用可能であることを確認
        assert hasattr(service, "_fetch_paginated_data")
        assert hasattr(service, "_get_latest_timestamp_from_db")
        assert hasattr(service, "normalize_symbol")

        # シンボル正規化のテスト
        assert service.normalize_symbol("BTC/USDT") == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_open_interest_service_initialization(self):
        """オープンインタレストサービスの初期化テスト"""
        service = BybitOpenInterestService()

        # 基底クラスのメソッドが利用可能であることを確認
        assert hasattr(service, "_fetch_paginated_data")
        assert hasattr(service, "_get_latest_timestamp_from_db")
        assert hasattr(service, "normalize_symbol")

        # シンボル正規化のテスト
        assert service.normalize_symbol("ETH/USDT") == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_funding_rate_service_uses_common_pagination(self):
        """ファンディングレートサービスが共通ページネーションを使用することをテスト"""
        service = BybitFundingRateService()

        with (
            patch.object(service, "_get_latest_timestamp_from_db", return_value=None),
            patch.object(
                service, "_fetch_paginated_data", return_value=[]
            ) as mock_paginated,
        ):

            await service.fetch_all_funding_rate_history("BTC/USDT")

            # 共通ページネーションメソッドが呼ばれることを確認
            mock_paginated.assert_called_once()
            args, kwargs = mock_paginated.call_args

            # 正しいパラメータが渡されることを確認
            assert kwargs["fetch_func"] == service.exchange.fetch_funding_rate_history
            assert kwargs["symbol"] == "BTC/USDT:USDT"
            assert kwargs["pagination_strategy"] == "until"

    @pytest.mark.asyncio
    async def test_open_interest_service_uses_common_pagination(self):
        """オープンインタレストサービスが共通ページネーションを使用することをテスト"""
        service = BybitOpenInterestService()

        with (
            patch.object(service, "_get_latest_timestamp_from_db", return_value=None),
            patch.object(
                service, "_fetch_paginated_data", return_value=[]
            ) as mock_paginated,
        ):

            await service.fetch_all_open_interest_history("BTC/USDT", "1h")

            # 共通ページネーションメソッドが呼ばれることを確認
            mock_paginated.assert_called_once()
            args, kwargs = mock_paginated.call_args

            # 正しいパラメータが渡されることを確認
            assert kwargs["fetch_func"] == service.exchange.fetch_open_interest_history
            assert kwargs["symbol"] == "BTC/USDT:USDT"
            assert kwargs["pagination_strategy"] == "time_range"
            assert kwargs["interval"] == "1h"

    @pytest.mark.asyncio
    async def test_services_share_common_methods(self):
        """両サービスが共通メソッドを共有することをテスト"""
        funding_service = BybitFundingRateService()
        open_interest_service = BybitOpenInterestService()

        # 共通メソッドが同じ結果を返すことを確認
        assert funding_service._get_interval_milliseconds(
            "1h"
        ) == open_interest_service._get_interval_milliseconds("1h")
        assert funding_service._convert_to_api_symbol(
            "BTC/USDT:USDT"
        ) == open_interest_service._convert_to_api_symbol("BTC/USDT:USDT")

        # 共通のページデータ処理メソッドが同じ動作をすることを確認
        page_data = [{"timestamp": 1000, "value": 1}]
        all_data = []

        funding_result = funding_service._process_page_data(
            page_data, all_data, None, 1
        )
        open_interest_result = open_interest_service._process_page_data(
            page_data, all_data, None, 1
        )

        assert funding_result == open_interest_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
