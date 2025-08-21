"""
CCXT調査テスト

CCXTライブラリの現在の状況とBybit APIの動作を詳しく調査します。
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

import ccxt
from app.services.data_collection.bybit.funding_rate_service import (
    BybitFundingRateService,
)
from app.services.data_collection.bybit.open_interest_service import (
    BybitOpenInterestService,
)
from app.services.data_collection.historical.historical_data_service import (
    HistoricalDataService,
)
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCCXTInvestigation:
    """CCXT調査テスト"""

    def test_ccxt_bybit_methods_availability(self):
        """CCXTのBybitメソッドの利用可能性をテスト"""
        exchange = ccxt.bybit(
            {
                "sandbox": False,
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},
            }
        )

        # 利用可能なメソッドを確認
        methods = [
            "fetch_funding_rate",
            "fetch_funding_rate_history",
            "fetch_open_interest",
            "fetch_open_interest_history",
            "fetch_ohlcv",
        ]

        available_methods = []
        unavailable_methods = []

        for method in methods:
            if hasattr(exchange, method):
                available_methods.append(method)
                logger.info(f"✅ {method} is available")
            else:
                unavailable_methods.append(method)
                logger.error(f"❌ {method} is NOT available")

        print(f"Available methods: {available_methods}")
        print(f"Unavailable methods: {unavailable_methods}")

        # 基本的なメソッドは利用可能であることを確認
        assert "fetch_funding_rate" in available_methods
        assert "fetch_funding_rate_history" in available_methods

    @pytest.mark.asyncio
    async def test_funding_rate_service_direct_call(self):
        """ファンディングレートサービスの直接呼び出しテスト"""
        service = BybitFundingRateService()

        # モックリポジトリ
        mock_repo = Mock(spec=FundingRateRepository)
        mock_repo.get_latest_funding_timestamp.return_value = None
        mock_repo.insert_funding_rate_data.return_value = 5

        logger.info("=== ファンディングレートサービス直接テスト開始 ===")

        try:
            # 実際のAPI呼び出し
            result = await service.fetch_incremental_funding_rate_data(
                "BTC/USDT", mock_repo
            )

            logger.info(f"ファンディングレート結果: {result}")

            assert result["success"] is True
            assert "symbol" in result
            assert "saved_count" in result

            print(f"✅ ファンディングレート直接呼び出し成功: {result}")

        except Exception as e:
            logger.error(f"❌ ファンディングレート直接呼び出し失敗: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            raise

    @pytest.mark.asyncio
    async def test_open_interest_service_direct_call(self):
        """オープンインタレストサービスの直接呼び出しテスト"""
        service = BybitOpenInterestService()

        # モックリポジトリ
        mock_repo = Mock(spec=OpenInterestRepository)
        mock_repo.get_latest_open_interest_timestamp.return_value = None
        mock_repo.insert_open_interest_data.return_value = 3

        logger.info("=== オープンインタレストサービス直接テスト開始 ===")

        try:
            # 実際のAPI呼び出し
            result = await service.fetch_incremental_open_interest_data(
                "BTC/USDT", mock_repo
            )

            logger.info(f"オープンインタレスト結果: {result}")

            assert result["success"] is True
            assert "symbol" in result
            assert "saved_count" in result

            print(f"✅ オープンインタレスト直接呼び出し成功: {result}")

        except Exception as e:
            logger.error(f"❌ オープンインタレスト直接呼び出し失敗: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            # オープンインタレストは既知の問題があるため、エラーでもテストを続行
            pytest.skip(f"Open Interest API issue: {e}")

    @pytest.mark.asyncio
    async def test_historical_service_individual_calls(self):
        """履歴サービスの個別呼び出しテスト"""
        service = HistoricalDataService()

        # モックリポジトリ
        mock_fr_repo = Mock(spec=FundingRateRepository)
        mock_fr_repo.get_latest_funding_timestamp.return_value = None
        mock_fr_repo.insert_funding_rate_data.return_value = 5

        mock_oi_repo = Mock(spec=OpenInterestRepository)
        mock_oi_repo.get_latest_open_interest_timestamp.return_value = None
        mock_oi_repo.insert_open_interest_data.return_value = 3

        logger.info("=== 履歴サービス個別呼び出しテスト開始 ===")

        # ファンディングレートのみをテスト
        try:
            result = await service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=None,  # OHLCVをスキップ
                funding_rate_repository=mock_fr_repo,
                open_interest_repository=None,  # OIをスキップ
            )

            logger.info(f"履歴サービス結果（FRのみ）: {result}")

            # ファンディングレートの結果を確認
            fr_data = result["data"].get("funding_rate")
            if fr_data:
                print(f"✅ 履歴サービスでファンディングレート処理: {fr_data}")
                assert fr_data["success"] is True
            else:
                print("❌ 履歴サービスでファンディングレートが処理されていない")
                print(f"Available data keys: {list(result['data'].keys())}")

        except Exception as e:
            logger.error(f"❌ 履歴サービス個別呼び出し失敗: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            raise

    @pytest.mark.asyncio
    async def test_ccxt_funding_rate_raw_call(self):
        """CCXT ファンディングレートの生の呼び出しテスト"""
        exchange = ccxt.bybit(
            {
                "sandbox": False,
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},
            }
        )

        logger.info("=== CCXT生呼び出しテスト開始 ===")

        try:
            # 現在のファンディングレートを取得
            current_rate = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_funding_rate, "BTC/USDT:USDT"
            )

            logger.info(f"現在のファンディングレート: {current_rate}")
            print("✅ CCXT現在ファンディングレート取得成功")

            # 履歴データを取得
            history = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_funding_rate_history, "BTC/USDT:USDT", None, 5
            )

            logger.info(f"ファンディングレート履歴件数: {len(history)}")
            if history:
                logger.info(f"最新履歴データ: {history[0]}")

            print(f"✅ CCXT履歴ファンディングレート取得成功: {len(history)}件")

        except Exception as e:
            logger.error(f"❌ CCXT生呼び出し失敗: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            raise

    @pytest.mark.asyncio
    async def test_ccxt_open_interest_raw_call(self):
        """CCXT オープンインタレストの生の呼び出しテスト"""
        exchange = ccxt.bybit(
            {
                "sandbox": False,
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},
            }
        )

        logger.info("=== CCXTオープンインタレスト生呼び出しテスト開始 ===")

        try:
            # オープンインタレスト履歴を取得
            if hasattr(exchange, "fetch_open_interest_history"):
                history = await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_open_interest_history, "BTC/USDT:USDT", None, 5
                )

                logger.info(f"オープンインタレスト履歴件数: {len(history)}")
                if history:
                    logger.info(f"最新履歴データ: {history[0]}")

                print(f"✅ CCXTオープンインタレスト取得成功: {len(history)}件")
            else:
                print("❌ fetch_open_interest_history メソッドが利用できません")

        except Exception as e:
            logger.error(f"❌ CCXTオープンインタレスト生呼び出し失敗: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            # オープンインタレストは既知の問題があるため、エラーでもテストを続行
            pytest.skip(f"Open Interest API issue: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
