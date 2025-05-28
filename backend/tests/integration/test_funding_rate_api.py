"""
ファンディングレートAPI調査テスト

CCXTライブラリを使用してBybitのファンディングレート取得方法を調査し、
データ形式を確認するためのテストです。

TDD実装の第一段階として、実際のAPI呼び出しを行い、
期待される動作を定義します。
"""

import pytest
import asyncio
import ccxt
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBybitFundingRateAPI:
    """Bybitファンディングレート取得APIのテスト"""

    @pytest.fixture
    def bybit_exchange(self):
        """Bybit取引所インスタンスを作成"""
        return ccxt.bybit(
            {
                "sandbox": False,  # 本番環境を使用（読み取り専用）
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # 無期限契約市場を使用
                },
            }
        )

    @pytest.mark.asyncio
    async def test_fetch_current_funding_rate(self, bybit_exchange):
        """
        現在のファンディングレート取得テスト

        このテストは最初は失敗することが期待されます（TDD）
        """
        logger.info("=== 現在のファンディングレート取得テスト ===")

        # テスト対象シンボル（無期限契約）
        symbol = "BTC/USDT:USDT"

        try:
            # 現在のファンディングレートを取得
            funding_rate = await asyncio.get_event_loop().run_in_executor(
                None, bybit_exchange.fetch_funding_rate, symbol
            )

            logger.info(f"ファンディングレート取得成功: {symbol}")
            logger.info(f"データ: {funding_rate}")

            # データ構造の検証
            assert funding_rate is not None
            assert "symbol" in funding_rate
            assert "fundingRate" in funding_rate
            assert "timestamp" in funding_rate
            assert "datetime" in funding_rate

            # データ型の検証
            assert isinstance(funding_rate["symbol"], str)
            assert isinstance(funding_rate["fundingRate"], (int, float))
            assert isinstance(funding_rate["timestamp"], (int, type(None)))

            logger.info("✅ 現在のファンディングレート取得テスト成功")

        except Exception as e:
            logger.error(f"❌ 現在のファンディングレート取得エラー: {e}")
            # TDDの最初の段階では、このエラーは期待される
            pytest.fail(f"ファンディングレート取得に失敗: {e}")

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_history(self, bybit_exchange):
        """
        ファンディングレート履歴取得テスト

        このテストは最初は失敗することが期待されます（TDD）
        """
        logger.info("=== ファンディングレート履歴取得テスト ===")

        # テスト対象シンボル（無期限契約）
        symbol = "BTC/USDT:USDT"

        # 過去7日間のデータを取得
        since = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        limit = 100

        try:
            # ファンディングレート履歴を取得
            funding_history = await asyncio.get_event_loop().run_in_executor(
                None, bybit_exchange.fetch_funding_rate_history, symbol, since, limit
            )

            logger.info(f"ファンディングレート履歴取得成功: {symbol}")
            logger.info(f"取得件数: {len(funding_history)}")

            if funding_history:
                logger.info(f"最新データ: {funding_history[-1]}")

                # データ構造の検証
                for rate_data in funding_history[:3]:  # 最初の3件をチェック
                    assert "symbol" in rate_data
                    assert "fundingRate" in rate_data
                    assert "timestamp" in rate_data
                    assert "datetime" in rate_data

                    # データ型の検証
                    assert isinstance(rate_data["symbol"], str)
                    assert isinstance(rate_data["fundingRate"], (int, float))
                    assert isinstance(rate_data["timestamp"], (int, type(None)))

                logger.info("✅ ファンディングレート履歴取得テスト成功")
            else:
                logger.warning("⚠️ ファンディングレート履歴データが空です")

        except Exception as e:
            logger.error(f"❌ ファンディングレート履歴取得エラー: {e}")
            # TDDの最初の段階では、このエラーは期待される
            pytest.fail(f"ファンディングレート履歴取得に失敗: {e}")

    @pytest.mark.asyncio
    async def test_multiple_symbols_funding_rates(self, bybit_exchange):
        """
        複数シンボルのファンディングレート取得テスト

        一括取得機能のためのテスト
        """
        logger.info("=== 複数シンボルファンディングレート取得テスト ===")

        # テスト対象シンボル（主要な無期限契約）
        symbols = [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "ADA/USDT:USDT",
            "DOT/USDT:USDT",
        ]

        results = {}

        for symbol in symbols:
            try:
                # 各シンボルのファンディングレートを取得
                funding_rate = await asyncio.get_event_loop().run_in_executor(
                    None, bybit_exchange.fetch_funding_rate, symbol
                )

                results[symbol] = funding_rate
                logger.info(f"✅ {symbol}: {funding_rate.get('fundingRate', 'N/A')}")

                # レート制限対応
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ {symbol} ファンディングレート取得エラー: {e}")
                results[symbol] = None

        # 結果の検証
        successful_results = {k: v for k, v in results.items() if v is not None}
        logger.info(f"成功した取得: {len(successful_results)}/{len(symbols)}")

        # 少なくとも1つは成功することを期待
        assert (
            len(successful_results) > 0
        ), "全てのシンボルでファンディングレート取得に失敗"

        logger.info("✅ 複数シンボルファンディングレート取得テスト完了")

    def test_funding_rate_data_structure_validation(self):
        """
        ファンディングレートデータ構造の検証テスト

        期待されるデータ構造を定義
        """
        logger.info("=== ファンディングレートデータ構造検証テスト ===")

        # 期待されるデータ構造
        expected_structure = {
            "symbol": str,
            "fundingRate": (int, float),
            "timestamp": (int, type(None)),
            "datetime": (str, type(None)),
            "info": dict,  # 取引所固有の追加情報
        }

        logger.info("期待されるファンディングレートデータ構造:")
        for field, field_type in expected_structure.items():
            logger.info(f"  {field}: {field_type}")

        # この構造は後でサービス実装時に使用される
        assert True  # 構造定義のみ

        logger.info("✅ データ構造定義完了")


if __name__ == "__main__":
    # 直接実行時のテスト
    pytest.main([__file__, "-v", "-s"])
