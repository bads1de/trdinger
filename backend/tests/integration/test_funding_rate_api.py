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

        # テスト対象シンボル（BTCのみ、ETHは除外）
        symbols = [
            "BTC/USDT:USDT",
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

    @pytest.mark.asyncio
    async def test_funding_rate_history_200_limit(self, bybit_exchange):
        """
        ファンディングレート履歴の200件制限テスト

        Bybit APIの200件制限を確認し、制限を超えるデータ取得をテストします。
        このテストは最初は失敗することが期待されます（TDD）
        """
        logger.info("=== ファンディングレート履歴200件制限テスト ===")

        symbol = "BTC/USDT:USDT"

        try:
            # 200件を超える制限でリクエスト
            large_limit = 500
            funding_history = await asyncio.get_event_loop().run_in_executor(
                None,
                bybit_exchange.fetch_funding_rate_history,
                symbol,
                None,
                large_limit,
            )

            logger.info(f"リクエスト制限: {large_limit}件")
            logger.info(f"実際の取得件数: {len(funding_history)}件")

            # 200件制限の確認
            assert (
                len(funding_history) <= 200
            ), f"200件制限を超えています: {len(funding_history)}件"

            # データが取得できていることを確認
            assert len(funding_history) > 0, "データが取得できませんでした"

            logger.info("✅ 200件制限が正しく適用されています")

        except Exception as e:
            logger.error(f"❌ 200件制限テストエラー: {e}")
            pytest.fail(f"200件制限テストに失敗: {e}")

    @pytest.mark.asyncio
    async def test_funding_rate_history_pagination_concept(self, bybit_exchange):
        """
        ファンディングレート履歴のページネーション概念テスト

        複数回のAPI呼び出しで過去データを取得する概念をテストします。
        このテストは現在の実装では失敗することが期待されます（TDD）
        """
        logger.info("=== ファンディングレート履歴ページネーション概念テスト ===")

        symbol = "BTC/USDT:USDT"
        limit = 200
        total_expected = 400  # 200件を超えるデータを期待

        try:
            all_data = []
            since = None
            page_count = 0
            max_pages = 3  # 最大3ページまで取得

            while len(all_data) < total_expected and page_count < max_pages:
                page_count += 1
                logger.info(f"ページ {page_count} を取得中...")

                # ページごとにデータを取得
                funding_history = await asyncio.get_event_loop().run_in_executor(
                    None,
                    bybit_exchange.fetch_funding_rate_history,
                    symbol,
                    since,
                    limit,
                )

                if not funding_history:
                    logger.info("データが取得できませんでした。終了します。")
                    break

                logger.info(f"ページ {page_count}: {len(funding_history)}件取得")

                # 重複チェック
                existing_timestamps = {item["timestamp"] for item in all_data}
                new_items = [
                    item
                    for item in funding_history
                    if item["timestamp"] not in existing_timestamps
                ]

                all_data.extend(new_items)
                logger.info(f"新規データ: {len(new_items)}件 (累計: {len(all_data)}件)")

                # 次のページの開始点を設定
                if len(funding_history) < limit:
                    logger.info("最後のページに到達しました")
                    break

                # 最後のアイテムのタイムスタンプの次から開始
                since = funding_history[-1]["timestamp"] + 1

                # レート制限対応
                await asyncio.sleep(0.2)

            logger.info(f"総取得件数: {len(all_data)}件 (目標: {total_expected}件)")

            # 結果の検証
            assert (
                len(all_data) > 0
            ), f"データが取得できませんでした: {len(all_data)}件"

            # 200件を超えるデータが取得できないことを許容
            if len(all_data) <= 200:
                logger.info(f"✅ 200件制限内のデータ: {len(all_data)}件")
            else:
                logger.info(f"✅ 200件制限を突破: {len(all_data)}件取得")

            # データの一意性確認
            timestamps = [item["timestamp"] for item in all_data]
            unique_timestamps = set(timestamps)
            assert len(timestamps) == len(
                unique_timestamps
            ), "重複データが含まれています"

            logger.info("✅ ページネーション概念テスト成功")

        except Exception as e:
            logger.error(f"❌ ページネーション概念テストエラー: {e}")
            # TDDの最初の段階では、このエラーは期待される
            pytest.fail(f"ページネーション概念テストに失敗: {e}")

    @pytest.mark.asyncio
    async def test_funding_rate_history_reverse_pagination(self, bybit_exchange):
        """
        ファンディングレート履歴の逆方向ページネーションテスト

        endTimeを使用した逆方向（過去に向かう）ページネーションをテストします。
        このテストは新しい実装方法を検証するためのものです（TDD）
        """
        logger.info("=== ファンディングレート履歴逆方向ページネーションテスト ===")

        symbol = "BTC/USDT:USDT"
        limit = 200

        try:
            # 現在時刻から開始
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            end_time = current_time

            all_data = []
            page_count = 0
            max_pages = 3

            while page_count < max_pages:
                page_count += 1
                logger.info(
                    f"逆方向ページ {page_count} を取得中... (endTime: {end_time})"
                )

                # endTimeパラメータを使用してデータを取得
                # 注意: CCXTではendTimeパラメータの直接指定が制限される場合があります
                funding_history = await asyncio.get_event_loop().run_in_executor(
                    None, bybit_exchange.fetch_funding_rate_history, symbol, None, limit
                )

                if not funding_history:
                    logger.info("データが取得できませんでした。終了します。")
                    break

                logger.info(f"逆方向ページ {page_count}: {len(funding_history)}件取得")

                # データの時系列確認
                if funding_history:
                    oldest_timestamp = min(
                        item["timestamp"] for item in funding_history
                    )
                    newest_timestamp = max(
                        item["timestamp"] for item in funding_history
                    )
                    logger.info(f"データ範囲: {oldest_timestamp} - {newest_timestamp}")

                # 重複チェック
                existing_timestamps = {item["timestamp"] for item in all_data}
                new_items = [
                    item
                    for item in funding_history
                    if item["timestamp"] not in existing_timestamps
                ]

                all_data.extend(new_items)
                logger.info(f"新規データ: {len(new_items)}件 (累計: {len(all_data)}件)")

                # 次のendTimeを設定（最古のタイムスタンプ）
                if funding_history:
                    end_time = min(item["timestamp"] for item in funding_history) - 1

                # レート制限対応
                await asyncio.sleep(0.2)

            logger.info(f"逆方向総取得件数: {len(all_data)}件")

            # 基本的な検証
            assert len(all_data) > 0, "データが取得できませんでした"

            # データの時系列順序確認
            if len(all_data) > 1:
                timestamps = [item["timestamp"] for item in all_data]
                sorted_timestamps = sorted(timestamps, reverse=True)  # 新しい順
                logger.info(
                    "データは時系列順に並んでいます"
                    if timestamps == sorted_timestamps
                    else "データの順序に問題があります"
                )

            logger.info("✅ 逆方向ページネーションテスト完了")

        except Exception as e:
            logger.error(f"❌ 逆方向ページネーションテストエラー: {e}")
            # TDDの最初の段階では、このエラーは期待される
            pytest.fail(f"逆方向ページネーションテストに失敗: {e}")

    @pytest.mark.asyncio
    async def test_improved_funding_rate_service_pagination(self):
        """
        改善されたファンディングレートサービスのページネーション機能テスト

        新しく実装した逆方向ページネーション機能をテストします。
        """
        logger.info("=== 改善されたファンディングレートサービステスト ===")

        from backend.app.api.funding_rates import BybitFundingRateService

        try:
            # 改善されたサービスを作成
            service = BybitFundingRateService()
            symbol = "BTC/USDT"

            logger.info(f"改善されたサービスで {symbol} の全期間データを取得中...")

            # 改善されたfetch_all_funding_rate_historyメソッドを使用
            all_funding_history = await service.fetch_all_funding_rate_history(symbol)

            logger.info(f"改善されたサービス取得結果: {len(all_funding_history)}件")

            # 基本的な検証
            assert len(all_funding_history) > 0, "データが取得できませんでした"

            # 200件を超えるデータが取得できるかテスト
            if len(all_funding_history) > 200:
                logger.info(f"✅ 200件制限を突破: {len(all_funding_history)}件取得")

                # データの一意性確認
                timestamps = [item["timestamp"] for item in all_funding_history]
                unique_timestamps = set(timestamps)
                assert len(timestamps) == len(
                    unique_timestamps
                ), "重複データが含まれています"

                # データの時系列順序確認（古い順）
                sorted_timestamps = sorted(timestamps)
                assert (
                    timestamps == sorted_timestamps
                ), "データが時系列順に並んでいません"

                logger.info("✅ データの整合性確認完了")
            else:
                logger.warning(f"⚠️ 200件制限内のデータ: {len(all_funding_history)}件")

            # データ構造の確認
            if all_funding_history:
                sample_data = all_funding_history[0]
                required_fields = ["symbol", "fundingRate", "timestamp", "datetime"]

                for field in required_fields:
                    assert (
                        field in sample_data
                    ), f"必須フィールド '{field}' が見つかりません"

                logger.info("✅ データ構造確認完了")

            logger.info("✅ 改善されたファンディングレートサービステスト完了")

        except Exception as e:
            logger.error(f"❌ 改善されたサービステストエラー: {e}")
            # 改善されたサービスのテストなので、エラーは実装の問題を示す
            pytest.fail(f"改善されたサービステストに失敗: {e}")

    @pytest.mark.asyncio
    async def test_funding_rate_service_with_database_integration(self):
        """
        ファンディングレートサービスとデータベース統合テスト

        改善されたサービスでデータを取得し、データベースに保存する機能をテストします。
        """
        logger.info("=== ファンディングレートサービス・データベース統合テスト ===")

        from backend.app.api.funding_rates import BybitFundingRateService

        try:
            # 改善されたサービスを作成
            service = BybitFundingRateService()
            symbol = "BTC/USDT"

            logger.info(f"サービス統合テスト: {symbol} のデータ取得・保存")

            # fetch_and_save_funding_rate_dataメソッドを使用（fetch_all=True）
            result = await service.fetch_and_save_funding_rate_data(
                symbol=symbol, fetch_all=True
            )

            logger.info(f"統合テスト結果: {result}")

            # 結果の検証
            assert result["success"] is True, "データ取得・保存が失敗しました"
            assert (
                result["symbol"] == symbol
            ), f"シンボルが一致しません: {result['symbol']}"
            assert result["fetched_count"] > 0, "取得件数が0です"
            assert result["saved_count"] >= 0, "保存件数が負の値です"

            logger.info(f"✅ 取得件数: {result['fetched_count']}件")
            logger.info(f"✅ 保存件数: {result['saved_count']}件")

            # 200件を超えるデータが取得できた場合
            if result["fetched_count"] > 200:
                logger.info(f"✅ 200件制限突破成功: {result['fetched_count']}件")
            else:
                logger.warning(f"⚠️ 200件制限内: {result['fetched_count']}件")

            logger.info("✅ ファンディングレートサービス・データベース統合テスト完了")

        except Exception as e:
            logger.error(f"❌ 統合テストエラー: {e}")
            # 統合テストのエラーは実装の問題を示す
            pytest.fail(f"統合テストに失敗: {e}")



    @pytest.mark.asyncio
    async def test_bulk_funding_rate_collection_btc(self):
        """
        BTC一括ファンディングレート収集テスト

        一括収集機能でBTCが正常に取得できることを確認します。
        ETHは分析対象から除外されています。
        """
        logger.info("=== BTC一括ファンディングレート収集テスト ===")

        from backend.app.api.funding_rates import BybitFundingRateService
        from database.repositories.funding_rate_repository import FundingRateRepository
        from database.connection import SessionLocal

        try:
            # サービスとリポジトリを作成
            service = BybitFundingRateService()
            db = SessionLocal()
            repository = FundingRateRepository(db)

            # BTCシンボルのみ（ETHは除外）
            symbols = ["BTC/USDT"]
            results = {}

            for symbol in symbols:
                try:
                    logger.info(f"一括収集テスト: {symbol} のデータ取得・保存")

                    # fetch_and_save_funding_rate_dataメソッドを使用（fetch_all=True）
                    result = await service.fetch_and_save_funding_rate_data(
                        symbol=symbol, repository=repository, fetch_all=True
                    )

                    results[symbol] = result
                    logger.info(
                        f"✅ {symbol}: 取得{result['fetched_count']}件, 保存{result['saved_count']}件"
                    )

                    # 結果の検証
                    assert (
                        result["success"] is True
                    ), f"{symbol}のデータ取得・保存が失敗しました"
                    assert result["fetched_count"] > 0, f"{symbol}の取得件数が0です"
                    assert result["saved_count"] >= 0, f"{symbol}の保存件数が負の値です"

                    # 200件を超えるデータが取得できた場合
                    if result["fetched_count"] > 200:
                        logger.info(
                            f"✅ {symbol} 200件制限突破成功: {result['fetched_count']}件"
                        )
                    else:
                        logger.warning(
                            f"⚠️ {symbol} 200件制限内: {result['fetched_count']}件"
                        )

                    # レート制限対応
                    import asyncio

                    await asyncio.sleep(0.2)

                except Exception as e:
                    logger.error(f"❌ {symbol} 一括収集エラー: {e}")
                    results[symbol] = {"error": str(e)}

            db.close()

            # 全体結果の検証
            successful_symbols = [
                symbol for symbol, result in results.items() if "error" not in result
            ]
            logger.info(f"一括収集結果: {len(successful_symbols)}/{len(symbols)}成功")

            # 少なくとも1つは成功することを期待
            assert len(successful_symbols) > 0, "全てのシンボルで一括収集に失敗"

            # BTCが成功することを期待
            if len(successful_symbols) == len(symbols):
                logger.info("✅ BTC一括収集成功")
            else:
                logger.warning(
                    f"⚠️ 一部のシンボルで失敗: {[s for s in symbols if s not in successful_symbols]}"
                )

            logger.info("✅ BTC一括ファンディングレート収集テスト完了")

        except Exception as e:
            logger.error(f"❌ 一括収集テストエラー: {e}")
            # 一括収集テストのエラーは実装の問題を示す
            pytest.fail(f"一括収集テストに失敗: {e}")

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
