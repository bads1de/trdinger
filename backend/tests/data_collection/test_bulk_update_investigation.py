"""
一括差分更新調査テスト

一括差分更新処理で何が起こっているかを詳しく調査します。
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestBulkUpdateInvestigation:
    """一括差分更新調査テスト"""

    @pytest.fixture
    def mock_db_session(self):
        """モックDBセッションを作成"""
        return Mock(spec=Session)

    @pytest.fixture
    def orchestration_service(self):
        """データ収集統合サービスのインスタンスを作成"""
        return DataCollectionOrchestrationService()

    @pytest.fixture
    def mock_repositories(self):
        """モックリポジトリを作成"""
        # OHLCVリポジトリ（既存データがあることにする）
        ohlcv_repo = Mock(spec=OHLCVRepository)
        ohlcv_repo.get_data_count.return_value = 100

        # ファンディングレートリポジトリ
        funding_rate_repo = Mock(spec=FundingRateRepository)
        funding_rate_repo.get_latest_funding_timestamp.return_value = None
        funding_rate_repo.insert_funding_rate_data.return_value = 10

        # オープンインタレストリポジトリ
        open_interest_repo = Mock(spec=OpenInterestRepository)
        open_interest_repo.get_latest_open_interest_timestamp.return_value = None
        open_interest_repo.insert_open_interest_data.return_value = 5

        return {
            "ohlcv": ohlcv_repo,
            "funding_rate": funding_rate_repo,
            "open_interest": open_interest_repo,
        }

    @pytest.mark.asyncio
    async def test_bulk_update_with_detailed_logging(
        self, orchestration_service, mock_db_session, mock_repositories
    ):
        """詳細ログ付きで一括差分更新をテスト"""

        logger.info("=== 一括差分更新詳細調査開始 ===")

        # リポジトリクラスをモックに置き換え
        with patch(
            "app.services.data_collection.orchestration.data_collection_orchestration_service.OHLCVRepository"
        ) as mock_ohlcv_class:
            with patch(
                "app.services.data_collection.orchestration.data_collection_orchestration_service.FundingRateRepository"
            ) as mock_funding_class:
                with patch(
                    "app.services.data_collection.orchestration.data_collection_orchestration_service.OpenInterestRepository"
                ) as mock_oi_class:

                    # モックリポジトリを設定
                    mock_ohlcv_class.return_value = mock_repositories["ohlcv"]
                    mock_funding_class.return_value = mock_repositories["funding_rate"]
                    mock_oi_class.return_value = mock_repositories["open_interest"]

                    logger.info("モックリポジトリ設定完了")

                    try:
                        # 一括差分更新を実行
                        result = (
                            await orchestration_service.execute_bulk_incremental_update(
                                symbol="BTC/USDT:USDT", db=mock_db_session
                            )
                        )

                        logger.info(f"一括差分更新結果: {result}")

                        # 結果の詳細分析
                        assert result["success"] is True
                        assert "data" in result

                        data = result["data"]

                        # 各データタイプの結果を詳しく確認
                        print("\n=== 結果詳細分析 ===")

                        # OHLCV
                        if "ohlcv" in data:
                            ohlcv_result = data["ohlcv"]
                            print(f"OHLCV結果: {ohlcv_result}")
                            if ohlcv_result.get("success"):
                                print("✅ OHLCV処理成功")
                            else:
                                print(
                                    f"❌ OHLCV処理失敗: {ohlcv_result.get('error', 'Unknown error')}"
                                )

                        # ファンディングレート
                        if "funding_rate" in data:
                            fr_result = data["funding_rate"]
                            print(f"ファンディングレート結果: {fr_result}")
                            if fr_result.get("success"):
                                print(
                                    f"✅ ファンディングレート処理成功: {fr_result.get('saved_count', 0)}件"
                                )
                                assert (
                                    fr_result["saved_count"] > 0
                                ), "ファンディングレートのデータが保存されていません"
                            else:
                                print(
                                    f"❌ ファンディングレート処理失敗: {fr_result.get('error', 'Unknown error')}"
                                )
                                pytest.fail(
                                    f"ファンディングレート処理が失敗しました: {fr_result.get('error')}"
                                )
                        else:
                            print("❌ ファンディングレートの結果がありません")
                            pytest.fail("ファンディングレートが処理されていません")

                        # オープンインタレスト
                        if "open_interest" in data:
                            oi_result = data["open_interest"]
                            print(f"オープンインタレスト結果: {oi_result}")
                            if oi_result.get("success"):
                                print(
                                    f"✅ オープンインタレスト処理成功: {oi_result.get('saved_count', 0)}件"
                                )
                            else:
                                print(
                                    f"⚠️ オープンインタレスト処理失敗（既知の問題）: {oi_result.get('error', 'Unknown error')}"
                                )
                        else:
                            print("❌ オープンインタレストの結果がありません")

                        # Fear & Greed Index
                        if "fear_greed_index" in data:
                            fg_result = data["fear_greed_index"]
                            print(f"Fear & Greed Index結果: {fg_result}")
                            if fg_result.get("success"):
                                print(
                                    f"✅ Fear & Greed Index処理成功: {fg_result.get('inserted_count', 0)}件"
                                )
                            else:
                                print(
                                    f"❌ Fear & Greed Index処理失敗: {fg_result.get('error', 'Unknown error')}"
                                )

                        print(
                            f"\n総保存件数: {result.get('data', {}).get('total_saved_count', 0)}"
                        )

                    except Exception as e:
                        logger.error(f"一括差分更新中にエラー: {e}")
                        print(f"Error details: {type(e).__name__}: {e}")
                        raise

    @pytest.mark.asyncio
    async def test_historical_service_bulk_call_direct(self, mock_repositories):
        """履歴サービスの一括呼び出しを直接テスト"""
        from app.services.data_collection.historical.historical_data_service import (
            HistoricalDataService,
        )

        service = HistoricalDataService()

        logger.info("=== 履歴サービス一括呼び出し直接テスト開始 ===")

        try:
            # 一括差分データ収集を直接呼び出し
            result = await service.collect_bulk_incremental_data(
                symbol="BTC/USDT:USDT",
                timeframe="1h",
                ohlcv_repository=mock_repositories["ohlcv"],
                funding_rate_repository=mock_repositories["funding_rate"],
                open_interest_repository=mock_repositories["open_interest"],
            )

            logger.info(f"履歴サービス一括結果: {result}")

            print("\n=== 履歴サービス結果詳細分析 ===")

            # 結果の詳細分析
            data = result.get("data", {})

            # ファンディングレート
            if "funding_rate" in data:
                fr_result = data["funding_rate"]
                print(f"ファンディングレート結果: {fr_result}")
                if fr_result.get("success"):
                    print(
                        f"✅ 履歴サービスでファンディングレート処理成功: {fr_result.get('saved_count', 0)}件"
                    )
                else:
                    print(
                        f"❌ 履歴サービスでファンディングレート処理失敗: {fr_result.get('error', 'Unknown error')}"
                    )
            else:
                print("❌ 履歴サービスでファンディングレートが処理されていません")
                print(f"Available data keys: {list(data.keys())}")

            # オープンインタレスト
            if "open_interest" in data:
                oi_result = data["open_interest"]
                print(f"オープンインタレスト結果: {oi_result}")
                if oi_result.get("success"):
                    print(
                        f"✅ 履歴サービスでオープンインタレスト処理成功: {oi_result.get('saved_count', 0)}件"
                    )
                else:
                    print(
                        f"⚠️ 履歴サービスでオープンインタレスト処理失敗: {oi_result.get('error', 'Unknown error')}"
                    )
            else:
                print("❌ 履歴サービスでオープンインタレストが処理されていません")

            # エラーがあるかチェック
            errors = result.get("errors", [])
            if errors:
                print(f"⚠️ エラーリスト: {errors}")

        except Exception as e:
            logger.error(f"履歴サービス一括呼び出し中にエラー: {e}")
            print(f"Error details: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
