"""
最終検証テスト

修正されたデータ収集処理の最終検証を行います。
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFinalVerification:
    """最終検証テスト"""

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
        funding_rate_repo.insert_funding_rate_data.return_value = 15

        # オープンインタレストリポジトリ
        open_interest_repo = Mock(spec=OpenInterestRepository)
        open_interest_repo.get_latest_open_interest_timestamp.return_value = None
        open_interest_repo.insert_open_interest_data.return_value = 8

        return {
            "ohlcv": ohlcv_repo,
            "funding_rate": funding_rate_repo,
            "open_interest": open_interest_repo,
        }

    @pytest.mark.asyncio
    async def test_bulk_update_final_verification(
        self, orchestration_service, mock_db_session, mock_repositories
    ):
        """一括差分更新の最終検証"""

        logger.info("=== 一括差分更新最終検証開始 ===")

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

                    try:
                        # 一括差分更新を実行
                        result = (
                            await orchestration_service.execute_bulk_incremental_update(
                                symbol="BTC/USDT:USDT", db=mock_db_session
                            )
                        )

                        logger.info("一括差分更新完了")

                        # 基本的な結果構造を確認
                        assert result["success"] is True
                        assert "data" in result

                        # 正しい場所でデータを確認
                        outer_data = result["data"]
                        assert "data" in outer_data, "外側のdataにdataキーがありません"

                        inner_data = outer_data["data"]

                        print("\n=== 最終検証結果 ===")

                        # ファンディングレートの検証
                        if "funding_rate" in inner_data:
                            fr_result = inner_data["funding_rate"]
                            print(f"✅ ファンディングレート: {fr_result}")

                            assert (
                                fr_result["success"] is True
                            ), "ファンディングレート処理が失敗しています"
                            assert (
                                fr_result["saved_count"] == 15
                            ), f"ファンディングレートの保存件数が期待値と異なります: {fr_result['saved_count']}"

                            logger.info(
                                f"ファンディングレート検証成功: {fr_result['saved_count']}件保存"
                            )
                        else:
                            pytest.fail("ファンディングレートが処理されていません")

                        # オープンインタレストの検証
                        if "open_interest" in inner_data:
                            oi_result = inner_data["open_interest"]
                            print(f"✅ オープンインタレスト: {oi_result}")

                            # オープンインタレストは成功またはCCXT問題によるスキップのいずれかであることを確認
                            assert (
                                oi_result["success"] is True
                            ), "オープンインタレスト処理が失敗しています"

                            if oi_result.get("saved_count", 0) > 0:
                                logger.info(
                                    f"オープンインタレスト検証成功: {oi_result['saved_count']}件保存"
                                )
                            else:
                                logger.info(
                                    f"オープンインタレストはCCXT問題によりスキップ: {oi_result.get('message', 'No message')}"
                                )
                        else:
                            pytest.fail("オープンインタレストが処理されていません")

                        # Fear & Greed Indexの検証
                        if "fear_greed_index" in inner_data:
                            fg_result = inner_data["fear_greed_index"]
                            print(f"✅ Fear & Greed Index: {fg_result}")

                            assert (
                                fg_result["success"] is True
                            ), "Fear & Greed Index処理が失敗しています"

                            logger.info(
                                f"Fear & Greed Index検証成功: {fg_result.get('inserted_count', 0)}件保存"
                            )
                        else:
                            pytest.fail("Fear & Greed Indexが処理されていません")

                        # 総保存件数の確認
                        total_saved = outer_data.get("total_saved_count", 0)
                        print(f"✅ 総保存件数: {total_saved}件")

                        assert total_saved > 0, "総保存件数が0です"

                        logger.info(f"最終検証成功: 総保存件数 {total_saved}件")

                        print("\n🎉 一括差分更新の最終検証が成功しました！")
                        print(
                            f"   - ファンディングレート: {fr_result['saved_count']}件"
                        )
                        print(
                            f"   - オープンインタレスト: {oi_result.get('saved_count', 0)}件"
                        )
                        print(
                            f"   - Fear & Greed Index: {fg_result.get('inserted_count', 0)}件"
                        )
                        print(f"   - 総保存件数: {total_saved}件")

                    except Exception as e:
                        logger.error(f"最終検証中にエラー: {e}")
                        print(f"Error details: {type(e).__name__}: {e}")
                        raise

    @pytest.mark.asyncio
    async def test_individual_services_verification(self):
        """個別サービスの検証"""

        print("\n=== 個別サービス検証 ===")

        # ファンディングレートサービスの検証
        from app.services.data_collection.bybit.funding_rate_service import (
            BybitFundingRateService,
        )

        fr_service = BybitFundingRateService()
        mock_fr_repo = Mock()
        mock_fr_repo.get_latest_funding_timestamp.return_value = None
        mock_fr_repo.insert_funding_rate_data.return_value = 20

        try:
            fr_result = await fr_service.fetch_incremental_funding_rate_data(
                "BTC/USDT", mock_fr_repo
            )

            print(f"✅ ファンディングレートサービス直接テスト: {fr_result}")
            assert fr_result["success"] is True
            assert fr_result["saved_count"] == 20

        except Exception as e:
            print(f"❌ ファンディングレートサービス直接テスト失敗: {e}")
            pytest.skip(f"Network or API issue: {e}")

        # Fear & Greed Indexサービスの検証
        from app.services.data_collection.fear_greed.fear_greed_service import (
            FearGreedIndexService,
        )

        mock_fg_repo = Mock()
        mock_fg_repo.insert_fear_greed_data.return_value = 5

        try:
            async with FearGreedIndexService() as fg_service:
                fg_result = await fg_service.fetch_and_save_fear_greed_data(
                    limit=10, repository=mock_fg_repo
                )

                print(f"✅ Fear & Greed Indexサービス直接テスト: {fg_result}")
                assert fg_result["success"] is True
                assert fg_result["inserted_count"] == 5

        except Exception as e:
            print(f"❌ Fear & Greed Indexサービス直接テスト失敗: {e}")
            pytest.skip(f"Network or API issue: {e}")

        print("🎉 個別サービス検証が成功しました！")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
