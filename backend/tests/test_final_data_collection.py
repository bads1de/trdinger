"""
最終的なデータ収集テスト

修正されたデータ収集処理をテストします。
"""

import asyncio
import logging
from unittest.mock import Mock
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


async def test_final_data_collection():
    """最終的なデータ収集テスト"""

    # モックDBセッション
    mock_db = Mock(spec=Session)

    # モックリポジトリ
    mock_ohlcv_repo = Mock(spec=OHLCVRepository)
    mock_ohlcv_repo.get_data_count.return_value = 100  # 既存データがある

    mock_funding_repo = Mock(spec=FundingRateRepository)
    mock_funding_repo.get_latest_funding_timestamp.return_value = None
    mock_funding_repo.insert_funding_rate_data.return_value = 10

    mock_oi_repo = Mock(spec=OpenInterestRepository)
    mock_oi_repo.get_latest_open_interest_timestamp.return_value = None
    mock_oi_repo.insert_open_interest_data.return_value = 5

    # データ収集サービス
    orchestration_service = DataCollectionOrchestrationService()

    try:
        # 一括差分更新を実行
        logger.info("=== 一括差分更新テスト開始 ===")

        # モックリポジトリを使用してテスト
        with (
            Mock() as mock_ohlcv_class,
            Mock() as mock_funding_class,
            Mock() as mock_oi_class,
        ):
            mock_ohlcv_class.return_value = mock_ohlcv_repo
            mock_funding_class.return_value = mock_funding_repo
            mock_oi_class.return_value = mock_oi_repo

            # パッチを適用
            import app.services.data_collection.orchestration.data_collection_orchestration_service as orchestration_module

            orchestration_module.OHLCVRepository = mock_ohlcv_class
            orchestration_module.FundingRateRepository = mock_funding_class
            orchestration_module.OpenInterestRepository = mock_oi_class

            result = await orchestration_service.execute_bulk_incremental_update(
                symbol="BTC/USDT:USDT", db=mock_db
            )

            logger.info(f"一括差分更新結果: {result}")

            # 結果の検証
            assert result["success"] is True
            assert "data" in result

            # 各データタイプの結果を確認
            data = result["data"]

            # OHLCV（既存データがあるのでスキップされる可能性）
            if "ohlcv" in data:
                logger.info(f"OHLCV結果: {data['ohlcv']}")

            # ファンディングレート
            if "funding_rate" in data:
                funding_result = data["funding_rate"]
                logger.info(f"ファンディングレート結果: {funding_result}")
                if funding_result["success"]:
                    logger.info("✅ ファンディングレート取得成功")
                else:
                    logger.error(
                        f"❌ ファンディングレート取得失敗: {funding_result.get('error')}"
                    )

            # オープンインタレスト
            if "open_interest" in data:
                oi_result = data["open_interest"]
                logger.info(f"オープンインタレスト結果: {oi_result}")
                if oi_result["success"]:
                    logger.info("✅ オープンインタレスト取得成功")
                else:
                    logger.error(
                        f"❌ オープンインタレスト取得失敗: {oi_result.get('error')}"
                    )

            # Fear & Greed Index
            if "fear_greed_index" in data:
                fg_result = data["fear_greed_index"]
                logger.info(f"Fear & Greed Index結果: {fg_result}")
                if fg_result["success"]:
                    logger.info("✅ Fear & Greed Index取得成功")
                else:
                    logger.error(
                        f"❌ Fear & Greed Index取得失敗: {fg_result.get('error')}"
                    )

            logger.info("=== 一括差分更新テスト完了 ===")

    except Exception as e:
        logger.error(f"テスト中にエラーが発生: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_final_data_collection())
