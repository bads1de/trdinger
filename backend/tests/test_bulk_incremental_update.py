"""
一括差分更新のテスト
"""

import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base
from app.services.data_collection.orchestration.data_collection_orchestration_service import (
    DataCollectionOrchestrationService,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_bulk_incremental_update():
    """一括差分更新のテスト"""

    # インメモリデータベースを作成
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        print("=" * 60)
        print("一括差分更新テスト開始")
        print("=" * 60)

        # データ収集オーケストレーションサービスを作成
        orchestration_service = DataCollectionOrchestrationService()

        # 一括差分更新を実行
        print("\n1. 一括差分更新を実行中...")
        result = await orchestration_service.execute_bulk_incremental_update(
            symbol="BTC/USDT:USDT", db=db
        )

        print(f"✅ 一括差分更新結果: {result['success']}")
        print(f"📝 メッセージ: {result['message']}")

        # 結果の詳細を表示
        if "data" in result:
            data = result["data"]
            print(f"\n📊 更新結果詳細:")

            # OHLCV結果
            if "ohlcv_results" in data:
                ohlcv_results = data["ohlcv_results"]
                print(f"  📈 OHLCV:")
                for timeframe, result_data in ohlcv_results.items():
                    print(f"    {timeframe}: {result_data.get('inserted_count', 0)}件")

            # Funding Rate結果
            if "funding_rate_result" in data:
                fr_result = data["funding_rate_result"]
                print(f"  💰 Funding Rate: {fr_result.get('inserted_count', 0)}件")

            # Open Interest結果
            if "open_interest_result" in data:
                oi_result = data["open_interest_result"]
                print(f"  📊 Open Interest: {oi_result.get('inserted_count', 0)}件")

            # Fear & Greed Index結果
            if "fear_greed_index" in data:
                fg_result = data["fear_greed_index"]
                print(
                    f"  😨 Fear & Greed Index: {fg_result.get('inserted_count', 0)}件"
                )
                if not fg_result.get("success", False):
                    print(f"    ❌ エラー: {fg_result.get('error', 'Unknown error')}")
                else:
                    print(f"    ✅ 成功: {fg_result.get('message', 'OK')}")

        print("\n" + "=" * 60)
        print("✅ 一括差分更新テスト完了")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    # 非同期実行
    asyncio.run(test_bulk_incremental_update())
