import asyncio
import logging
from app.core.services.data_collection.historical_data_service import HistoricalDataService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# ログレベルを設定
logging.basicConfig(level=logging.INFO)

async def test_bulk_with_external_debug():
    session = SessionLocal()
    try:
        service = HistoricalDataService()
        ohlcv_repo = OHLCVRepository(session)
        fr_repo = FundingRateRepository(session)
        oi_repo = OpenInterestRepository(session)
        
        print("=== 一括差分更新デバッグテスト ===")
        result = await service.collect_bulk_incremental_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            ohlcv_repository=ohlcv_repo,
            funding_rate_repository=fr_repo,
            open_interest_repository=oi_repo,
            include_external_market=True
        )
        
        print("\n=== 結果 ===")
        print(f"成功: {result.get('success')}")
        print(f"総保存件数: {result.get('total_saved_count', 0)}")
        
        data_results = result.get("data", {})
        print("\nデータ別結果:")
        for key, value in data_results.items():
            print(f"  {key}: {value}")
            
        if result.get("errors"):
            print(f"\nエラー: {result['errors']}")
            
    finally:
        session.close()

asyncio.run(test_bulk_with_external_debug())
