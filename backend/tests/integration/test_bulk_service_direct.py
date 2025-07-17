import asyncio
from app.core.services.data_collection.historical_data_service import HistoricalDataService
from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

async def test_bulk_with_external():
    session = SessionLocal()
    try:
        service = HistoricalDataService()
        ohlcv_repo = OHLCVRepository(session)
        fr_repo = FundingRateRepository(session)
        oi_repo = OpenInterestRepository(session)
        
        print("=== 一括差分更新テスト（外部市場データ含む） ===")
        result = await service.collect_bulk_incremental_data(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            ohlcv_repository=ohlcv_repo,
            funding_rate_repository=fr_repo,
            open_interest_repository=oi_repo,
            include_external_market=True
        )
        
        print(f"成功: {result.get('success')}")
        print(f"総保存件数: {result.get('total_saved_count', 0)}")
        
        data_results = result.get("data", {})
        print(f"OHLCV: {data_results.get('ohlcv', {}).get('saved_count', 0)}件")
        print(f"FR: {data_results.get('funding_rate', {}).get('saved_count', 0)}件")
        print(f"OI: {data_results.get('open_interest', {}).get('saved_count', 0)}件")
        
        em_result = data_results.get("external_market", {})
        if em_result:
            print(f"外部市場: {em_result.get('inserted_count', 0)}件 (取得: {em_result.get('fetched_count', 0)}件)")
            print(f"  成功: {em_result.get('success', False)}")
            print(f"  エラー: {em_result.get('error', 'なし')}")
        else:
            print("外部市場: 結果なし")
            
        if result.get("errors"):
            print(f"エラー: {result['errors']}")
            
    finally:
        session.close()

asyncio.run(test_bulk_with_external())
