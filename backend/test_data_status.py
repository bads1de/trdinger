from database.connection import SessionLocal
from database.models import (
    OHLCVData,
    FundingRateData,
    OpenInterestData,
    FearGreedIndexData,
    ExternalMarketData,
)
from database.repositories.external_market_repository import ExternalMarketRepository

print("=== データ状態テスト ===")

session = SessionLocal()
try:
    # 各データの件数を取得
    ohlcv_count = session.query(OHLCVData).count()
    fr_count = session.query(FundingRateData).count()
    oi_count = session.query(OpenInterestData).count()
    fg_count = session.query(FearGreedIndexData).count()
    em_count = session.query(ExternalMarketData).count()

    print(f"OHLCV: {ohlcv_count:,}件")
    print(f"ファンディングレート: {fr_count:,}件")
    print(f"オープンインタレスト: {oi_count:,}件")
    print(f"Fear & Greed Index: {fg_count:,}件")
    print(f"外部市場データ: {em_count:,}件")
    print(f"総件数: {ohlcv_count + fr_count + oi_count + fg_count + em_count:,}件")

    # 外部市場データの詳細
    if em_count > 0:
        em_repo = ExternalMarketRepository(session)
        em_statistics = em_repo.get_data_statistics()
        em_latest = em_repo.get_latest_data_timestamp()

        print("\n外部市場データ詳細:")
        print(f"  シンボル: {em_statistics.get('symbols', [])}")
        print(f"  シンボル数: {em_statistics.get('symbol_count', 0)}")
        print(f"  最新データ: {em_latest.isoformat() if em_latest else 'なし'}")

        date_range = em_statistics.get("date_range")
        if date_range:
            print(f"  データ範囲: {date_range['oldest']} ～ {date_range['newest']}")

finally:
    session.close()

print("✅ テスト完了")
