#!/usr/bin/env python3
"""
データ収集テストスクリプト（SQLite使用）
"""
import asyncio
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SQLite用の設定
os.environ["DATABASE_URL"] = "sqlite:///./trdinger_test.db"

from database.connection import Base
from database.repository import OHLCVRepository, DataCollectionLogRepository
from data_collector.collector import DataCollector

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_collection():
    """
    データ収集のテスト
    """
    try:
        logger.info("=== データ収集テスト開始 ===")
        
        # SQLiteエンジンを作成
        engine = create_engine("sqlite:///./trdinger_test.db", echo=True)
        
        # テーブルを作成
        Base.metadata.create_all(bind=engine)
        logger.info("SQLiteデータベースとテーブルを作成しました")
        
        # セッションを作成
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # データ収集器を初期化
        collector = DataCollector()
        
        # BTC/USD日足データを少量収集（テスト用）
        logger.info("BTC/USD日足データを収集中...")
        collected_count = await collector.collect_historical_data(
            symbol="BTC/USD:BTC",
            timeframe="1d",
            days_back=30,  # テスト用に30日分のみ
            batch_size=100
        )
        
        logger.info(f"=== テスト完了 ===")
        logger.info(f"収集されたデータ件数: {collected_count}")
        
        # データベースの内容を確認
        db = SessionLocal()
        try:
            ohlcv_repo = OHLCVRepository(db)
            
            # データ件数を確認
            count = ohlcv_repo.get_data_count("BTC/USD:BTC", "1d")
            logger.info(f"データベース内のBTC/USD日足データ件数: {count}")
            
            # 最新データを確認
            latest_data = ohlcv_repo.get_ohlcv_data("BTC/USD:BTC", "1d", limit=5)
            logger.info("最新5件のデータ:")
            for data in latest_data[-5:]:
                logger.info(f"  {data.timestamp}: Close={data.close}")
                
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_data_collection())
