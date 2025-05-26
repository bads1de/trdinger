#!/usr/bin/env python3
"""
サンプルデータを使用したデータベーステスト
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SQLite用の設定
os.environ["DATABASE_URL"] = "sqlite:///./trdinger_test.db"

from database.connection import Base
from database.repository import OHLCVRepository, DataCollectionLogRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_btc_data(days: int = 365) -> list:
    """
    BTC/USDのサンプルデータを生成
    
    Args:
        days: 生成する日数
        
    Returns:
        データベース挿入用の辞書リスト
    """
    logger.info(f"BTC/USDサンプルデータを{days}日分生成中...")
    
    # 日付範囲を生成
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
    
    # ランダムウォークで価格データを生成
    np.random.seed(42)  # 再現性のため
    base_price = 50000  # BTC基準価格
    
    # より現実的な価格変動を生成
    returns = np.random.normal(0, 0.02, len(date_range))  # 2%の標準偏差
    price_changes = np.exp(returns.cumsum())
    prices = base_price * price_changes
    
    # OHLCV データを生成
    data_records = []
    for i, timestamp in enumerate(date_range):
        close_price = prices[i]
        
        # 日中の変動を生成
        daily_volatility = abs(np.random.normal(0, 0.03))  # 3%の日中変動
        high = close_price * (1 + daily_volatility)
        low = close_price * (1 - daily_volatility)
        
        # 始値は前日終値の近辺
        if i > 0:
            prev_close = prices[i-1]
            open_price = prev_close * (1 + np.random.normal(0, 0.01))
        else:
            open_price = close_price
        
        # 高値・安値の調整
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # 出来高（ランダム）
        volume = np.random.randint(1000, 10000)
        
        record = {
            'symbol': 'BTC/USD:BTC',
            'timeframe': '1d',
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        }
        
        data_records.append(record)
    
    logger.info(f"サンプルデータ生成完了: {len(data_records)}件")
    return data_records


def test_database_operations():
    """
    データベース操作のテスト
    """
    try:
        logger.info("=== データベース操作テスト開始 ===")
        
        # SQLiteエンジンを作成
        engine = create_engine("sqlite:///./trdinger_test.db", echo=False)
        
        # 既存のテーブルを削除して再作成
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("SQLiteデータベースとテーブルを作成しました")
        
        # セッションを作成
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        try:
            # リポジトリを初期化
            ohlcv_repo = OHLCVRepository(db)
            log_repo = DataCollectionLogRepository(db)
            
            # サンプルデータを生成
            sample_data = generate_sample_btc_data(days=365)
            
            # データベースに挿入
            logger.info("データベースにサンプルデータを挿入中...")
            inserted_count = ohlcv_repo.insert_ohlcv_data(sample_data)
            logger.info(f"挿入完了: {inserted_count}件")
            
            # データ収集ログを記録
            start_time = datetime.now(timezone.utc) - timedelta(days=365)
            end_time = datetime.now(timezone.utc)
            log_repo.log_collection(
                symbol='BTC/USD:BTC',
                timeframe='1d',
                start_time=start_time,
                end_time=end_time,
                records_collected=inserted_count,
                status='success'
            )
            
            # データ取得テスト
            logger.info("データ取得テスト中...")
            
            # 件数確認
            count = ohlcv_repo.get_data_count("BTC/USD:BTC", "1d")
            logger.info(f"データベース内のBTC/USD日足データ件数: {count}")
            
            # 最新データを取得
            latest_data = ohlcv_repo.get_ohlcv_data("BTC/USD:BTC", "1d", limit=5)
            logger.info("最新5件のデータ:")
            for data in latest_data[-5:]:
                logger.info(f"  {data.timestamp.date()}: Close=${data.close:,.2f}")
            
            # DataFrameとして取得
            df = ohlcv_repo.get_ohlcv_dataframe("BTC/USD:BTC", "1d", limit=10)
            logger.info(f"DataFrame形式で取得: {len(df)}件")
            logger.info(f"DataFrame列: {list(df.columns)}")
            
            # 期間指定での取得
            recent_start = datetime.now(timezone.utc) - timedelta(days=30)
            recent_data = ohlcv_repo.get_ohlcv_data(
                "BTC/USD:BTC", "1d", 
                start_time=recent_start
            )
            logger.info(f"過去30日のデータ件数: {len(recent_data)}")
            
            # 最新タイムスタンプ取得
            latest_timestamp = ohlcv_repo.get_latest_timestamp("BTC/USD:BTC", "1d")
            logger.info(f"最新タイムスタンプ: {latest_timestamp}")
            
            # データ期間取得
            date_range = ohlcv_repo.get_date_range("BTC/USD:BTC", "1d")
            logger.info(f"データ期間: {date_range[0]} ～ {date_range[1]}")
            
            # 利用可能シンボル取得
            symbols = ohlcv_repo.get_available_symbols()
            logger.info(f"利用可能シンボル: {symbols}")
            
            logger.info("=== テスト完了 ===")
            logger.info("✅ データベース基盤が正常に動作しています")
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        raise


if __name__ == "__main__":
    test_database_operations()
