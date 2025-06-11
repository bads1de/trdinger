"""
テスト用サンプルデータ作成スクリプト

GA実験テスト用のOHLCVサンプルデータを作成します。
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_ohlcv_data(days=100, symbol="BTCUSDT", timeframe="1h"):
    """
    現実的なOHLCVデータを生成
    
    Args:
        days: 生成する日数
        symbol: シンボル
        timeframe: 時間軸
        
    Returns:
        OHLCVデータのリスト
    """
    # 時間軸に応じた期間数を計算
    if timeframe == "1h":
        periods = days * 24
        freq = "1H"
    elif timeframe == "4h":
        periods = days * 6
        freq = "4H"
    elif timeframe == "1d":
        periods = days
        freq = "1D"
    else:
        periods = days * 24
        freq = "1H"
    
    # 日付範囲を生成
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # 価格データを生成
    np.random.seed(42)  # 再現性のため
    initial_price = 50000.0
    
    # より現実的な価格変動を生成
    volatility = 0.02 if timeframe == "1d" else 0.005  # 日足は高ボラティリティ
    returns = np.random.normal(0, volatility, periods)
    
    # トレンドを追加
    trend = np.linspace(0, 0.2, periods)  # 20%の上昇トレンド
    returns += trend / periods
    
    # 価格を計算
    prices = [initial_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # 最低価格を設定
    
    # OHLCVデータを生成
    ohlcv_data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        # 各ローソク足のOHLCを生成
        volatility_factor = np.random.uniform(0.5, 1.5)
        high_low_range = close_price * 0.01 * volatility_factor  # 1%の範囲
        
        open_price = close_price * np.random.uniform(0.995, 1.005)
        high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range)
        low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range)
        
        # 価格の整合性を保証
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # ボリュームを生成
        base_volume = 1000000
        volume = base_volume * np.random.uniform(0.5, 2.0)
        
        ohlcv_data.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": timestamp,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })
    
    return ohlcv_data


def create_sample_data():
    """サンプルデータを作成してデータベースに保存"""
    try:
        logger.info("=== サンプルデータ作成開始 ===")
        
        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)
            
            # 既存データの確認
            existing_count = repo.count_records("BTCUSDT", "1h")
            logger.info(f"既存データ数: {existing_count} 件")
            
            if existing_count > 100:
                logger.info("十分なデータが既に存在します")
                return True
            
            # サンプルデータを生成
            logger.info("サンプルデータ生成中...")
            
            # 複数の時間軸でデータを作成
            datasets = [
                {"symbol": "BTCUSDT", "timeframe": "1h", "days": 30},
                {"symbol": "BTCUSDT", "timeframe": "4h", "days": 30},
                {"symbol": "BTCUSDT", "timeframe": "1d", "days": 30},
                {"symbol": "ETHUSDT", "timeframe": "1h", "days": 30},
            ]
            
            total_inserted = 0
            
            for dataset in datasets:
                logger.info(f"生成中: {dataset['symbol']} {dataset['timeframe']}")
                
                ohlcv_data = generate_realistic_ohlcv_data(
                    days=dataset["days"],
                    symbol=dataset["symbol"],
                    timeframe=dataset["timeframe"]
                )
                
                # データベースに保存
                inserted_count = repo.insert_ohlcv_data(ohlcv_data)
                total_inserted += inserted_count
                
                logger.info(f"保存完了: {inserted_count} 件")
            
            logger.info(f"✅ サンプルデータ作成完了: 合計 {total_inserted} 件")
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"サンプルデータ作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_sample_data():
    """作成されたサンプルデータを確認"""
    try:
        logger.info("=== サンプルデータ確認 ===")
        
        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)
            
            symbols = ["BTCUSDT", "ETHUSDT"]
            timeframes = ["1h", "4h", "1d"]
            
            for symbol in symbols:
                for timeframe in timeframes:
                    count = repo.count_records(symbol, timeframe)
                    if count > 0:
                        # 最新・最古データを取得
                        latest = repo.get_latest_timestamp(symbol, timeframe)
                        oldest = repo.get_oldest_timestamp(symbol, timeframe)
                        
                        logger.info(f"{symbol} {timeframe}: {count} 件 "
                                  f"({oldest.strftime('%Y-%m-%d')} - {latest.strftime('%Y-%m-%d')})")
                    else:
                        logger.info(f"{symbol} {timeframe}: データなし")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"データ確認エラー: {e}")
        return False


def main():
    """メイン処理"""
    try:
        # サンプルデータ作成
        if not create_sample_data():
            return False
        
        # データ確認
        if not verify_sample_data():
            return False
        
        logger.info("=== 処理完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
