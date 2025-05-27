#!/usr/bin/env python3
"""
履歴データ収集スクリプト

ビットコインの全時間軸データを収集してデータベースに保存します。
"""
import asyncio
import logging
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.historical_data_service import HistoricalDataService
from database.connection import SessionLocal, init_db
from database.repository import OHLCVRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def collect_bitcoin_data():
    """ビットコインの全時間軸データを収集"""
    try:
        # データベース初期化
        init_db()
        logger.info("データベース初期化完了")

        # 履歴データサービス
        service = HistoricalDataService()

        # 収集する時間軸
        timeframes = ["1d", "4h", "1h", "30m", "15m"]
        symbol = "BTC/USDT"

        total_saved = 0

        for timeframe in timeframes:
            logger.info(f"=== {symbol} {timeframe} データ収集開始 ===")

            # データベースセッション
            db = SessionLocal()
            try:
                repository = OHLCVRepository(db)

                # 履歴データ収集
                result = await service.collect_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    repository=repository
                )

                if result["success"]:
                    saved_count = result["saved_count"]
                    total_saved += saved_count
                    logger.info(f"✅ {symbol} {timeframe}: {saved_count}件保存")
                else:
                    logger.error(f"❌ {symbol} {timeframe}: {result.get('message')}")

            except Exception as e:
                logger.error(f"❌ {symbol} {timeframe} エラー: {e}")
            finally:
                db.close()

            # API制限対応のため少し待機
            await asyncio.sleep(1)

        logger.info(f"=== 収集完了 ===")
        logger.info(f"総保存件数: {total_saved}件")

    except Exception as e:
        logger.error(f"履歴データ収集エラー: {e}")
        raise


async def update_incremental_data():
    """差分データを更新"""
    try:
        service = HistoricalDataService()
        timeframes = ["1d", "4h", "1h", "30m", "15m"]
        symbol = "BTC/USDT"

        total_saved = 0

        for timeframe in timeframes:
            logger.info(f"=== {symbol} {timeframe} 差分更新開始 ===")

            db = SessionLocal()
            try:
                repository = OHLCVRepository(db)

                result = await service.collect_incremental_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    repository=repository
                )

                if result["success"]:
                    saved_count = result["saved_count"]
                    total_saved += saved_count
                    logger.info(f"✅ {symbol} {timeframe}: {saved_count}件更新")
                else:
                    logger.error(f"❌ {symbol} {timeframe}: {result.get('message')}")

            except Exception as e:
                logger.error(f"❌ {symbol} {timeframe} エラー: {e}")
            finally:
                db.close()

            await asyncio.sleep(0.5)

        logger.info(f"=== 差分更新完了 ===")
        logger.info(f"総更新件数: {total_saved}件")

    except Exception as e:
        logger.error(f"差分データ更新エラー: {e}")
        raise


async def show_data_status():
    """データ収集状況を表示"""
    try:
        db = SessionLocal()
        try:
            repository = OHLCVRepository(db)
            timeframes = ["1d", "4h", "1h", "30m", "15m"]
            symbol = "BTC/USDT"

            logger.info("=== データ収集状況 ===")

            for timeframe in timeframes:
                count = repository.get_data_count(symbol, timeframe)
                latest = repository.get_latest_timestamp(symbol, timeframe)
                oldest = repository.get_oldest_timestamp(symbol, timeframe)

                logger.info(f"{symbol} {timeframe}: {count}件")
                if latest and oldest:
                    logger.info(f"  期間: {oldest} ～ {latest}")
                else:
                    logger.info("  データなし")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"データ状況確認エラー: {e}")


async def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python collect_historical_data.py collect    # 履歴データ収集")
        print("  python collect_historical_data.py update     # 差分データ更新")
        print("  python collect_historical_data.py status     # データ状況確認")
        return

    command = sys.argv[1]

    if command == "collect":
        await collect_bitcoin_data()
    elif command == "update":
        await update_incremental_data()
    elif command == "status":
        await show_data_status()
    else:
        print(f"不明なコマンド: {command}")


if __name__ == "__main__":
    asyncio.run(main())
