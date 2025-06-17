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

from data_collector.collector import DataCollector
from database.connection import SessionLocal, init_db
from database.repositories.ohlcv_repository import OHLCVRepository

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_data_for_all_timeframes(days_back: int = 365):
    """
    すべての定義済み時間軸でデータを収集・更新します。
    """
    try:
        init_db()
        logger.info("データベース初期化完了")

        timeframes = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
        symbol = "BTC/USDT"
        total_saved = 0

        for timeframe in timeframes:
            logger.info(f"=== {symbol} {timeframe} データ収集開始 ===")
            try:
                with SessionLocal() as db:
                    collector = DataCollector(db)
                    saved_count = await collector.collect_historical_data(
                        symbol=symbol, timeframe=timeframe, days_back=days_back
                    )
                    total_saved += saved_count
                    logger.info(f"✅ {symbol} {timeframe}: {saved_count}件保存")

            except Exception as e:
                logger.error(f"❌ {symbol} {timeframe} エラー: {e}", exc_info=True)

            await asyncio.sleep(1)  # APIレート制限

        logger.info("=== 収集完了 ===")
        logger.info(f"総保存件数: {total_saved}件")

    except Exception as e:
        logger.error(f"データ収集プロセス全体でエラー: {e}", exc_info=True)
        raise


async def show_data_status():
    """データ収集状況を表示"""
    try:
        with SessionLocal() as db:
            repository = OHLCVRepository(db)
            timeframes = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
            symbol = "BTC/USDT"

            logger.info("=== データ収集状況 ===")

            for timeframe in timeframes:
                count = repository.get_data_count(symbol, timeframe)
                latest = repository.get_latest_timestamp(symbol, timeframe)
                oldest = repository.get_oldest_timestamp(symbol, timeframe)

                logger.info(f"{symbol} {timeframe}: {count}件")
                if latest and oldest:
                    logger.info(f"  期間: {oldest.date()} ～ {latest.date()}")
                else:
                    logger.info("  データなし")

    except Exception as e:
        logger.error(f"データ状況確認エラー: {e}", exc_info=True)


async def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python collect_historical_data.py collect    # 履歴データ収集・更新")
        print("  python collect_historical_data.py status     # データ状況確認")
        return

    command = sys.argv[1]

    if command == "collect":
        await collect_data_for_all_timeframes()
    elif command == "status":
        await show_data_status()
    else:
        print(f"不明なコマンド: {command}")


if __name__ == "__main__":
    asyncio.run(main())
