#!/usr/bin/env python3
"""
BTC/USDT:USDT複数時間足データ収集スクリプト

要求された時間足（1d, 4h, 1h, 30m, 15m）でBTC/USDT:USDTの無期限先物データを収集します。
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db
from database.repositories.ohlcv_repository import OHLCVRepository
from data_collector.collector import DataCollector
from app.core.services.market_data_service import BybitMarketDataService
from app.config.market_config import MarketDataConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_multi_timeframe_data(
    symbol: str = "BTC/USDT:USDT", timeframes: List[str] = None, days_back: int = 30
) -> dict:
    """
    複数時間足でのデータ収集

    Args:
        symbol: 収集対象のシンボル
        timeframes: 収集する時間足のリスト
        days_back: 過去何日分のデータを収集するか

    Returns:
        収集結果の辞書
    """
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "30m", "15m"]

    logger.info(f"=== {symbol} 複数時間足データ収集開始 ===")
    logger.info(f"対象時間足: {timeframes}")
    logger.info(f"収集期間: 過去{days_back}日")

    # コレクターを初期化（内部でBybitMarketDataServiceを作成）
    collector = DataCollector()

    # データベースセッション
    db = SessionLocal()
    ohlcv_repo = OHLCVRepository(db)

    results = {}

    try:
        # 各時間足でデータ収集
        for timeframe in timeframes:
            logger.info(f"\n--- {timeframe} データ収集開始 ---")

            try:
                # 履歴データ収集
                collected_count = await collector.collect_historical_data(
                    symbol=symbol, timeframe=timeframe, days_back=days_back
                )

                # 結果を記録
                results[timeframe] = {
                    "collected_count": collected_count,
                    "status": "success",
                    "error": None,
                }

                logger.info(
                    f"✅ {timeframe}: {collected_count}件のデータを収集しました"
                )

                # データベースの件数を確認
                db_count = ohlcv_repo.get_data_count(symbol, timeframe)
                logger.info(f"   データベース内の{timeframe}データ件数: {db_count}")

                # レート制限を考慮して少し待機
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ {timeframe} データ収集エラー: {e}")
                results[timeframe] = {
                    "collected_count": 0,
                    "status": "error",
                    "error": str(e),
                }

        # 収集結果のサマリー
        logger.info(f"\n=== 収集結果サマリー ===")
        total_collected = 0
        successful_timeframes = 0

        for timeframe, result in results.items():
            if result["status"] == "success":
                logger.info(f"✅ {timeframe}: {result['collected_count']}件")
                total_collected += result["collected_count"]
                successful_timeframes += 1
            else:
                logger.error(f"❌ {timeframe}: エラー - {result['error']}")

        logger.info(f"成功した時間足: {successful_timeframes}/{len(timeframes)}")
        logger.info(f"総収集件数: {total_collected}件")

        return results

    finally:
        db.close()


async def collect_latest_data_for_all_timeframes(
    symbol: str = "BTC/USDT:USDT", timeframes: List[str] = None
) -> dict:
    """
    全時間足での最新データ収集

    Args:
        symbol: 収集対象のシンボル
        timeframes: 収集する時間足のリスト

    Returns:
        収集結果の辞書
    """
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "30m", "15m"]

    logger.info(f"=== {symbol} 最新データ収集開始 ===")
    logger.info(f"対象時間足: {timeframes}")

    # コレクターを初期化（内部でBybitMarketDataServiceを作成）
    collector = DataCollector()

    # データベースセッション
    db = SessionLocal()
    ohlcv_repo = OHLCVRepository(db)

    results = {}

    try:
        # 各時間足で最新データ収集
        for timeframe in timeframes:
            logger.info(f"\n--- {timeframe} 最新データ収集 ---")

            try:
                collected_count = await collector.collect_latest_data(
                    symbol=symbol, timeframe=timeframe
                )

                results[timeframe] = {
                    "collected_count": collected_count,
                    "status": "success",
                    "error": None,
                }

                logger.info(
                    f"✅ {timeframe}: {collected_count}件の最新データを収集しました"
                )

                # レート制限を考慮して少し待機
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"❌ {timeframe} 最新データ収集エラー: {e}")
                results[timeframe] = {
                    "collected_count": 0,
                    "status": "error",
                    "error": str(e),
                }

        return results

    finally:
        db.close()


def show_collected_data_summary():
    """
    収集されたデータのサマリーを表示
    """
    logger.info("\n=== 収集データサマリー ===")

    db = SessionLocal()
    try:
        ohlcv_repo = OHLCVRepository(db)

        symbol = "BTC/USDT:USDT"
        timeframes = ["1d", "4h", "1h", "30m", "15m"]

        for timeframe in timeframes:
            count = ohlcv_repo.get_data_count(symbol, timeframe)

            if count > 0:
                # 最新と最古のタイムスタンプを取得
                latest = ohlcv_repo.get_latest_timestamp(symbol, timeframe)
                oldest = ohlcv_repo.get_oldest_timestamp(symbol, timeframe)

                logger.info(
                    f"{timeframe}: {count}件 ({oldest.date()} ～ {latest.date()})"
                )
            else:
                logger.info(f"{timeframe}: データなし")

    except Exception as e:
        logger.error(f"データサマリー表示エラー: {e}")
    finally:
        db.close()


async def main():
    """
    メイン処理
    """
    logger.info("=== BTC/USDT:USDT複数時間足データ収集スクリプト ===")

    try:
        # データベース初期化
        init_db()

        # 設定確認
        logger.info(f"サポートされているシンボル: {MarketDataConfig.SUPPORTED_SYMBOLS}")
        logger.info(
            f"サポートされている時間足: {MarketDataConfig.SUPPORTED_TIMEFRAMES}"
        )

        print("\n選択してください:")
        print("1. 履歴データ収集（過去30日）")
        print("2. 履歴データ収集（過去7日）")
        print("3. 最新データのみ収集")
        print("4. 収集データサマリー表示")
        print("5. 終了")

        choice = input("\n選択 (1-5): ").strip()

        if choice == "1":
            await collect_multi_timeframe_data(days_back=30)
        elif choice == "2":
            await collect_multi_timeframe_data(days_back=7)
        elif choice == "3":
            await collect_latest_data_for_all_timeframes()
        elif choice == "4":
            show_collected_data_summary()
        elif choice == "5":
            logger.info("終了します")
            return
        else:
            logger.error("無効な選択です")
            return

        # 処理後のデータサマリーを表示
        if choice in ["1", "2", "3"]:
            show_collected_data_summary()

    except Exception as e:
        logger.error(f"❌ スクリプト実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
