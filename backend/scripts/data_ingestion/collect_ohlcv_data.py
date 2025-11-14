#!/usr/bin/env python3
"""
OHLCVデータ取得スクリプト
指定されたシンボルとタイムフレームの全期間データを取得します
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_ohlcv_data(symbol: str, timeframe: str):
    logger.info(f"OHLCVデータ取得開始: {symbol} {timeframe}")

    try:
        from app.services.data_collection.historical.historical_data_service import (
            HistoricalDataService,
        )
        from database.connection import SessionLocal
        from database.repositories.ohlcv_repository import OHLCVRepository

        db = SessionLocal()
        try:
            repository = OHLCVRepository(db)
            service = HistoricalDataService()

            initial_count = repository.get_data_count(symbol, timeframe)
            logger.info(f"取得前のデータ件数: {initial_count}")

            start_time = datetime.now(timezone.utc)

            result = await service.collect_historical_data_with_start_date(
                symbol=symbol,
                timeframe=timeframe,
                repository=repository,
                since_timestamp=None,
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            logger.info(f"取得結果: {result}件保存 (処理時間: {duration:.1f}秒)")

            final_count = repository.get_data_count(symbol, timeframe)
            saved_count = final_count - initial_count
            logger.info(
                f"取得後のデータ件数: {final_count} (新規保存: {saved_count}件)"
            )

            date_range = repository.get_date_range(
                timestamp_column="timestamp",
                filter_conditions={"symbol": symbol, "timeframe": timeframe},
            )

            if date_range[0] and date_range[1]:
                logger.info(f"データ期間: {date_range[0]} から {date_range[1]}")
                days_diff = (date_range[1] - date_range[0]).days
                hours_diff = days_diff * 24 + (
                    (date_range[1] - date_range[0]).seconds // 3600
                )
                minutes_diff = (
                    hours_diff * 60
                    + ((date_range[1] - date_range[0]).seconds % 3600) // 60
                )
                logger.info(
                    f"データ範囲: {days_diff}日 ({hours_diff}時間, {minutes_diff}分)"
                )
            else:
                logger.warning("データ期間情報が取得できませんでした")

            db.commit()
            logger.info(f"SUCCESS: {symbol} {timeframe} データ取得完了")
            return True

        finally:
            db.close()

    except Exception as e:
        logger.error(f"FAILED: {symbol} {timeframe} データ取得失敗: {e}")
        import traceback

        logger.error(f"エラーの詳細:\n{traceback.format_exc()}")
        return False


def validate_timeframe(timeframe: str) -> str:
    from app.config.unified_config import unified_config

    supported_timeframes = unified_config.market.supported_timeframes

    if timeframe not in supported_timeframes:
        raise argparse.ArgumentTypeError(
            f"無効なタイムフレーム: {timeframe}. "
            f"サポート対象: {', '.join(supported_timeframes)}"
        )
    return timeframe


def validate_symbol(symbol: str) -> str:
    if ":" not in symbol:
        raise argparse.ArgumentTypeError(
            f"無効なシンボル形式: {symbol}. " "正しい形式: 'BTC/USDT:USDT'"
        )
    return symbol


def main():
    parser = argparse.ArgumentParser(
        description="OHLCVデータ取得スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python collect_ohlcv_data.py --symbol BTC/USDT:USDT --timeframe 15m
  python collect_ohlcv_data.py -s ETH/USDT:USDT -t 1h
        """,
    )

    parser.add_argument(
        "-s",
        "--symbol",
        type=validate_symbol,
        default="BTC/USDT:USDT",
        help="取引ペアシンボル (デフォルト: BTC/USDT:USDT)",
    )

    parser.add_argument(
        "-t",
        "--timeframe",
        type=validate_timeframe,
        default="15m",
        help="時間軸 (デフォルト: 15m)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="詳細なログ出力")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print(f"OHLCVデータ取得スクリプト")
    print(f"シンボル: {args.symbol}")
    print(f"タイムフレーム: {args.timeframe}")
    print("=" * 60)

    try:
        success = asyncio.run(collect_ohlcv_data(args.symbol, args.timeframe))
        if success:
            print("\n[SUCCESS] データ取得完了")
            return 0
        else:
            print("\n[FAILED] データ取得失敗")
            return 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 処理が中断されました")
        return 130
    except Exception as e:
        print(f"\n[FATAL] 予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
