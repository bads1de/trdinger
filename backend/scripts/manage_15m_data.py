#!/usr/bin/env python3
"""
15分足データ管理スクリプト

機能:
- 15分足データの詳細な状況確認
- 全期間（2020年〜）の履歴データ収集
- 指定した期間の履歴データ収集
"""

import asyncio
import logging
import sys
import os


# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal, init_db
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.historical_data_service import HistoricalDataService
from app.core.services.market_data_service import BybitMarketDataService
from database.models import OHLCVData
from sqlalchemy import func

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def show_data_details(symbol: str = "BTC/USDT:USDT"):
    """
    指定されたシンボルのデータ詳細を表示する
    check_15m_data.py の機能を統合
    """
    logger.info(f"\n=== {symbol} データ詳細確認 ===")
    db = SessionLocal()
    try:
        repo = OHLCVRepository(db)

        # 全体のデータ件数
        total_count = db.query(OHLCVData).count()
        logger.info(f"総OHLCVデータ件数: {total_count}")

        # シンボル別・時間軸別の件数
        logger.info("\n📊 シンボル・時間軸別データ件数:")
        results = (
            db.query(
                OHLCVData.symbol,
                OHLCVData.timeframe,
                func.count(OHLCVData.id).label("count"),
                func.min(OHLCVData.timestamp).label("oldest"),
                func.max(OHLCVData.timestamp).label("latest"),
            )
            .group_by(OHLCVData.symbol, OHLCVData.timeframe)
            .all()
        )

        for result in results:
            logger.info(f"  {result.symbol} - {result.timeframe}: {result.count}件")
            if result.oldest and result.latest:
                logger.info(f"    期間: {result.oldest} ～ {result.latest}")

        # 15分足データの詳細確認
        logger.info("\n🔍 15分足データの詳細:")
        timeframe = "15m"
        count_15m = repo.get_data_count(symbol, timeframe)
        logger.info(f"15分足データ件数: {count_15m}")

        if count_15m > 0:
            oldest = repo.get_oldest_timestamp(symbol, timeframe)
            latest = repo.get_latest_timestamp(symbol, timeframe)
            logger.info(f"最古データ: {oldest}")
            logger.info(f"最新データ: {latest}")

            if oldest and latest:
                duration = latest - oldest
                logger.info(f"データ期間: {duration.days}日")

                # 理論的な15分足データ数を計算
                total_minutes = duration.total_seconds() / 60
                theoretical_count = int(total_minutes / 15)
                if theoretical_count > 0:
                    logger.info(f"理論的15分足データ数: {theoretical_count}")
                    logger.info(f"実際のデータ数: {count_15m}")
                    logger.info(
                        f"データ充足率: {count_15m / theoretical_count * 100:.2f}%"
                    )

    except Exception as e:
        logger.error(f"データ詳細表示エラー: {e}")
    finally:
        db.close()


async def collect_full_historical_data(symbol: str = "BTC/USDT:USDT") -> dict:
    """
    15分足データを全期間収集（2020年から現在まで）
    """
    timeframe = "15m"
    logger.info(f"=== {symbol} {timeframe} 全期間データ収集開始 ===")
    logger.info("収集期間: 2020年3月25日から現在まで（他の時間軸と同期）")

    market_service = BybitMarketDataService()
    historical_service = HistoricalDataService(market_service)
    db = SessionLocal()
    ohlcv_repo = OHLCVRepository(db)

    try:
        count_before = ohlcv_repo.get_data_count(symbol, timeframe)
        logger.info(f"収集前のデータ件数: {count_before}")
        if count_before > 0:
            logger.info(
                f"既存データ期間: {ohlcv_repo.get_oldest_timestamp(symbol, timeframe)} ～ {ohlcv_repo.get_latest_timestamp(symbol, timeframe)}"
            )

        logger.info("\n🚀 全期間履歴データ収集開始...")
        total_collected = 0
        max_iterations = 10  # 連続して実行し、取れるだけデータを取得

        for i in range(max_iterations):
            logger.info(f"--- 収集ラウンド {i + 1}/{max_iterations} ---")
            result = await historical_service.collect_historical_data(
                symbol=symbol, timeframe=timeframe, repository=ohlcv_repo
            )
            if result.get("success"):
                collected_count = result.get("saved_count", 0)
                total_collected += collected_count
                logger.info(f"ラウンド {i + 1}: {collected_count}件収集")
                if collected_count < 50:  # 新規データが少なくなったら完了とみなす
                    logger.info("新規データが少なくなったため、収集を完了します。")
                    break
            else:
                logger.warning(f"ラウンド {i + 1}: 収集に失敗またはデータなし。")
                break
            await asyncio.sleep(3)  # APIレート制限のための待機

        logger.info(f"✅ 全期間収集完了: 総計{total_collected}件")

        count_after = ohlcv_repo.get_data_count(symbol, timeframe)
        logger.info(f"収集後のデータ件数: {count_after}")
        logger.info(f"新規追加件数: {count_after - count_before}")
        if count_after > 0:
            logger.info(
                f"最終データ期間: {ohlcv_repo.get_oldest_timestamp(symbol, timeframe)} ～ {ohlcv_repo.get_latest_timestamp(symbol, timeframe)}"
            )

        return {
            "collected_count": total_collected,
            "total_count": count_after,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"❌ 全期間データ収集エラー: {e}", exc_info=True)
        return {
            "collected_count": 0,
            "total_count": 0,
            "status": "error",
            "error": str(e),
        }
    finally:
        db.close()


async def collect_recent_data(
    symbol: str = "BTC/USDT:USDT", days_back: int = 90
) -> dict:
    """
    指定された日数分のデータを集中的に収集する
    注: 現状のHistoricalDataServiceは全期間を対象とするため、days_backは将来的な拡張のためのものです。
    """
    timeframe = "15m"
    logger.info(f"=== {symbol} {timeframe} 過去{days_back}日データ集中収集開始 ===")

    market_service = BybitMarketDataService()
    historical_service = HistoricalDataService(market_service)
    db = SessionLocal()
    ohlcv_repo = OHLCVRepository(db)

    try:
        count_before = ohlcv_repo.get_data_count(symbol, timeframe)
        logger.info(f"収集前のデータ件数: {count_before}")

        logger.info(f"\n🚀 過去{days_back}日間のデータ収集開始...")

        result = await historical_service.collect_historical_data(
            symbol=symbol, timeframe=timeframe, repository=ohlcv_repo
        )

        collected_count = result.get("saved_count", 0) if result.get("success") else 0
        logger.info(f"✅ 収集完了: {collected_count}件")

        count_after = ohlcv_repo.get_data_count(symbol, timeframe)
        logger.info(f"収集後のデータ件数: {count_after}")
        logger.info(f"新規追加件数: {count_after - count_before}")

        return {
            "collected_count": collected_count,
            "total_count": count_after,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"❌ 集中データ収集エラー: {e}", exc_info=True)
        return {
            "collected_count": 0,
            "total_count": 0,
            "status": "error",
            "error": str(e),
        }
    finally:
        db.close()


async def main():
    """メイン処理"""
    logger.info("=== 15分足データ管理スクリプト ===")

    try:
        init_db()
    except Exception as e:
        logger.error(f"データベースの初期化に失敗しました: {e}")
        return

    while True:
        print("\n" + "=" * 50)
        print("メニューを選択してください:")
        print("1. データ詳細確認")
        print("2. 全期間データ収集（2020年〜）")
        print("3. 期間指定データ収集（過去90日）")
        print("4. 期間指定データ収集（過去180日）")
        print("5. 期間指定データ収集（過去365日）")
        print("0. 終了")
        print("=" * 50)

        choice = input("選択 (0-5): ").strip()

        if choice == "1":
            show_data_details()
        elif choice == "2":
            await collect_full_historical_data()
            show_data_details()  # 実行後に最新の状況を表示
        elif choice == "3":
            await collect_recent_data(days_back=90)
            show_data_details()
        elif choice == "4":
            await collect_recent_data(days_back=180)
            show_data_details()
        elif choice == "5":
            await collect_recent_data(days_back=365)
            show_data_details()
        elif choice == "0":
            logger.info("スクリプトを終了します。")
            break
        else:
            logger.error("無効な選択です。もう一度選択してください。")

        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nスクリプトが中断されました。")
    except Exception as e:
        logger.error(
            f"❌ スクリプト実行中に予期せぬエラーが発生しました: {e}", exc_info=True
        )
        sys.exit(1)
