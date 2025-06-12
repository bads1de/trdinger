#!/usr/bin/env python3
"""
15分足データ管理スクリプト（改良版）

機能:
- 15分足データの詳細な状況確認
- データギャップ分析と他の時間軸との比較
- 期間指定データ収集
- 他の時間軸との同期機能
- 効率的な差分データ収集
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


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


class Enhanced15mDataManager:
    """15分足データ管理の拡張クラス"""

    def __init__(self):
        self.symbol = "BTC/USDT:USDT"
        self.timeframe = "15m"
        self.all_timeframes = ["15m", "30m", "1h", "4h", "1d"]

    def get_timeframe_stats(self) -> Dict:
        """全時間軸のデータ統計を取得"""
        db = SessionLocal()
        try:
            repo = OHLCVRepository(db)
            stats = {}

            for tf in self.all_timeframes:
                count = repo.get_data_count(self.symbol, tf)
                if count > 0:
                    oldest = repo.get_oldest_timestamp(self.symbol, tf)
                    latest = repo.get_latest_timestamp(self.symbol, tf)
                    duration = (latest - oldest).days if oldest and latest else 0
                    stats[tf] = {
                        "count": count,
                        "oldest": oldest,
                        "latest": latest,
                        "duration_days": duration,
                    }
                else:
                    stats[tf] = {
                        "count": 0,
                        "oldest": None,
                        "latest": None,
                        "duration_days": 0,
                    }

            return stats
        finally:
            db.close()

    def analyze_data_gaps(self) -> Dict:
        """データギャップを分析"""
        stats = self.get_timeframe_stats()

        # 他の時間軸の最古データを参照
        reference_oldest = None
        reference_latest = None

        for tf in ["1d", "4h", "1h", "30m"]:  # 15m以外の時間軸
            if stats[tf]["count"] > 0 and stats[tf]["oldest"]:
                if reference_oldest is None or stats[tf]["oldest"] < reference_oldest:
                    reference_oldest = stats[tf]["oldest"]
                if reference_latest is None or stats[tf]["latest"] > reference_latest:
                    reference_latest = stats[tf]["latest"]

        # 15分足の状況
        current_15m = stats["15m"]

        gaps = {
            "reference_period": {
                "oldest": reference_oldest,
                "latest": reference_latest,
                "duration_days": (
                    (reference_latest - reference_oldest).days
                    if reference_oldest and reference_latest
                    else 0
                ),
            },
            "current_15m": current_15m,
            "missing_periods": [],
        }

        if reference_oldest and current_15m["oldest"]:
            # 開始期間のギャップ
            if current_15m["oldest"] > reference_oldest:
                gap_days = (current_15m["oldest"] - reference_oldest).days
                gaps["missing_periods"].append(
                    {
                        "type": "historical_gap",
                        "start": reference_oldest,
                        "end": current_15m["oldest"],
                        "duration_days": gap_days,
                        "priority": "high",
                    }
                )

        return gaps

    async def collect_period_data(
        self, start_date: datetime, end_date: datetime, max_batches: int = 200
    ) -> Dict:
        """指定期間のデータを収集"""
        # タイムゾーン情報の正規化（offset-naive を offset-aware に変換）
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        logger.info(f"期間指定データ収集: {start_date} ～ {end_date}")

        market_service = BybitMarketDataService()
        db = SessionLocal()

        try:
            repo = OHLCVRepository(db)
            total_collected = 0

            # 期間を逆順（新しい日付から古い日付へ）で処理
            current_end = end_date
            batch_size = 1000

            for batch_num in range(max_batches):
                # since パラメータを計算（現在の終了時刻から1000件分前）
                since_timestamp = int(
                    (current_end - timedelta(minutes=15 * batch_size)).timestamp()
                    * 1000
                )

                try:
                    # データ取得
                    ohlcv_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        market_service.exchange.fetch_ohlcv,
                        market_service.normalize_symbol(self.symbol),
                        self.timeframe,
                        since_timestamp,
                        batch_size,
                    )

                    if not ohlcv_data or len(ohlcv_data) < 10:
                        logger.info(f"バッチ {batch_num + 1}: データ終了")
                        break

                    # 指定期間内のデータのみフィルタ
                    filtered_data = []
                    for candle in ohlcv_data:
                        candle_time = datetime.fromtimestamp(
                            candle[0] / 1000, tz=timezone.utc
                        )
                        if start_date <= candle_time <= end_date:
                            filtered_data.append(candle)

                    if filtered_data:
                        # データベースに保存
                        records = []
                        for candle in filtered_data:
                            timestamp, open_price, high, low, close, volume = candle
                            records.append(
                                {
                                    "symbol": self.symbol,
                                    "timeframe": self.timeframe,
                                    "timestamp": datetime.fromtimestamp(
                                        timestamp / 1000, tz=timezone.utc
                                    ),
                                    "open": float(open_price),
                                    "high": float(high),
                                    "low": float(low),
                                    "close": float(close),
                                    "volume": float(volume),
                                }
                            )

                        saved_count = repo.insert_ohlcv_data(records)
                        total_collected += saved_count
                        logger.info(f"バッチ {batch_num + 1}: {saved_count}件保存")

                    # 次のバッチの準備
                    oldest_timestamp = min(candle[0] for candle in ohlcv_data)
                    current_end = datetime.fromtimestamp(
                        oldest_timestamp / 1000, tz=timezone.utc
                    )

                    # 開始日時に到達したら終了
                    if current_end <= start_date:
                        logger.info(f"指定期間の開始日時に到達しました")
                        break

                    await asyncio.sleep(0.1)  # APIレート制限対応

                except Exception as e:
                    logger.warning(f"バッチ {batch_num + 1} でエラー: {e}")
                    continue

            logger.info(f"期間指定収集完了: 総計 {total_collected}件")
            return {
                "success": True,
                "collected_count": total_collected,
                "period": f"{start_date} ～ {end_date}",
            }

        except Exception as e:
            logger.error(f"期間指定収集エラー: {e}")
            return {"success": False, "error": str(e)}
        finally:
            db.close()

    async def sync_with_other_timeframes(self) -> Dict:
        """他の時間軸と同じ期間までデータを同期"""
        logger.info("他の時間軸との同期開始")

        gaps = self.analyze_data_gaps()

        if not gaps["missing_periods"]:
            logger.info("同期の必要なギャップが見つかりませんでした")
            return {"success": True, "message": "同期済み"}

        total_collected = 0

        for gap in gaps["missing_periods"]:
            logger.info(
                f"ギャップ収集: {gap['start']} ～ {gap['end']} ({gap['duration_days']}日)"
            )

            result = await self.collect_period_data(gap["start"], gap["end"])

            if result["success"]:
                total_collected += result["collected_count"]
                logger.info(f"ギャップ収集完了: {result['collected_count']}件")
            else:
                logger.error(f"ギャップ収集失敗: {result.get('error')}")

        return {
            "success": True,
            "total_collected": total_collected,
            "gaps_processed": len(gaps["missing_periods"]),
        }


def show_data_details(symbol: str = "BTC/USDT:USDT"):
    """
    指定されたシンボルのデータ詳細を表示する（改良版）
    """
    logger.info(f"\n=== {symbol} データ詳細確認 ===")

    manager = Enhanced15mDataManager()
    stats = manager.get_timeframe_stats()

    # 全体統計
    db = SessionLocal()
    try:
        total_count = db.query(OHLCVData).count()
        logger.info(f"総OHLCVデータ件数: {total_count}")
    finally:
        db.close()

    # 時間軸別詳細
    logger.info("\n📊 時間軸別データ統計:")
    for tf in manager.all_timeframes:
        stat = stats[tf]
        logger.info(f"  {tf}: {stat['count']}件")
        if stat["oldest"] and stat["latest"]:
            logger.info(
                f"    期間: {stat['oldest']} ～ {stat['latest']} ({stat['duration_days']}日)"
            )

    # ギャップ分析
    gaps = manager.analyze_data_gaps()
    logger.info("\n🔍 データギャップ分析:")

    if gaps["reference_period"]["oldest"]:
        logger.info(
            f"参照期間（他の時間軸）: {gaps['reference_period']['oldest']} ～ {gaps['reference_period']['latest']} ({gaps['reference_period']['duration_days']}日)"
        )

        current = gaps["current_15m"]
        if current["oldest"]:
            logger.info(
                f"15分足現在期間: {current['oldest']} ～ {current['latest']} ({current['duration_days']}日)"
            )

            missing_days = (
                gaps["reference_period"]["duration_days"] - current["duration_days"]
            )
            logger.info(f"不足期間: 約{missing_days}日")

            for gap in gaps["missing_periods"]:
                logger.info(
                    f"  ギャップ: {gap['start']} ～ {gap['end']} ({gap['duration_days']}日) - 優先度: {gap['priority']}"
                )
        else:
            logger.info("15分足データが存在しません")
    else:
        logger.info("参照データが不足しています")


async def collect_gap_data(symbol: str = "BTC/USDT:USDT") -> dict:
    """データギャップを自動検出して収集"""
    logger.info("=== データギャップ自動収集 ===")

    manager = Enhanced15mDataManager()
    result = await manager.sync_with_other_timeframes()

    return result


async def collect_specific_period(
    symbol: str = "BTC/USDT:USDT", days_back: int = 365
) -> dict:
    """指定期間のデータを収集"""
    logger.info(f"=== 指定期間データ収集（過去{days_back}日） ===")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    manager = Enhanced15mDataManager()
    result = await manager.collect_period_data(start_date, end_date)

    return result


async def collect_to_match_reference(symbol: str = "BTC/USDT:USDT") -> dict:
    """他の時間軸の最古データまで遡って収集"""
    logger.info("=== 参照期間まで遡り収集 ===")

    manager = Enhanced15mDataManager()
    gaps = manager.analyze_data_gaps()

    if not gaps["reference_period"]["oldest"]:
        return {"success": False, "message": "参照期間が見つかりません"}

    # 参照期間の開始から現在まで収集
    start_date = gaps["reference_period"]["oldest"]
    end_date = datetime.now(timezone.utc)

    logger.info(f"収集期間: {start_date} ～ {end_date}")

    result = await manager.collect_period_data(start_date, end_date, max_batches=500)

    return result


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
        print("\n" + "=" * 60)
        print("15分足データ管理メニュー（改良版）:")
        print("=" * 60)
        print("📊 データ確認:")
        print("  1. データ詳細確認（ギャップ分析含む）")
        print("")
        print("🔄 自動収集:")
        print("  2. データギャップ自動収集（推奨）")
        print("  3. 他の時間軸と同期（参照期間まで遡り）")
        print("")
        print("📅 期間指定収集:")
        print("  4. 期間指定収集（過去90日）")
        print("  5. 期間指定収集（過去180日）")
        print("  6. 期間指定収集（過去365日）")
        print("  7. 期間指定収集（過去730日）")
        print("")
        print("🔧 従来機能:")
        print("  8. 全期間データ収集（従来版）")
        print("  9. 最新データ収集（過去90日）")
        print("")
        print("  0. 終了")
        print("=" * 60)

        choice = input("選択 (0-9): ").strip()

        if choice == "1":
            show_data_details()
        elif choice == "2":
            result = await collect_gap_data()
            if result["success"]:
                logger.info(
                    f"✅ ギャップ収集完了: {result.get('total_collected', 0)}件"
                )
            show_data_details()
        elif choice == "3":
            result = await collect_to_match_reference()
            if result["success"]:
                logger.info(
                    f"✅ 参照期間同期完了: {result.get('collected_count', 0)}件"
                )
            show_data_details()
        elif choice == "4":
            result = await collect_specific_period(days_back=90)
            if result["success"]:
                logger.info(f"✅ 90日収集完了: {result.get('collected_count', 0)}件")
            show_data_details()
        elif choice == "5":
            result = await collect_specific_period(days_back=180)
            if result["success"]:
                logger.info(f"✅ 180日収集完了: {result.get('collected_count', 0)}件")
            show_data_details()
        elif choice == "6":
            result = await collect_specific_period(days_back=365)
            if result["success"]:
                logger.info(f"✅ 365日収集完了: {result.get('collected_count', 0)}件")
            show_data_details()
        elif choice == "7":
            result = await collect_specific_period(days_back=730)
            if result["success"]:
                logger.info(f"✅ 730日収集完了: {result.get('collected_count', 0)}件")
            show_data_details()
        elif choice == "8":
            await collect_full_historical_data()
            show_data_details()
        elif choice == "9":
            await collect_recent_data(days_back=90)
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
