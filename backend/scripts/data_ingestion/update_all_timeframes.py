#!/usr/bin/env python3
"""
全期間データ収集スクリプト (OHLCV, Funding Rate, Open Interest)

サポートされている全ての時間足について、DBに存在する最古のデータよりさらに過去へ遡って、
APIで取得可能な全期間のデータを取得します。

収集対象:
- OHLCV: 全時間足（1m, 5m, 15m, 1h, 4h, 1d など）
- Funding Rate: 8時間ごと
- Open Interest: 指定されたインターバル（デフォルト: 1h）
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# OHLCV データ収集
# =============================================================================
async def collect_ohlcv_data(symbol: str = "BTC/USDT:USDT") -> dict:
    """
    全時間足のOHLCVデータを全期間収集

    Args:
        symbol: 取引ペアシンボル

    Returns:
        収集結果の辞書
    """
    logger.info(f"=== OHLCV データ収集開始: {symbol} ===")

    from app.config.unified_config import unified_config
    from database.connection import SessionLocal
    from database.repositories.ohlcv_repository import OHLCVRepository
    from app.services.data_collection.bybit.market_data_service import (
        BybitMarketDataService,
    )

    db = SessionLocal()
    total_saved = 0
    results = {}

    try:
        repository = OHLCVRepository(db)
        market_service = BybitMarketDataService()

        # サポートされている全ての時間足を取得
        timeframes = unified_config.market.supported_timeframes
        logger.info(f"対象時間足: {timeframes}")

        for tf in timeframes:
            try:
                logger.info("-" * 40)
                logger.info(f"時間足 {tf} の処理を開始")

                saved_count_tf = 0
                max_pages = 10000
                limit = 1000

                if tf == "1m":
                    logger.info(
                        "1m足の全期間データ取得を開始します（数分かかる場合があります）"
                    )

                # DBから最古のタイムスタンプを取得
                oldest_ts = repository.get_oldest_timestamp(
                    timestamp_column="timestamp",
                    filter_conditions={"symbol": symbol, "timeframe": tf},
                )

                if oldest_ts:
                    end_ms = int(oldest_ts.timestamp() * 1000) - 1
                    logger.info(
                        f"  既存データあり。最古: {oldest_ts}。続きから過去へ取得します"
                    )
                else:
                    end_ms = None
                    logger.info("  既存データなし。現在時刻から過去へ取得します")

                for i in range(max_pages):
                    params = {}
                    if end_ms:
                        params["end"] = end_ms

                    ohlcv_data = await market_service.fetch_ohlcv_data(
                        symbol, tf, limit=limit, params=params
                    )

                    if not ohlcv_data:
                        logger.info(
                            f"  {tf}: データ取得終了 (これ以上過去のデータはありません)"
                        )
                        break

                    saved = await market_service._save_ohlcv_to_database(
                        ohlcv_data, symbol, tf, repository
                    )
                    saved_count_tf += saved

                    oldest_in_batch = ohlcv_data[0][0]
                    first_dt = datetime.fromtimestamp(
                        oldest_in_batch / 1000, tz=timezone.utc
                    )
                    last_dt = datetime.fromtimestamp(
                        ohlcv_data[-1][0] / 1000, tz=timezone.utc
                    )

                    logger.info(
                        f"  {tf} バッチ {i+1}/{max_pages}: {len(ohlcv_data)}件取得 "
                        f"({first_dt} ~ {last_dt}) -> {saved}件保存"
                    )

                    end_ms = oldest_in_batch - 1
                    await asyncio.sleep(0.1)

                logger.info(f"  {tf} 完了: 合計 {saved_count_tf}件 追加保存")
                results[tf] = {"status": "success", "saved": saved_count_tf}
                total_saved += saved_count_tf

            except Exception as e:
                logger.error(f"  {tf} の処理中にエラー: {e}")
                results[tf] = {"status": "error", "error": str(e)}

        db.commit()

    finally:
        db.close()

    logger.info(f"=== OHLCV データ収集完了: 総追加保存件数 {total_saved} ===")
    return {"type": "OHLCV", "total_saved": total_saved, "details": results}


# =============================================================================
# Funding Rate データ収集
# =============================================================================
async def collect_funding_rate_data(symbol: str = "BTC/USDT:USDT") -> dict:
    """
    ファンディングレートデータを全期間収集

    Args:
        symbol: 取引ペアシンボル

    Returns:
        収集結果の辞書
    """
    logger.info(f"=== Funding Rate データ収集開始: {symbol} ===")

    from database.connection import SessionLocal
    from database.repositories.funding_rate_repository import FundingRateRepository
    from app.services.data_collection.bybit.funding_rate_service import (
        BybitFundingRateService,
    )
    from app.utils.data_conversion import FundingRateDataConverter

    db = SessionLocal()
    total_saved = 0

    try:
        repository = FundingRateRepository(db)
        fr_service = BybitFundingRateService()

        max_pages = 5000  # FRは8時間ごとなので、5000ページあれば十分
        limit = 200  # Bybit FRのデフォルトリミット

        # DBから最古のタイムスタンプを取得
        oldest_ts = repository.get_oldest_funding_timestamp(symbol)

        if oldest_ts:
            # DBにある最古データの1ミリ秒前を終了時刻として設定
            end_ms = int(oldest_ts.timestamp() * 1000) - 1
            logger.info(
                f"  既存データあり。最古: {oldest_ts}。続きから過去へ取得します"
            )
        else:
            end_ms = None
            logger.info("  既存データなし。現在時刻から過去へ取得します")

        for i in range(max_pages):
            try:
                # Bybit API では 'until' パラメータで終了時刻を指定
                # fetch_funding_rate_history は since を使うが、過去へ遡る場合は別のアプローチが必要
                # CCXT の fetch_funding_rate_history は since と limit を使用

                # 過去方向へ遡るため、sinceは指定せず、取得データから最古を見つける方式
                if end_ms is None:
                    # 最初のリクエスト: 現在から取得
                    fr_data = await fr_service.fetch_funding_rate_history(
                        symbol, limit=limit, since=None
                    )
                else:
                    # Bybit APIでは直接過去データを指定できないため、
                    # since を使って少しずつ過去へ遡る
                    # ただしCCXTのfetch_funding_rate_historyはsinceより後のデータを返す
                    # ここでは取得済みの最古タイムスタンプより前のデータを取得するため、
                    # 推定される過去のタイムスタンプをsinceに設定
                    estimated_since = end_ms - (
                        limit * 8 * 60 * 60 * 1000
                    )  # 8時間 x limit
                    fr_data = await fr_service.fetch_funding_rate_history(
                        symbol, limit=limit, since=estimated_since
                    )

                if not fr_data:
                    logger.info(
                        "  FR: データ取得終了 (これ以上過去のデータはありません)"
                    )
                    break

                # 既に取得済みのデータをフィルタリング
                if end_ms is not None:
                    fr_data = [d for d in fr_data if d.get("timestamp", 0) < end_ms]

                if not fr_data:
                    logger.info("  FR: 新規データなし (全て取得済み)")
                    break

                # DBフォーマットに変換して保存
                db_records = FundingRateDataConverter.ccxt_to_db_format(fr_data, symbol)
                saved = repository.insert_funding_rate_data(db_records)
                total_saved += saved

                # タイムスタンプ情報を取得（CCXTのFRデータはtimestampフィールドを持つ）
                timestamps = [
                    d.get("timestamp", 0) for d in fr_data if d.get("timestamp")
                ]
                if timestamps:
                    oldest_in_batch = min(timestamps)
                    newest_in_batch = max(timestamps)
                    first_dt = datetime.fromtimestamp(
                        oldest_in_batch / 1000, tz=timezone.utc
                    )
                    last_dt = datetime.fromtimestamp(
                        newest_in_batch / 1000, tz=timezone.utc
                    )

                    logger.info(
                        f"  FR バッチ {i+1}/{max_pages}: {len(fr_data)}件取得 "
                        f"({first_dt} ~ {last_dt}) -> {saved}件保存"
                    )

                    end_ms = oldest_in_batch - 1
                else:
                    logger.warning("  FR: タイムスタンプが見つかりません")
                    break

                await asyncio.sleep(0.2)  # レート制限対策

            except Exception as e:
                logger.error(f"  FR バッチ {i+1} でエラー: {e}")
                # 続行を試みる
                break

        db.commit()

    finally:
        db.close()

    logger.info(f"=== Funding Rate データ収集完了: 総追加保存件数 {total_saved} ===")
    return {"type": "FundingRate", "total_saved": total_saved}


# =============================================================================
# Open Interest データ収集
# =============================================================================
async def collect_open_interest_data(
    symbol: str = "BTC/USDT:USDT", interval: str = "1h"
) -> dict:
    """
    オープンインタレストデータを全期間収集

    Args:
        symbol: 取引ペアシンボル
        interval: データ間隔（デフォルト: 1h）

    Returns:
        収集結果の辞書
    """
    logger.info(f"=== Open Interest データ収集開始: {symbol} (interval={interval}) ===")

    from database.connection import SessionLocal
    from database.repositories.open_interest_repository import OpenInterestRepository
    from app.services.data_collection.bybit.open_interest_service import (
        BybitOpenInterestService,
    )
    from app.utils.data_conversion import OpenInterestDataConverter

    db = SessionLocal()
    total_saved = 0

    try:
        repository = OpenInterestRepository(db)
        oi_service = BybitOpenInterestService()

        max_pages = 5000  # 1hインターバルの場合、5000ページで十分
        limit = 200

        # DBから最古のタイムスタンプを取得
        oldest_ts = repository.get_oldest_open_interest_timestamp(symbol)

        if oldest_ts:
            end_ms = int(oldest_ts.timestamp() * 1000) - 1
            logger.info(
                f"  既存データあり。最古: {oldest_ts}。続きから過去へ取得します"
            )
        else:
            end_ms = None
            logger.info("  既存データなし。現在時刻から過去へ取得します")

        for i in range(max_pages):
            try:
                # OIデータの取得
                if end_ms is None:
                    oi_data = await oi_service.fetch_open_interest_history(
                        symbol, limit=limit, since=None, interval=interval
                    )
                else:
                    # 推定される過去のタイムスタンプ
                    interval_ms = _interval_to_ms(interval)
                    estimated_since = end_ms - (limit * interval_ms)
                    oi_data = await oi_service.fetch_open_interest_history(
                        symbol, limit=limit, since=estimated_since, interval=interval
                    )

                if not oi_data:
                    logger.info(
                        "  OI: データ取得終了 (これ以上過去のデータはありません)"
                    )
                    break

                # 既に取得済みのデータをフィルタリング
                if end_ms is not None:
                    oi_data = [d for d in oi_data if d.get("timestamp", 0) < end_ms]

                if not oi_data:
                    logger.info("  OI: 新規データなし (全て取得済み)")
                    break

                # DBフォーマットに変換して保存
                db_records = OpenInterestDataConverter.ccxt_to_db_format(
                    oi_data, symbol
                )
                saved = repository.insert_open_interest_data(db_records)
                total_saved += saved

                # タイムスタンプ情報を取得
                timestamps = [
                    d.get("timestamp", 0) for d in oi_data if d.get("timestamp")
                ]
                if timestamps:
                    oldest_in_batch = min(timestamps)
                    newest_in_batch = max(timestamps)
                    first_dt = datetime.fromtimestamp(
                        oldest_in_batch / 1000, tz=timezone.utc
                    )
                    last_dt = datetime.fromtimestamp(
                        newest_in_batch / 1000, tz=timezone.utc
                    )

                    logger.info(
                        f"  OI バッチ {i+1}/{max_pages}: {len(oi_data)}件取得 "
                        f"({first_dt} ~ {last_dt}) -> {saved}件保存"
                    )

                    end_ms = oldest_in_batch - 1
                else:
                    logger.warning("  OI: タイムスタンプが見つかりません")
                    break

                await asyncio.sleep(0.2)  # レート制限対策

            except Exception as e:
                logger.error(f"  OI バッチ {i+1} でエラー: {e}")
                break

        db.commit()

    finally:
        db.close()

    logger.info(f"=== Open Interest データ収集完了: 総追加保存件数 {total_saved} ===")
    return {"type": "OpenInterest", "interval": interval, "total_saved": total_saved}


def _interval_to_ms(interval: str) -> int:
    """インターバル文字列をミリ秒に変換"""
    multipliers = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    return multipliers.get(interval, 60 * 60 * 1000)  # デフォルト: 1h


# =============================================================================
# メイン収集関数
# =============================================================================
async def collect_all_historical_data(
    symbol: str = "BTC/USDT:USDT",
    collect_ohlcv: bool = True,
    collect_fr: bool = True,
    collect_oi: bool = True,
    oi_interval: str = "1h",
) -> bool:
    """
    全種類のデータを全期間収集

    Args:
        symbol: 取引ペアシンボル
        collect_ohlcv: OHLCVデータを収集するか
        collect_fr: Funding Rateデータを収集するか
        collect_oi: Open Interestデータを収集するか
        oi_interval: OIデータのインターバル

    Returns:
        成功した場合True
    """
    logger.info("=" * 60)
    logger.info(f"全種類データ全期間収集開始 (過去方向へスクロール): {symbol}")
    logger.info(f"  OHLCV: {collect_ohlcv}, FR: {collect_fr}, OI: {collect_oi}")
    logger.info("=" * 60)

    results = []

    try:
        # OHLCV データ収集
        if collect_ohlcv:
            ohlcv_result = await collect_ohlcv_data(symbol)
            results.append(ohlcv_result)

        # Funding Rate データ収集
        if collect_fr:
            fr_result = await collect_funding_rate_data(symbol)
            results.append(fr_result)

        # Open Interest データ収集
        if collect_oi:
            oi_result = await collect_open_interest_data(symbol, oi_interval)
            results.append(oi_result)

        # 結果サマリー
        logger.info("=" * 60)
        logger.info("全処理完了。結果サマリー:")
        for result in results:
            logger.info(f"  {result['type']}: {result['total_saved']}件追加保存")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"致命的なエラー: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="全期間データ収集スクリプト")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="取引ペアシンボル")
    parser.add_argument(
        "--no-ohlcv", action="store_true", help="OHLCVデータの収集をスキップ"
    )
    parser.add_argument(
        "--no-fr", action="store_true", help="Funding Rateデータの収集をスキップ"
    )
    parser.add_argument(
        "--no-oi", action="store_true", help="Open Interestデータの収集をスキップ"
    )
    parser.add_argument(
        "--oi-interval", default="1h", help="OIデータのインターバル (1h, 4h, 1d)"
    )
    parser.add_argument("--only-ohlcv", action="store_true", help="OHLCVデータのみ収集")
    parser.add_argument(
        "--only-fr", action="store_true", help="Funding Rateデータのみ収集"
    )
    parser.add_argument(
        "--only-oi", action="store_true", help="Open Interestデータのみ収集"
    )

    args = parser.parse_args()

    # --only-* オプションの処理
    if args.only_ohlcv:
        collect_ohlcv = True
        collect_fr = False
        collect_oi = False
    elif args.only_fr:
        collect_ohlcv = False
        collect_fr = True
        collect_oi = False
    elif args.only_oi:
        collect_ohlcv = False
        collect_fr = False
        collect_oi = True
    else:
        collect_ohlcv = not args.no_ohlcv
        collect_fr = not args.no_fr
        collect_oi = not args.no_oi

    try:
        success = asyncio.run(
            collect_all_historical_data(
                symbol=args.symbol,
                collect_ohlcv=collect_ohlcv,
                collect_fr=collect_fr,
                collect_oi=collect_oi,
                oi_interval=args.oi_interval,
            )
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        sys.exit(130)



