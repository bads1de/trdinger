"""
15分足差分更新問題のデバッグスクリプト

このスクリプトは15分足の差分更新が正常に動作しない問題を調査します。
"""

import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.repositories.ohlcv_repository import OHLCVRepository
from app.core.services.historical_data_service import HistoricalDataService
from app.core.services.market_data_service import BybitMarketDataService

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def debug_15m_incremental_update():
    """15分足の差分更新問題をデバッグ"""
    symbol = "BTC/USDT:USDT"
    timeframe = "15m"

    logger.info("=== 15分足差分更新デバッグ開始 ===")

    # データベース接続
    db = SessionLocal()
    repo = OHLCVRepository(db)

    try:
        # 1. 現在のデータベース状況を確認
        logger.info("1. データベース状況確認")
        data_count = repo.get_data_count(symbol, timeframe)
        logger.info(f"15分足データ件数: {data_count}")

        if data_count > 0:
            latest_ts = repo.get_latest_timestamp(symbol, timeframe)
            oldest_ts = repo.get_oldest_timestamp(symbol, timeframe)
            logger.info(f"最新データ: {latest_ts}")
            logger.info(f"最古データ: {oldest_ts}")

            # 最新データの詳細を確認
            latest_records = repo.get_latest_ohlcv_data(symbol, timeframe, limit=5)
            logger.info("最新5件のデータ:")
            for record in latest_records:
                logger.info(
                    f"  {record.timestamp}: O={record.open}, H={record.high}, L={record.low}, C={record.close}, V={record.volume}"
                )
        else:
            logger.warning("15分足データが存在しません")

        # 2. 市場データサービスの動作確認
        logger.info("\n2. 市場データサービス確認")
        market_service = BybitMarketDataService()

        # 最新データを取得（since なし）
        logger.info("最新データ取得テスト（since なし）:")
        latest_market_data = await market_service.fetch_ohlcv_data(symbol, timeframe, 5)
        if latest_market_data:
            logger.info(f"取得件数: {len(latest_market_data)}")
            for i, candle in enumerate(latest_market_data):
                ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                logger.info(
                    f"  {i+1}: {ts} - O={candle[1]}, H={candle[2]}, L={candle[3]}, C={candle[4]}, V={candle[5]}"
                )
        else:
            logger.error("市場データの取得に失敗")

        # 3. since パラメータありでのデータ取得テスト
        if data_count > 0:
            logger.info("\n3. since パラメータ付きデータ取得テスト")
            latest_ts = repo.get_latest_timestamp(symbol, timeframe)
            since_ms = int(latest_ts.timestamp() * 1000)
            logger.info(f"since タイムスタンプ: {since_ms} ({latest_ts})")

            since_market_data = await market_service.fetch_ohlcv_data(
                symbol, timeframe, 10, since=since_ms
            )
            if since_market_data:
                logger.info(f"since付き取得件数: {len(since_market_data)}")
                for i, candle in enumerate(since_market_data):
                    ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                    logger.info(
                        f"  {i+1}: {ts} - O={candle[1]}, H={candle[2]}, L={candle[3]}, C={candle[4]}, V={candle[5]}"
                    )

                # 最新データとの比較
                if latest_market_data:
                    latest_market_ts = datetime.fromtimestamp(
                        latest_market_data[-1][0] / 1000, tz=timezone.utc
                    )
                    since_market_ts = (
                        datetime.fromtimestamp(
                            since_market_data[-1][0] / 1000, tz=timezone.utc
                        )
                        if since_market_data
                        else None
                    )
                    logger.info(f"最新市場データ時刻: {latest_market_ts}")
                    logger.info(f"since付き最新時刻: {since_market_ts}")
                    logger.info(f"DB最新データ時刻: {latest_ts}")

                    # 新しいデータがあるかチェック
                    new_data_count = 0
                    # latest_ts をタイムゾーン対応に変換
                    latest_ts_aware = (
                        latest_ts.replace(tzinfo=timezone.utc)
                        if latest_ts.tzinfo is None
                        else latest_ts
                    )
                    for candle in since_market_data:
                        candle_ts = datetime.fromtimestamp(
                            candle[0] / 1000, tz=timezone.utc
                        )
                        if candle_ts > latest_ts_aware:
                            new_data_count += 1
                    logger.info(f"新しいデータ件数: {new_data_count}")
            else:
                logger.warning("since付きデータ取得で結果なし")

        # 4. HistoricalDataService の差分更新テスト
        logger.info("\n4. HistoricalDataService 差分更新テスト")
        historical_service = HistoricalDataService()

        result = await historical_service.collect_incremental_data(
            symbol, timeframe, repo
        )
        logger.info(f"差分更新結果: {result}")

        # 5. 更新後のデータ確認
        if result.get("success") and result.get("saved_count", 0) > 0:
            logger.info("\n5. 更新後データ確認")
            new_data_count = repo.get_data_count(symbol, timeframe)
            new_latest_ts = repo.get_latest_timestamp(symbol, timeframe)
            logger.info(f"更新後データ件数: {new_data_count}")
            logger.info(f"更新後最新データ: {new_latest_ts}")

            # 最新データの詳細
            latest_records = repo.get_latest_ohlcv_data(symbol, timeframe, limit=3)
            logger.info("更新後最新3件:")
            for record in latest_records:
                logger.info(
                    f"  {record.timestamp}: O={record.open}, H={record.high}, L={record.low}, C={record.close}, V={record.volume}"
                )

        # 6. 他の時間軸との比較
        logger.info("\n6. 他の時間軸との比較")
        timeframes = ["30m", "1h", "4h", "1d"]
        for tf in timeframes:
            count = repo.get_data_count(symbol, tf)
            if count > 0:
                latest = repo.get_latest_timestamp(symbol, tf)
                logger.info(f"{tf}: {count}件, 最新={latest}")
            else:
                logger.info(f"{tf}: データなし")

    except Exception as e:
        logger.error(f"デバッグ中にエラー: {e}", exc_info=True)
    finally:
        db.close()


async def test_15m_api_endpoint():
    """15分足の差分更新APIエンドポイントをテスト"""
    logger.info("\n=== 15分足API エンドポイントテスト ===")

    import aiohttp

    url = "http://127.0.0.1:8000/api/data-collection/update"
    params = {"symbol": "BTC/USDT:USDT", "timeframe": "15m"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"API レスポンス: {result}")
                else:
                    text = await response.text()
                    logger.error(f"API エラー ({response.status}): {text}")
    except Exception as e:
        logger.error(f"API テスト中にエラー: {e}")


async def compare_timeframe_data():
    """各時間軸のデータ状況を比較"""
    logger.info("\n=== 時間軸別データ状況比較 ===")

    symbol = "BTC/USDT:USDT"
    timeframes = ["15m", "30m", "1h", "4h", "1d"]

    db = SessionLocal()
    repo = OHLCVRepository(db)

    try:
        for tf in timeframes:
            count = repo.get_data_count(symbol, tf)
            if count > 0:
                latest = repo.get_latest_timestamp(symbol, tf)
                oldest = repo.get_oldest_timestamp(symbol, tf)

                # 現在時刻との差を計算
                now = datetime.now(timezone.utc)
                # latest をタイムゾーン対応に変換
                latest_aware = (
                    latest.replace(tzinfo=timezone.utc)
                    if latest and latest.tzinfo is None
                    else latest
                )
                time_diff = now - latest_aware if latest_aware else None

                logger.info(f"{tf:>4}: {count:>6}件 | 最新: {latest} | 差: {time_diff}")
            else:
                logger.info(f"{tf:>4}: データなし")
    finally:
        db.close()


async def main():
    """メイン実行関数"""
    await debug_15m_incremental_update()
    await compare_timeframe_data()

    # APIエンドポイントテスト（サーバーが起動している場合のみ）
    try:
        await test_15m_api_endpoint()
    except Exception as e:
        logger.info(f"APIテストをスキップ（サーバー未起動?）: {e}")


if __name__ == "__main__":
    asyncio.run(main())
