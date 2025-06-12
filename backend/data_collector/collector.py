"""
データ収集メインモジュール
"""

import asyncio
from datetime import datetime, timezone, timedelta
import logging

from database.connection import SessionLocal, ensure_db_initialized
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.data_collection_log_repository import (
    DataCollectionLogRepository,
)
from app.core.services.market_data_service import BybitMarketDataService
from app.config.market_config import MarketDataConfig
from app.core.utils.data_converter import OHLCVDataConverter

logger = logging.getLogger(__name__)


class DataCollector:
    """
    市場データ収集クラス

    Bybit取引所からOHLCVデータを取得してデータベースに保存します。
    """

    def __init__(self):
        # データベース初期化確認
        if not ensure_db_initialized():
            raise RuntimeError("データベースの初期化に失敗しました")

        self.market_service = BybitMarketDataService()

    async def collect_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        days_back: int = 365,
        batch_size: int = 1000,
    ) -> int:
        """
        過去データを収集

        Args:
            symbol: 取引ペア（例: 'BTC/USD:BTC'）
            timeframe: 時間軸（例: '1d'）
            days_back: 何日前まで取得するか
            batch_size: 一度に取得する件数

        Returns:
            収集された総件数
        """
        logger.info(f"過去データ収集開始: {symbol} {timeframe} {days_back}日分")

        # データベースセッション
        db = SessionLocal()
        ohlcv_repo = OHLCVRepository(db)
        log_repo = DataCollectionLogRepository(db)

        total_collected = 0
        start_time = datetime.now(timezone.utc)

        try:
            # 正規化されたシンボルを取得
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)

            # 既存データの最新タイムスタンプを確認
            latest_timestamp = ohlcv_repo.get_latest_timestamp(
                normalized_symbol, timeframe
            )

            if latest_timestamp:
                logger.info(f"既存データの最新タイムスタンプ: {latest_timestamp}")
                # タイムゾーンを統一
                if latest_timestamp.tzinfo is None:
                    latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
                # 既存データより古いデータを取得するため、最古データから遡る
                oldest_timestamp = ohlcv_repo.get_oldest_timestamp(
                    normalized_symbol, timeframe
                )
                if oldest_timestamp:
                    if oldest_timestamp.tzinfo is None:
                        oldest_timestamp = oldest_timestamp.replace(tzinfo=timezone.utc)
                    # 最古データより古いデータを取得
                    end_time = oldest_timestamp
                    start_time_data = end_time - timedelta(days=days_back)
                else:
                    # 最新データより新しいデータを取得
                    start_time_data = latest_timestamp + timedelta(minutes=1)
                    end_time = datetime.now(timezone.utc)
            else:
                # 指定日数前から開始
                end_time = datetime.now(timezone.utc)
                start_time_data = end_time - timedelta(days=days_back)

            logger.info(f"データ収集期間: {start_time_data} ～ {end_time}")

            # バッチごとにデータを取得
            current_time = start_time_data

            while current_time < end_time:
                try:
                    # 一度に取得する期間を計算
                    batch_end = min(current_time + timedelta(days=batch_size), end_time)

                    logger.info(f"バッチ収集: {current_time} ～ {batch_end}")

                    # OHLCVデータを取得
                    ohlcv_data = await self.market_service.fetch_ohlcv_data(
                        normalized_symbol, timeframe, limit=batch_size
                    )

                    if not ohlcv_data:
                        logger.warning(
                            f"データが取得できませんでした: {normalized_symbol} {timeframe}"
                        )
                        break

                    # データベース形式に変換
                    db_records = OHLCVDataConverter.ccxt_to_db_format(
                        ohlcv_data, normalized_symbol, timeframe
                    )

                    # データベースに挿入
                    inserted_count = ohlcv_repo.insert_ohlcv_data(db_records)
                    total_collected += inserted_count

                    logger.info(f"バッチ完了: {inserted_count} 件挿入")

                    # 次のバッチへ
                    current_time = batch_end

                    # レート制限を考慮して少し待機
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"バッチ収集エラー: {e}")
                    # エラーログを記録
                    log_repo.log_collection(
                        normalized_symbol,
                        timeframe,
                        current_time,
                        batch_end,
                        0,
                        "error",
                        str(e),
                    )
                    break

            # 成功ログを記録
            end_time_actual = datetime.now(timezone.utc)
            log_repo.log_collection(
                normalized_symbol,
                timeframe,
                start_time_data,
                end_time_actual,
                total_collected,
                "success",
            )

            logger.info(f"過去データ収集完了: 総 {total_collected} 件")
            return total_collected

        except Exception as e:
            logger.error(f"過去データ収集エラー: {e}")
            # エラーログを記録
            end_time_actual = datetime.now(timezone.utc)
            log_repo.log_collection(
                normalized_symbol,
                timeframe,
                start_time_data if "start_time_data" in locals() else start_time,
                end_time_actual,
                total_collected,
                "error",
                str(e),
            )
            raise
        finally:
            db.close()

    async def collect_latest_data(self, symbol: str, timeframe: str = "1d") -> int:
        """
        最新データを収集（増分更新）

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            収集された件数
        """
        logger.info(f"最新データ収集開始: {symbol} {timeframe}")

        db = SessionLocal()
        ohlcv_repo = OHLCVRepository(db)
        log_repo = DataCollectionLogRepository(db)

        try:
            # 正規化されたシンボルを取得
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)

            # 最新のOHLCVデータを取得
            ohlcv_data = await self.market_service.fetch_ohlcv_data(
                normalized_symbol, timeframe, limit=100  # 最新100件
            )

            if not ohlcv_data:
                logger.warning(f"最新データが取得できませんでした: {normalized_symbol}")
                return 0

            # データベース形式に変換
            db_records = OHLCVDataConverter.ccxt_to_db_format(
                ohlcv_data, normalized_symbol, timeframe
            )

            # データベースに挿入（重複は無視）
            inserted_count = ohlcv_repo.insert_ohlcv_data(db_records)

            # ログを記録
            start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            end_time = datetime.now(timezone.utc)
            log_repo.log_collection(
                normalized_symbol,
                timeframe,
                start_time,
                end_time,
                inserted_count,
                "success",
            )

            logger.info(f"最新データ収集完了: {inserted_count} 件")
            return inserted_count

        except Exception as e:
            logger.error(f"最新データ収集エラー: {e}")
            # エラーログを記録
            start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            end_time = datetime.now(timezone.utc)
            log_repo.log_collection(
                normalized_symbol, timeframe, start_time, end_time, 0, "error", str(e)
            )
            raise
        finally:
            db.close()


# _convert_to_db_format メソッドは OHLCVDataConverter.ccxt_to_db_format に移動されました


# 便利関数
async def collect_btc_daily_data(days_back: int = 365) -> int:
    """
    BTC/USDT日足データを収集

    Args:
        days_back: 何日前まで取得するか

    Returns:
        収集された件数
    """
    collector = DataCollector()
    return await collector.collect_historical_data(
        symbol="BTC/USDT", timeframe="1d", days_back=days_back
    )
