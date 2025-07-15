"""
データ収集メインモジュール
"""

import asyncio
from datetime import datetime, timezone, timedelta
import logging

from database.connection import ensure_db_initialized
from database.repositories.ohlcv_repository import OHLCVRepository

from ..app.core.services.data_collection.market_data_service import (
    BybitMarketDataService,
)
from app.config.market_config import MarketDataConfig
from app.core.utils.data_converter import OHLCVDataConverter

logger = logging.getLogger(__name__)


class DataCollector:
    """
    市場データ収集クラス

    Bybit取引所からOHLCVデータを取得してデータベースに保存します。
    データベースセッションは外部から注入されることを想定しています。
    """

    def __init__(self, db_session):
        """
        コンストラクタ

        Args:
            db_session: SQLAlchemyのセッションオブジェクト
        """
        if not ensure_db_initialized():
            raise RuntimeError("データベースの初期化に失敗しました")
        self.db = db_session
        self.market_service = BybitMarketDataService()
        self.ohlcv_repo = OHLCVRepository(self.db)

    async def _determine_collection_range(
        self, symbol: str, timeframe: str, days_back: int
    ) -> tuple[datetime, datetime]:
        """データ収集期間を決定する"""
        latest_timestamp = self.ohlcv_repo.get_latest_timestamp(symbol, timeframe)
        if latest_timestamp:
            logger.info(f"既存データの最新タイムスタンプ: {latest_timestamp}")
            # タイムゾーンをUTCに統一
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)

            # 最新データ以降のデータを取得
            start_time = latest_timestamp + timedelta(minutes=1)
            end_time = datetime.now(timezone.utc)
        else:
            # 新規に過去データを取得
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)

        logger.info(f"データ収集期間: {start_time} ～ {end_time}")
        return start_time, end_time

    async def collect_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        days_back: int = 365,
        batch_size: int = 1000,
    ) -> int:
        """
        過去データを収集・更新します。
        既存データがある場合は、最新のタイムスタンプから現在までのデータを収集します。
        データがない場合は、指定された日数分の過去データを収集します。

        Args:
            symbol: 取引ペア（例: 'BTC/USDT:USDT'）
            timeframe: 時間軸（例: '1d'）
            days_back: 遡ってデータを取得する日数（データがない場合のみ有効）
            batch_size: 一度にAPIから取得するデータ件数

        Returns:
            収集された総件数
        """
        logger.info(f"データ収集開始: {symbol} {timeframe}")
        total_collected = 0

        try:
            normalized_symbol = MarketDataConfig.normalize_symbol(symbol)
            start_time_data, end_time = await self._determine_collection_range(
                normalized_symbol, timeframe, days_back
            )

            current_time = start_time_data
            while current_time < end_time:
                try:
                    limit = 1000  # Bybit APIの最大取得件数

                    logger.info(f"バッチ収集: {current_time} から {limit} 件")

                    # datetimeをミリ秒のUNIXタイムスタンプに変換
                    since_timestamp_ms = int(current_time.timestamp() * 1000)

                    ohlcv_data = await self.market_service.fetch_ohlcv_data(
                        normalized_symbol,
                        timeframe,
                        since=since_timestamp_ms,
                        limit=limit,
                    )

                    if not ohlcv_data:
                        logger.info(
                            "これ以上取得するデータがありません。収集を終了します。"
                        )
                        break

                    db_records = OHLCVDataConverter.ccxt_to_db_format(
                        ohlcv_data, normalized_symbol, timeframe
                    )
                    inserted_count = self.ohlcv_repo.insert_ohlcv_data(db_records)
                    total_collected += inserted_count
                    logger.info(f"バッチ完了: {inserted_count} 件挿入")

                    # 取得したデータの最後のタイムスタンプの次から収集を再開
                    last_timestamp_ms = ohlcv_data[-1][0]
                    current_time = datetime.fromtimestamp(
                        last_timestamp_ms / 1000, tz=timezone.utc
                    ) + timedelta(milliseconds=1)

                    await asyncio.sleep(1)  # APIレート制限

                except Exception as e:
                    logger.error(f"バッチ収集エラー: {e}", exc_info=True)

                    break

            logger.info(f"データ収集完了: 総 {total_collected} 件")
            return total_collected

        except Exception as e:
            logger.error(f"データ収集処理全体でエラー: {e}", exc_info=True)
            raise
