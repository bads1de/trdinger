"""
履歴データ収集サービス

バックテスト用の包括的なOHLCVデータ収集を行います。
"""

import asyncio
import logging
from typing import Dict, Optional

from app.core.services.market_data_service import BybitMarketDataService
from database.repositories.ohlcv_repository import OHLCVRepository

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """履歴データ収集サービス"""

    def __init__(self, market_service: Optional[BybitMarketDataService] = None):
        self.market_service = market_service or BybitMarketDataService()
        self.request_delay = 0.2  # APIレート制限対応

    async def collect_historical_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        repository: Optional[OHLCVRepository] = None,
    ) -> Dict:
        """
        指定シンボルの履歴データを包括的に収集
        """
        if not repository:
            return {"success": False, "message": "リポジトリが必要です"}

        try:
            logger.info(f"履歴データ収集開始: {symbol} {timeframe}")
            total_saved = 0
            total_fetched = 0
            max_limit = 1000
            end_timestamp = None

            for i in range(100):  # 安全のためのループ回数制限
                await asyncio.sleep(self.request_delay)

                params = {}
                if end_timestamp:
                    params["end"] = end_timestamp

                historical_data = await self.market_service.fetch_ohlcv_data(
                    symbol, timeframe, limit=max_limit, params=params
                )

                if not historical_data:
                    logger.info(f"全期間データ取得完了: バッチ{i+1}でデータ終了")
                    break

                # 最初のデータは重複している可能性があるので、最新のタイムスタンプと比較
                latest_db_ts = repository.get_latest_timestamp(symbol, timeframe)
                if latest_db_ts:
                    historical_data = [
                        d
                        for d in historical_data
                        if d[0] < latest_db_ts.timestamp() * 1000
                    ]

                if not historical_data:
                    logger.info(f"全期間データ取得完了: バッチ{i+1}で重複データのみ")
                    break

                saved_count = await self.market_service._save_ohlcv_to_database(
                    historical_data, symbol, timeframe, repository
                )
                total_saved += saved_count
                total_fetched += len(historical_data)

                end_timestamp = historical_data[0][0]

                logger.info(
                    f"バッチ {i+1}: {len(historical_data)}件取得 (次のend: {end_timestamp})"
                )

                if len(historical_data) < max_limit:
                    logger.info("全期間データ取得完了: 最終バッチ")
                    break

            logger.info(
                f"履歴データ収集完了: 取得{total_fetched}件, 保存{total_saved}件"
            )
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "fetched_count": total_fetched,
                "saved_count": total_saved,
            }

        except Exception as e:
            logger.error(f"履歴データ収集エラー: {e}")
            return {"success": False, "message": str(e)}

    async def collect_incremental_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        repository: Optional[OHLCVRepository] = None,
    ) -> Dict:
        """
        差分データを収集（最新タイムスタンプ以降）
        """
        if not repository:
            return {"success": False, "message": "リポジトリが必要です"}

        try:
            latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)
            since_ms = (
                int(latest_timestamp.timestamp() * 1000) if latest_timestamp else None
            )

            if since_ms:
                logger.info(
                    f"差分データ収集開始: {symbol} {timeframe} (since: {since_ms})"
                )
            else:
                logger.info(f"初回データ収集開始: {symbol} {timeframe}")

            ohlcv_data = await self.market_service.fetch_ohlcv_data(
                symbol, timeframe, 1000, since=since_ms
            )

            if not ohlcv_data:
                return {
                    "success": True,
                    "message": "新しいデータはありません",
                    "saved_count": 0,
                }

            saved_count = await self.market_service._save_ohlcv_to_database(
                ohlcv_data, symbol, timeframe, repository
            )
            logger.info(f"差分データ収集完了: {saved_count}件保存")
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "saved_count": saved_count,
            }

        except Exception as e:
            logger.error(f"差分データ収集エラー: {e}")
            return {"success": False, "message": str(e)}
