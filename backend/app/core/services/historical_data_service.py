"""
履歴データ収集サービス

バックテスト用の包括的なOHLCVデータ収集を行います。
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from app.core.services.market_data_service import BybitMarketDataService
from database.repository import OHLCVRepository

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """履歴データ収集サービス"""

    def __init__(self, market_service: Optional[BybitMarketDataService] = None):
        self.market_service = market_service or BybitMarketDataService()

        # ビットコインの対応時間軸
        self.timeframes = ["15m", "30m", "1h", "4h", "1d"]

        # APIレート制限対応（リクエスト間隔）
        self.request_delay = 0.1  # 100ms

    async def collect_historical_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        repository: Optional[OHLCVRepository] = None,
    ) -> Dict:
        """
        指定シンボルの履歴データを包括的に収集

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            repository: データベースリポジトリ

        Returns:
            収集結果
        """
        try:
            logger.info(f"履歴データ収集開始: {symbol} {timeframe}")

            total_saved = 0
            total_fetched = 0

            # 複数回に分けてデータを取得（APIの制限対応）
            max_limit = 1000  # Bybitの最大取得件数

            # 最初のデータ取得
            ohlcv_data = await self.market_service.fetch_ohlcv_data(
                symbol, timeframe, max_limit
            )

            if not ohlcv_data:
                return {"success": False, "message": "データが取得できませんでした"}

            # データベースに保存
            if repository:
                saved_count = await self._save_ohlcv_to_database(
                    ohlcv_data, symbol, timeframe, repository
                )
                total_saved += saved_count

            total_fetched += len(ohlcv_data)

            # 過去のデータを段階的に取得
            oldest_timestamp = min(candle[0] for candle in ohlcv_data)

            # 最大10回まで過去データを取得
            for i in range(10):
                await asyncio.sleep(self.request_delay)

                # より古いデータを取得
                since_timestamp = oldest_timestamp - (
                    max_limit * self._get_timeframe_ms(timeframe)
                )

                try:
                    historical_data = await self._fetch_historical_batch(
                        symbol, timeframe, since_timestamp, max_limit
                    )

                    if not historical_data or len(historical_data) < 100:
                        # データが少なくなったら終了
                        break

                    if repository:
                        saved_count = await self._save_ohlcv_to_database(
                            historical_data, symbol, timeframe, repository
                        )
                        total_saved += saved_count

                    total_fetched += len(historical_data)
                    oldest_timestamp = min(candle[0] for candle in historical_data)

                    logger.info(f"バッチ {i+1}: {len(historical_data)}件取得")

                except Exception as e:
                    logger.warning(f"バッチ {i+1} でエラー: {e}")
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

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            repository: データベースリポジトリ

        Returns:
            収集結果
        """
        try:
            if not repository:
                return {"success": False, "message": "リポジトリが必要です"}

            # 最新タイムスタンプを取得
            latest_timestamp = repository.get_latest_timestamp(symbol, timeframe)

            if latest_timestamp:
                logger.info(f"最新データ: {latest_timestamp}")
                # 最新タイムスタンプ以降のデータを取得
                since_ms = int(latest_timestamp.timestamp() * 1000)
            else:
                # データがない場合は直近100件を取得
                since_ms = None
                logger.info("初回データ取得")

            # 新しいデータを取得
            ohlcv_data = await self.market_service.fetch_ohlcv_data(
                symbol, timeframe, 1000
            )

            if not ohlcv_data:
                return {
                    "success": True,
                    "message": "新しいデータはありません",
                    "saved_count": 0,
                }

            # 重複を除外（最新タイムスタンプより新しいもののみ）
            if latest_timestamp:
                latest_ms = int(latest_timestamp.timestamp() * 1000)
                ohlcv_data = [candle for candle in ohlcv_data if candle[0] > latest_ms]

            if not ohlcv_data:
                return {
                    "success": True,
                    "message": "新しいデータはありません",
                    "saved_count": 0,
                }

            # データベースに保存
            saved_count = await self._save_ohlcv_to_database(
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

    async def _fetch_historical_batch(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> List:
        """履歴データのバッチ取得"""
        try:
            # CCXTのfetch_ohlcvを直接使用してsinceパラメータを指定
            import asyncio

            ohlcv_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.market_service.exchange.fetch_ohlcv,
                self.market_service.normalize_symbol(symbol),
                timeframe,
                since,
                limit,
            )
            return ohlcv_data
        except Exception as e:
            logger.error(f"履歴バッチ取得エラー: {e}")
            return []

    async def _save_ohlcv_to_database(
        self, ohlcv_data: List, symbol: str, timeframe: str, repository: OHLCVRepository
    ) -> int:
        """OHLCVデータをデータベースに保存"""
        try:
            # データを変換
            records = []
            for candle in ohlcv_data:
                timestamp, open_price, high, low, close, volume = candle
                records.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
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

            # 一括挿入
            return repository.insert_ohlcv_data(records)

        except Exception as e:
            logger.error(f"データベース保存エラー: {e}")
            return 0

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """時間軸をミリ秒に変換"""
        timeframe_ms = {
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return timeframe_ms.get(timeframe, 60 * 60 * 1000)
