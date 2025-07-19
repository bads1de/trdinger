"""
履歴データ収集サービス

バックテスト用の包括的なOHLCVデータ収集を行います。
"""

import asyncio
import logging
import ccxt
from typing import Optional
from ..bybit.market_data_service import BybitMarketDataService
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.open_interest_repository import OpenInterestRepository

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
    ) -> int:
        """
        指定シンボルの履歴データを包括的に収集

        Returns:
            保存された件数

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
            RuntimeError: その他の予期せぬエラー
        """
        if not repository:
            raise ValueError("リポジトリが必要です")

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
            return total_saved

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボルによる履歴データ収集エラー: {symbol} - {e}")
            raise
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラーによる履歴データ収集エラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラーによる履歴データ収集エラー: {e}")
            raise
        except ValueError as e:
            logger.error(f"パラメータ検証エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しない履歴データ収集エラー: {e}", exc_info=True)
            raise RuntimeError(
                f"履歴データ収集中に予期しないエラーが発生しました: {e}"
            ) from e

    async def collect_bulk_incremental_data(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        ohlcv_repository: Optional[OHLCVRepository] = None,
        funding_rate_repository: Optional[FundingRateRepository] = None,
        open_interest_repository: Optional[OpenInterestRepository] = None,
        include_external_market: bool = True,
    ) -> dict:
        """
        一括差分データを収集（OHLCV、FR、OI、外部市場データ）

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            timeframe: 時間軸（例: '1h'）
            ohlcv_repository: OHLCVRepository（テスト用）
            funding_rate_repository: FundingRateRepository（テスト用）
            open_interest_repository: OpenInterestRepository（テスト用）
            include_external_market: 外部市場データも収集するかどうか

        Returns:
            一括差分更新結果を含む辞書
        """
        from ..bybit.funding_rate_service import BybitFundingRateService
        from ..bybit.open_interest_service import BybitOpenInterestService

        logger.info(f"一括差分データ収集開始: {symbol} {timeframe}")

        # 全ての時間足を定義
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        logger.info(f"処理対象時間足: {timeframes}")

        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "success": True,
            "data": {},
            "total_saved_count": 0,
            "errors": [],
        }

        # 1. OHLCVデータの差分更新（全時間足）
        ohlcv_results = {}
        total_ohlcv_saved = 0

        if ohlcv_repository:
            for tf in timeframes:
                try:
                    logger.info(f"OHLCV差分データ収集開始: {symbol} {tf}")

                    # 最新タイムスタンプを取得
                    latest_timestamp = ohlcv_repository.get_latest_timestamp(symbol, tf)
                    since_ms = (
                        int(latest_timestamp.timestamp() * 1000)
                        if latest_timestamp
                        else None
                    )

                    if since_ms:
                        logger.info(
                            f"差分データ収集: {symbol} {tf} (since: {since_ms})"
                        )
                    else:
                        logger.info(f"初回データ収集: {symbol} {tf}")

                    # OHLCVデータを取得
                    ohlcv_data = await self.market_service.fetch_ohlcv_data(
                        symbol, tf, 1000, since=since_ms
                    )

                    if not ohlcv_data:
                        logger.info(f"新しいデータはありません: {symbol} {tf}")
                        ohlcv_result = 0
                    else:
                        # データベースに保存
                        ohlcv_result = (
                            await self.market_service._save_ohlcv_to_database(
                                ohlcv_data, symbol, tf, ohlcv_repository
                            )
                        )

                    ohlcv_results[tf] = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "saved_count": ohlcv_result,
                        "success": True,
                    }
                    total_ohlcv_saved += ohlcv_result
                    logger.info(
                        f"OHLCV差分データ収集完了: {symbol} {tf} - {ohlcv_result}件保存"
                    )

                    # API制限を回避するため少し待機
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"OHLCV差分データ収集エラー: {symbol} {tf} - {e}")
                    ohlcv_results[tf] = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "saved_count": 0,
                        "success": False,
                        "error": str(e),
                    }
                    results["errors"].append(f"OHLCV {tf}: {str(e)}")
        else:
            logger.info(
                "OHLCVリポジトリが提供されていないため、OHLCVデータ収集をスキップします。"
            )

        # OHLCV結果の集計
        successful_timeframes = [r for r in ohlcv_results.values() if r["success"]]
        logger.info(
            f"OHLCV処理完了: 総保存件数={total_ohlcv_saved}, 成功時間足数={len(successful_timeframes)}/{len(timeframes)}"
        )
        logger.info(f"時間足別結果: {ohlcv_results}")

        results["data"]["ohlcv"] = {
            "symbol": symbol,
            "timeframe": "all",  # 全時間足を示す
            "saved_count": total_ohlcv_saved,
            "success": len(successful_timeframes) > 0,
            "timeframe_results": ohlcv_results,
        }
        results["total_saved_count"] += total_ohlcv_saved

        # 少し待機してAPI制限を回避
        await asyncio.sleep(0.2)

        # 2. ファンディングレートデータの差分更新
        if funding_rate_repository:
            try:
                logger.info(f"FR差分データ収集開始: {symbol}")
                funding_rate_service = BybitFundingRateService()
                fr_result = (
                    await funding_rate_service.fetch_incremental_funding_rate_data(
                        symbol, funding_rate_repository
                    )
                )

                results["data"]["funding_rate"] = {
                    "symbol": fr_result["symbol"],
                    "saved_count": fr_result["saved_count"],
                    "success": fr_result["success"],
                    "latest_timestamp": fr_result.get("latest_timestamp"),
                }
                results["total_saved_count"] += fr_result["saved_count"]
                logger.info(f"FR差分データ収集完了: {fr_result['saved_count']}件保存")

            except Exception as e:
                logger.error(f"FR差分データ収集エラー: {e}")
                results["data"]["funding_rate"] = {
                    "symbol": symbol,
                    "saved_count": 0,
                    "success": False,
                    "error": str(e),
                }
                results["errors"].append(f"FR: {str(e)}")
        else:
            logger.info(
                "ファンディングレートリポジトリが提供されていないため、FRデータ収集をスキップします。"
            )
            results["data"]["funding_rate"] = {
                "symbol": symbol,
                "saved_count": 0,
                "success": True,
                "error": "Repository not provided",
            }

        # 少し待機してAPI制限を回避
        await asyncio.sleep(0.2)

        # 3. オープンインタレストデータの差分更新
        if open_interest_repository:
            try:
                logger.info(f"OI差分データ収集開始: {symbol}")
                open_interest_service = BybitOpenInterestService()
                oi_result = (
                    await open_interest_service.fetch_incremental_open_interest_data(
                        symbol, open_interest_repository
                    )
                )

                results["data"]["open_interest"] = {
                    "symbol": oi_result["symbol"],
                    "saved_count": oi_result["saved_count"],
                    "success": oi_result["success"],
                    "latest_timestamp": oi_result.get("latest_timestamp"),
                }
                results["total_saved_count"] += oi_result["saved_count"]
                logger.info(f"OI差分データ収集完了: {oi_result['saved_count']}件保存")

            except Exception as e:
                logger.error(f"OI差分データ収集エラー: {e}")
                results["data"]["open_interest"] = {
                    "symbol": symbol,
                    "saved_count": 0,
                    "success": False,
                    "error": str(e),
                }
                results["errors"].append(f"OI: {str(e)}")
        else:
            logger.info(
                "オープンインタレストリポジトリが提供されていないため、OIデータ収集をスキップします。"
            )
            results["data"]["open_interest"] = {
                "symbol": symbol,
                "saved_count": 0,
                "success": True,
                "error": "Repository not provided",
            }

        # 少し待機してAPI制限を回避
        await asyncio.sleep(0.2)

        # 4. 外部市場データの差分更新
        if include_external_market:
            try:
                logger.info("外部市場データ差分収集開始")
                from ......data_collector.external_market_collector import (
                    ExternalMarketDataCollector,
                )

                async with ExternalMarketDataCollector() as collector:
                    em_result = (
                        await collector.collect_incremental_external_market_data()
                    )

                results["data"]["external_market"] = {
                    "fetched_count": em_result["fetched_count"],
                    "inserted_count": em_result["inserted_count"],
                    "success": em_result["success"],
                    "collection_type": em_result.get("collection_type", "incremental"),
                }
                results["total_saved_count"] += em_result["inserted_count"]
                logger.info(
                    f"外部市場データ差分収集完了: {em_result['inserted_count']}件保存"
                )

            except Exception as e:
                logger.error(f"外部市場データ差分収集エラー: {e}")
                results["data"]["external_market"] = {
                    "fetched_count": 0,
                    "inserted_count": 0,
                    "success": False,
                    "error": str(e),
                }
                results["errors"].append(f"外部市場データ: {str(e)}")
        else:
            logger.info("外部市場データ収集をスキップします。")
            results["data"]["external_market"] = {
                "fetched_count": 0,
                "inserted_count": 0,
                "success": True,
                "error": "Skipped",
            }

        # 全体の成功判定
        if results["errors"]:
            results["success"] = False
            logger.warning(f"一括差分データ収集で一部エラー: {results['errors']}")

        logger.info(
            f"一括差分データ収集完了: {symbol} {timeframe} - "
            f"総保存件数: {results['total_saved_count']}件"
        )

        return results
