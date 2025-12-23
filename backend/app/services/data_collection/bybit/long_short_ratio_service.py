"""
Bybit ロング/ショート比率データ収集サービス
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from app.utils.error_handler import ErrorHandler
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

from .bybit_service import BybitService

logger = logging.getLogger(__name__)


class BybitLongShortRatioService(BybitService):
    """
    Bybitからロング/ショート比率データを収集するサービスクラス
    """

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        super().__init__(exchange)

    async def fetch_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bybitからロング/ショート比率データを取得

        Args:
            symbol: 取引ペア（例: BTC/USDT:USDT）
            period: 期間（例: 5min, 1h, 1d）
            limit: 取得件数制限 (max 500)
            start_time: 開始タイムスタンプ（ミリ秒）
            end_time: 終了タイムスタンプ（ミリ秒）

        Returns:
            データリスト
        """
        # Bybitのシンボル形式に変換
        market_symbol = self._convert_to_api_symbol(symbol)

        params = {
            "category": "linear",  # 基本的にUSDT無期限を想定
            "symbol": market_symbol,
            "period": period,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        # self._handle_ccxt_errors を経由してAPIを呼び出す
        response = await self._handle_ccxt_errors(
            "Bybit V5 Market Account Ratio",
            self.exchange.publicGetV5MarketAccountRatio,
            params=params,
        )

        if (
            not response
            or "result" not in response
            or "list" not in response["result"]
        ):
            logger.warning(
                f"ロング/ショート比率データの取得に失敗またはデータなし: {symbol}"
            )
            return []

        data_list = response["result"]["list"]

        # APIは新しい順で返すことが多いが、念のため確認
        # 呼び出し元で扱いやすいように、ここでperiod情報を付与しておく
        for item in data_list:
            item["period"] = period
            # アプリケーション内での統一のため、シンボルはリクエスト時のCCXT形式（例: BTC/USDT:USDT）で上書きする
            item["symbol"] = symbol

        return data_list

    async def fetch_incremental_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        repository: LongShortRatioRepository,
    ) -> Dict[str, Any]:
        """
        ロング/ショート比率データの差分更新（最新データの取得）

        Args:
            symbol: 取引ペア
            period: 期間
            repository: LongShortRatioRepository

        Returns:
            処理結果辞書
        """
        try:
            # 最新のデータ時刻を取得
            latest_db_record = repository.get_latest_ratio(symbol, period)

            start_time = None
            if latest_db_record:
                # 最後に取得した時刻の次から取得したいが、
                # Bybitの仕様上、指定時刻を含むデータを返す可能性があるため、
                # 重複排除はリポジトリ層（INSERT OR IGNORE）に任せるか、ここでフィルタリングする。
                # 安全のため、少し前から取得して重複チェックに任せる。
                start_time = int(latest_db_record.timestamp.timestamp() * 1000)

            logger.info(
                f"LS比率差分データ収集開始: {symbol} ({period}) start_time={start_time}"
            )

            # データ取得
            # start_timeを指定すると、それ以降のデータを取得できる
            data_list = await self.fetch_long_short_ratio_data(
                symbol, period, limit=500, start_time=start_time
            )

            if not data_list:
                logger.info(f"新しいLS比率データはありません: {symbol}")
                return {
                    "symbol": symbol,
                    "saved_count": 0,
                    "success": True,
                    "latest_timestamp": (
                        latest_db_record.timestamp if latest_db_record else None
                    ),
                }

            # データ保存
            saved_count = repository.insert_long_short_ratio_data(data_list)

            logger.info(f"LS比率差分データ収集完了: {saved_count}件保存")

            # 最新のタイムスタンプを取得し直す
            new_latest_record = repository.get_latest_ratio(symbol, period)

            return {
                "symbol": symbol,
                "saved_count": saved_count,
                "success": True,
                "latest_timestamp": (
                    new_latest_record.timestamp if new_latest_record else None
                ),
            }

        except Exception as e:
            logger.error(f"LS比率差分データ収集エラー: {e}")
            return {
                "symbol": symbol,
                "saved_count": 0,
                "success": False,
                "error": str(e),
            }

    async def collect_historical_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        repository: LongShortRatioRepository,
        start_date: Optional[datetime] = None,
    ) -> int:
        """
        過去のロング/ショート比率データを収集（ページネーション）

        Args:
            symbol: 取引ペア
            period: 期間
            repository: リポジトリ
            start_date: 開始日時（指定しない場合は可能な限り過去から）

        Returns:
            保存件数
        """
        try:
            total_saved = 0
            # デフォルトで2020-10-01 (API制限の古さ)
            if not start_date:
                start_ts = 1601510400000  # 2020-10-01 UTC
            else:
                start_ts = int(start_date.timestamp() * 1000)

            current_end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

            logger.info(
                f"LS比率履歴データ収集開始: {symbol} ({period}) from {start_ts} to {current_end_ts}"
            )

            # BybitのAPIは startTime を指定すると、そこから未来に向かって limit 件取得する動作が一般的だが、
            # get_long_short_ratio_data は新しい順にリストを返すことが多い。
            # しかしドキュメントによると startTime を指定すると昇順（古い順）になるとは限らないため、
            # ループ処理には注意が必要。
            # 通常、ページネーションは「最新から過去へ」遡るか、「最古から未来へ」進むかのどちらか。
            # ここでは「期間を指定して一括取得」を繰り返す実装にする。

            # 安全のため、1日（24時間）単位などでループを回して取得する戦略を取る。
            # period='5min'の場合、1日 = 288件 < limit(500) なので1日単位なら安全。
            # period='1d'の場合、1年 = 365件 < limit(500) なので1年単位でもいけるが。

            chunk_ms = 24 * 60 * 60 * 1000  # 1日単位
            if period == "5min":
                chunk_ms = 12 * 60 * 60 * 1000  # 12時間 (144件)
            elif period == "1h":
                chunk_ms = 10 * 24 * 60 * 60 * 1000  # 10日 (240件)
            elif period == "1d":
                chunk_ms = 300 * 24 * 60 * 60 * 1000  # 300日

            current_cursor = start_ts

            while current_cursor < current_end_ts:
                next_cursor = min(current_cursor + chunk_ms, current_end_ts)

                try:
                    # データ取得
                    data_list = await self.fetch_long_short_ratio_data(
                        symbol,
                        period,
                        limit=500,
                        start_time=current_cursor,
                        end_time=next_cursor,
                    )

                    if data_list:
                        saved = repository.insert_long_short_ratio_data(data_list)
                        total_saved += saved
                        logger.info(
                            f"Chunk {current_cursor} - {next_cursor}: {len(data_list)} fetched, {saved} saved."
                        )
                except Exception as chunk_error:
                    logger.error(
                        f"Chunk {current_cursor} - {next_cursor} failed: {chunk_error}"
                    )

                current_cursor = next_cursor
                await asyncio.sleep(0.1)  # レート制限配慮

            logger.info(f"LS比率履歴データ収集完了: 合計 {total_saved} 件")
            return total_saved

        except Exception as e:
            logger.error(f"LS比率履歴データ収集エラー: {e}")
            return total_saved



