"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.data_converter import OpenInterestDataConverter
from app.core.services.bybit_service import BybitService

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BybitService):
    """Bybitオープンインタレストサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()

    async def fetch_current_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        現在のオープンインタレストを取得
        """
        normalized_symbol = self.normalize_symbol(symbol)
        return await self._handle_ccxt_errors(
            f"現在のオープンインタレスト取得: {normalized_symbol}",
            self.exchange.fetch_open_interest,
            normalized_symbol,
        )

    async def fetch_open_interest_history(
        self,
        symbol: str,
        limit: int = 100,
        since: Optional[int] = None,
        interval: str = "1h",
    ) -> List[Dict[str, Any]]:
        """
        オープンインタレスト履歴を取得
        """
        self._validate_parameters(symbol, limit)
        normalized_symbol = self.normalize_symbol(symbol)
        return await self._handle_ccxt_errors(
            f"オープンインタレスト履歴取得: {normalized_symbol}, limit={limit}",
            self.exchange.fetch_open_interest_history,
            normalized_symbol,
            since=since,
            limit=limit,
            params={"intervalTime": interval},
        )

    async def fetch_all_open_interest_history(
        self, symbol: str, interval: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        全期間のオープンインタレスト履歴を取得
        """
        normalized_symbol = self.normalize_symbol(symbol)
        latest_timestamp = await self._get_latest_timestamp_from_db(
            repository_class=OpenInterestRepository,
            get_timestamp_method_name="get_latest_open_interest_timestamp",
            symbol=normalized_symbol,
        )
        return await self._fetch_paginated_open_interest_data(
            symbol=normalized_symbol,
            interval=interval,
            page_limit=200,
            max_pages=500,
            latest_existing_timestamp=latest_timestamp,
        )

    async def fetch_and_save_open_interest_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        repository: Optional[OpenInterestRepository] = None,
        fetch_all: bool = False,
        interval: str = "1h",
    ) -> dict:
        """
        オープンインタレストデータを取得してデータベースに保存
        """
        if fetch_all:
            open_interest_history = await self.fetch_all_open_interest_history(
                symbol, interval
            )
        else:
            open_interest_history = await self.fetch_open_interest_history(
                symbol, limit or 100, interval=interval
            )

        async def save_with_db(db, repository):
            repo = repository or OpenInterestRepository(db)
            return await self._save_open_interest_to_database(
                open_interest_history, symbol, repo
            )

        saved_count = await self._execute_with_db_session(
            func=save_with_db, repository=repository
        )

        return {
            "symbol": symbol,
            "fetched_count": len(open_interest_history),
            "saved_count": saved_count,
            "success": True,
        }

    async def _fetch_paginated_open_interest_data(
        self,
        symbol: str,
        interval: str = "1h",
        page_limit: int = 200,
        max_pages: int = 500,
        latest_existing_timestamp: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        オープンインタレスト専用のページネーション処理で全期間データを取得

        CCXTはBybitのcursorベースのページネーションを提供しないため、
        時間ベースのページネーションを使用します。

        Args:
            symbol: 正規化されたシンボル
            interval: 時間間隔（5min, 15min, 30min, 1h, 4h, 1d）
            page_limit: 1ページあたりの取得件数（最大200）
            max_pages: 最大ページ数
            latest_existing_timestamp: 既存データの最新タイムスタンプ（差分更新用）

        Returns:
            全期間のオープンインタレストデータリスト
        """
        logger.info(
            f"オープンインタレスト全期間データ取得開始: {symbol}, interval={interval}"
        )

        all_data = []
        page_count = 0

        # カテゴリを決定（USDTペアはlinear、USDペアはinverse）
        category = "linear" if symbol.endswith("USDT") else "inverse"

        # オープンインタレスト履歴API用のシンボル形式に変換
        # BTC/USDT:USDT -> BTCUSDT
        api_symbol = symbol.replace("/", "").replace(":USDT", "")

        # 時間ベースのページネーション用
        from datetime import datetime, timezone

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        # 間隔に応じた時間差を計算（ミリ秒）
        interval_ms = {
            "5min": 5 * 60 * 1000,
            "15min": 15 * 60 * 1000,
            "30min": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }.get(
            interval, 60 * 60 * 1000
        )  # デフォルトは1時間

        while page_count < max_pages:
            page_count += 1

            try:
                # 時間範囲を計算（1ページあたりの時間範囲）
                start_time = end_time - (page_limit * interval_ms)

                # パラメータを構築
                params = {
                    "category": category,
                    "symbol": api_symbol,
                    "intervalTime": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": page_limit,
                }

                logger.debug(f"ページ {page_count}: パラメータ {params}")

                # ページごとにデータを取得
                # CCXTのfetch_open_interest_historyは引数の順序が異なる
                page_data = await self._handle_ccxt_errors(
                    f"オープンインタレストページデータ取得 (page={page_count})",
                    self.exchange.fetch_open_interest_history,
                    api_symbol,
                    since=start_time,
                    limit=page_limit,
                    params=params,
                )

                if not page_data:
                    logger.info(f"ページ {page_count}: データなし。取得終了")
                    break

                logger.info(
                    f"ページ {page_count}: {len(page_data)}件取得 "
                    f"(累計: {len(all_data) + len(page_data)}件)"
                )

                # 重複チェック（タイムスタンプベース）
                existing_timestamps = {item["timestamp"] for item in all_data}
                new_items = [
                    item
                    for item in page_data
                    if item["timestamp"] not in existing_timestamps
                ]

                # 差分更新: 既存データより古いデータのみ追加
                if latest_existing_timestamp:
                    new_items = [
                        item
                        for item in new_items
                        if item["timestamp"] < latest_existing_timestamp
                    ]

                    if not new_items:
                        logger.info(
                            f"ページ {page_count}: 既存データに到達。差分更新完了"
                        )
                        break

                all_data.extend(new_items)
                logger.info(
                    f"ページ {page_count}: 新規データ {len(new_items)}件追加 "
                    f"(累計: {len(all_data)}件)"
                )

                # 次のページの時間範囲を設定
                # 取得件数が期待値より少ない場合は最後のページ
                if len(page_data) < page_limit:
                    logger.info(
                        f"ページ {page_count}: 最後のページに到達（件数不足: {len(page_data)} < {page_limit}）"
                    )
                    break

                # 次のページの終了時刻を現在のページの開始時刻に設定
                end_time = start_time - 1  # 1ミリ秒前から開始

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"ページ {page_count} 取得エラー: {e}")
                # cursorベースのページネーションでエラーが発生した場合は終了
                break

        logger.info(
            f"オープンインタレスト全期間データ取得完了: {symbol}, 総件数: {len(all_data)}件"
        )
        return all_data

    async def _save_open_interest_to_database(
        self,
        open_interest_history: List[Dict[str, Any]],
        symbol: str,
        repository: OpenInterestRepository,
    ) -> int:
        """
        オープンインタレストデータをデータベースに保存（内部メソッド）
        """
        records = OpenInterestDataConverter.ccxt_to_db_format(
            open_interest_history, self.normalize_symbol(symbol)
        )
        return repository.insert_open_interest_data(records)
