"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from database.repositories.open_interest_repository import OpenInterestRepository
from app.utils.data_converter import OpenInterestDataConverter
from app.services.data_collection.bybit.bybit_service import BybitService

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
        return await self._fetch_paginated_data(
            fetch_func=self.exchange.fetch_open_interest_history,
            symbol=normalized_symbol,
            page_limit=200,
            max_pages=500,
            latest_existing_timestamp=latest_timestamp,
            pagination_strategy="time_range",
            interval=interval,
        )

    async def fetch_incremental_open_interest_data(
        self,
        symbol: str,
        repository: Optional[OpenInterestRepository] = None,
        interval: str = "1h",
    ) -> dict:
        """
        差分オープンインタレストデータを取得してデータベースに保存

        最新のタイムスタンプ以降のデータのみを取得します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            repository: OpenInterestRepository（テスト用）
            interval: データ間隔（デフォルト: '1h'）

        Returns:
            差分更新結果を含む辞書
        """
        normalized_symbol = self.normalize_symbol(symbol)

        # データベースから最新タイムスタンプを取得
        latest_timestamp = await self._get_latest_timestamp_from_db(
            repository_class=OpenInterestRepository,
            get_timestamp_method_name="get_latest_open_interest_timestamp",
            symbol=normalized_symbol,
        )

        if latest_timestamp:
            logger.info(
                f"OI差分データ収集開始: {normalized_symbol} (since: {latest_timestamp})"
            )
            # 最新タイムスタンプより新しいデータを取得
            open_interest_history = await self.fetch_open_interest_history(
                symbol, limit=1000, since=latest_timestamp, interval=interval
            )

            # 重複を避けるため、最新タイムスタンプより新しいデータのみフィルタ
            open_interest_history = [
                item
                for item in open_interest_history
                if item["timestamp"] > latest_timestamp
            ]
        else:
            logger.info(f"OI初回データ収集開始: {normalized_symbol}")
            # データがない場合は最新100件を取得
            open_interest_history = await self.fetch_open_interest_history(
                symbol, limit=100, interval=interval
            )

        async def save_with_db(db, repository):
            repo = repository or OpenInterestRepository(db)
            return await self._save_open_interest_to_database(
                open_interest_history, symbol, repo
            )

        saved_count = await self._execute_with_db_session(
            func=save_with_db, repository=repository
        )

        logger.info(f"OI差分データ収集完了: {saved_count}件保存")
        return {
            "symbol": normalized_symbol,
            "fetched_count": len(open_interest_history),
            "saved_count": saved_count,
            "success": True,
            "latest_timestamp": latest_timestamp,
        }

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
