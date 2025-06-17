"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.data_converter import OpenInterestDataConverter
from app.core.services.base_bybit_service import BaseBybitService

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BaseBybitService):
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
        latest_timestamp = await self._get_latest_open_interest_timestamp(
            normalized_symbol
        )
        return await self._fetch_paginated_data(
            fetch_func=self.exchange.fetch_open_interest_history,
            symbol=normalized_symbol,
            page_limit=200,
            max_pages=500,
            latest_existing_timestamp=latest_timestamp,
            fetch_kwargs={"params": {"intervalTime": interval}},
        )

    async def _get_latest_open_interest_timestamp(self, symbol: str) -> Optional[int]:
        """
        データベースから最新のオープンインタレストタイムスタンプを取得
        """

        async def get_timestamp(db, **kwargs):
            repo = OpenInterestRepository(db)
            latest_datetime = repo.get_latest_open_interest_timestamp(symbol)
            if latest_datetime:
                return int(latest_datetime.timestamp() * 1000)
            return None

        try:
            return await self._execute_with_db_session(
                func=get_timestamp, repository=None
            )
        except Exception as e:
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            return None

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
