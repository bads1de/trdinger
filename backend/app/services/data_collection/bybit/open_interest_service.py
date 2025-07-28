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
from app.services.data_collection.bybit.data_config import get_open_interest_config

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BybitService):
    """Bybitオープンインタレストサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()
        self.config = get_open_interest_config()

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
            repository_class=self.config.repository_class,
            get_timestamp_method_name=self.config.get_timestamp_method_name,
            symbol=normalized_symbol,
        )
        return await self._fetch_paginated_data(
            fetch_func=getattr(self.exchange, self.config.fetch_history_method_name),
            symbol=normalized_symbol,
            page_limit=self.config.page_limit,
            max_pages=self.config.max_pages,
            latest_existing_timestamp=latest_timestamp,
            pagination_strategy=self.config.pagination_strategy,
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
        return await self.fetch_incremental_data(
            symbol=symbol,
            config=self.config,
            repository=repository,
            intervalTime=interval,
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
        return await self.fetch_and_save_data(
            symbol=symbol,
            config=self.config,
            limit=limit,
            repository=repository,
            fetch_all=fetch_all,
            intervalTime=interval,
        )

    async def _save_open_interest_to_database(
        self,
        open_interest_history: List[Dict[str, Any]],
        symbol: str,
        repository: OpenInterestRepository,
    ) -> int:
        """
        オープンインタレストデータをデータベースに保存（内部メソッド）

        注意: このメソッドは後方互換性のために残されています。
        新しいコードでは基底クラスの_save_data_to_databaseを使用してください。
        """
        return await self._save_data_to_database(
            open_interest_history, symbol, repository, self.config
        )
