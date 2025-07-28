"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional
from database.repositories.funding_rate_repository import FundingRateRepository
from app.utils.data_converter import FundingRateDataConverter
from app.services.data_collection.bybit.bybit_service import BybitService
from app.services.data_collection.bybit.data_config import get_funding_rate_config

logger = logging.getLogger(__name__)


class BybitFundingRateService(BybitService):
    """Bybitファンディングレートサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()
        self.config = get_funding_rate_config()

    def _validate_parameters(self, symbol: str, limit: Optional[int] = None):
        """
        パラメータの検証（funding rate専用）

        Args:
            symbol: 取引ペアシンボル
            limit: 取得件数制限

        Raises:
            ValueError: パラメータが無効な場合
        """
        super()._validate_parameters(symbol, limit)

    async def fetch_current_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        現在のファンディングレートを取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）

        Returns:
            現在のファンディングレートデータ
        """
        normalized_symbol = self.normalize_symbol(symbol)
        return await self._handle_ccxt_errors(
            f"現在のファンディングレート取得: {normalized_symbol}",
            self.exchange.fetch_funding_rate,
            normalized_symbol,
        )

    async def fetch_funding_rate_history(
        self, symbol: str, limit: int = 100, since: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        ファンディングレート履歴を取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            limit: 取得するデータ数（1-1000）
            since: 開始タイムスタンプ（ミリ秒）

        Returns:
            ファンディングレート履歴データのリスト
        """
        self._validate_parameters(symbol, limit)
        normalized_symbol = self.normalize_symbol(symbol)
        return await self._handle_ccxt_errors(
            f"ファンディングレート履歴取得: {normalized_symbol}, limit={limit}",
            self.exchange.fetch_funding_rate_history,
            normalized_symbol,
            since,
            limit,
        )

    async def fetch_all_funding_rate_history(self, symbol: str) -> List[Dict[str, Any]]:
        """
        全期間のファンディングレート履歴を取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）

        Returns:
            全期間のファンディングレート履歴データのリスト
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
        )

    async def fetch_incremental_funding_rate_data(
        self,
        symbol: str,
        repository: Optional[FundingRateRepository] = None,
    ) -> dict:
        """
        差分ファンディングレートデータを取得してデータベースに保存

        最新のタイムスタンプ以降のデータのみを取得します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            repository: FundingRateRepository（テスト用）

        Returns:
            差分更新結果を含む辞書
        """
        return await self.fetch_incremental_data(
            symbol=symbol,
            config=self.config,
            repository=repository,
        )

    async def fetch_and_save_funding_rate_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        repository: Optional[FundingRateRepository] = None,
        fetch_all: bool = False,
    ) -> dict:
        """
        ファンディングレートデータを取得してデータベースに保存
        """
        return await self.fetch_and_save_data(
            symbol=symbol,
            config=self.config,
            limit=limit,
            repository=repository,
            fetch_all=fetch_all,
        )

    async def _save_funding_rate_to_database(
        self,
        funding_history: List[Dict[str, Any]],
        symbol: str,
        repository: FundingRateRepository,
    ) -> int:
        """
        ファンディングレートデータをデータベースに保存（内部メソッド）

        注意: このメソッドは後方互換性のために残されています。
        新しいコードでは基底クラスの_save_data_to_databaseを使用してください。
        """
        return await self._save_data_to_database(
            funding_history, symbol, repository, self.config
        )
