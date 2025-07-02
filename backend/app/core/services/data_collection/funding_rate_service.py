"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional
from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.utils.data_converter import FundingRateDataConverter
from app.core.services.data_collection.bybit_service import BybitService

logger = logging.getLogger(__name__)


class BybitFundingRateService(BybitService):
    """Bybitファンディングレートサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()

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
            repository_class=FundingRateRepository,
            get_timestamp_method_name="get_latest_funding_timestamp",
            symbol=normalized_symbol,
        )
        return await self._fetch_paginated_data(
            fetch_func=self.exchange.fetch_funding_rate_history,
            symbol=normalized_symbol,
            page_limit=200,
            max_pages=50,
            latest_existing_timestamp=latest_timestamp,
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
        if fetch_all:
            funding_history = await self.fetch_all_funding_rate_history(symbol)
        else:
            funding_history = await self.fetch_funding_rate_history(
                symbol, limit or 100
            )

        async def save_with_db(db, repository):
            repo = repository or FundingRateRepository(db)
            return await self._save_funding_rate_to_database(
                funding_history, symbol, repo
            )

        saved_count = await self._execute_with_db_session(
            func=save_with_db, repository=repository
        )

        return {
            "symbol": symbol,
            "fetched_count": len(funding_history),
            "saved_count": saved_count,
            "success": True,
        }

    async def _save_funding_rate_to_database(
        self,
        funding_history: List[Dict[str, Any]],
        symbol: str,
        repository: FundingRateRepository,
    ) -> int:
        """
        ファンディングレートデータをデータベースに保存（内部メソッド）
        """
        logger.info(
            f"FRデータのDB保存開始: {len(funding_history)}件のデータを変換中..."
        )

        # データ変換
        records = FundingRateDataConverter.ccxt_to_db_format(
            funding_history, self.normalize_symbol(symbol)
        )

        logger.info(f"データ変換完了: {len(records)}件のレコードをDB挿入開始...")

        # データベース挿入（タイムアウト対応）
        try:
            inserted_count = repository.insert_funding_rate_data(records)
            logger.info(f"FRデータのDB保存完了: {inserted_count}件挿入")
            return inserted_count
        except Exception as e:
            logger.error(f"FRデータのDB保存エラー: {e}")
            raise
