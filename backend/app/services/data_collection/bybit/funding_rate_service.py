"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.data_collection.bybit.bybit_service import BybitService
from app.services.data_collection.bybit.data_config import get_funding_rate_config
from database.repositories.funding_rate_repository import FundingRateRepository

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

