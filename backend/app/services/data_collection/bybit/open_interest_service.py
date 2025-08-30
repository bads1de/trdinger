"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.data_collection.bybit.bybit_service import BybitService
from app.services.data_collection.bybit.data_config import get_open_interest_config
from database.repositories.open_interest_repository import OpenInterestRepository
from app.utils.normalization_service import SymbolNormalizationService

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BybitService):
    """Bybitオープンインタレストサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()
        self.config = get_open_interest_config()

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
        normalized_symbol = SymbolNormalizationService.normalize_symbol(symbol, "bybit")
        return await self._handle_ccxt_errors(
            f"オープンインタレスト履歴取得: {normalized_symbol}, limit={limit}",
            self.exchange.fetch_open_interest_history,
            normalized_symbol,
            interval,  # timeframeパラメータを追加
            since,
            limit,
            {"intervalTime": interval},
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
