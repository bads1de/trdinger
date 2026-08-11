"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import logging
from typing import Any, cast

from app.services.data_collection.bybit.bybit_service import BybitService
from app.services.data_collection.bybit.data_config import (
    get_funding_rate_config,
)
from app.utils.data_conversion import normalize_market_symbol
from database.repositories.funding_rate_repository import FundingRateRepository

logger = logging.getLogger(__name__)


class BybitFundingRateService(BybitService):
    """Bybitファンディングレートサービス"""

    def __init__(self) -> None:
        """
        BybitFundingRateServiceを初期化

        Bybit取引所からのファンディングレートデータ取得サービスを初期化します。
        親クラスの初期化を実行し、ファンディングレート専用の設定を読み込みます。

        Note:
            このサービスはBybitServiceを継承し、
            ファンディングレート専用の設定をself.configに保持します。
        """
        super().__init__()
        self.config = get_funding_rate_config()

    def _validate_parameters(self, symbol: str, limit: int | None = None) -> None:
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
        self, symbol: str, limit: int = 100, since: int | None = None
    ) -> list[dict[str, Any]]:
        """
        ファンディングレート履歴を取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            limit: 取得するデータ数（1-1000）
            since: 開始タイムスタンプ（ミリ秒）

        Returns:
            ファンディングレート履歴データのリスト
        """
        self._validate_parameters(symbol, limit)
        normalized_symbol = normalize_market_symbol(symbol)
        return cast(
            list[dict[str, Any]],
            await self._handle_ccxt_errors(
                f"ファンディングレート履歴取得: {normalized_symbol}, limit={limit}",
                self.exchange.fetch_funding_rate_history,
                normalized_symbol,
                since,
                limit,
            ),
        )

    async def fetch_all_funding_rate_history(self, symbol: str) -> list[dict[str, Any]]:
        """
        全期間のファンディングレート履歴を取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）

        Returns:
            全期間のファンディングレート履歴データのリスト
        """
        normalized_symbol = normalize_market_symbol(symbol)
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
        repository: FundingRateRepository | None = None,
    ) -> dict:
        """
        差分ファンディングレートデータを取得してデータベースに保存

        最新のタイムスタンプ以降のデータのみを取得します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
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
        limit: int | None = None,
        repository: FundingRateRepository | None = None,
        fetch_all: bool = False,
    ) -> dict:
        """
        ファンディングレートデータを取得してデータベースに保存

        指定されたシンボルのファンディングレートデータをBybitから取得し、
        データベースに保存します。全履歴の取得または差分更新が可能です。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            limit: 取得件数制限（オプション、デフォルトは設定値）
            repository: FundingRateRepositoryインスタンス（テスト用、オプション）
            fetch_all: 全履歴を取得するフラグ（デフォルト: False）

        Returns:
            dict: 取得・保存結果を含む辞書。
                  以下のキーを含みます：
                  - fetched_count: 取得件数
                  - saved_count: 保存件数
                  - skipped_count: 重複スキップ件数
                  - error_count: エラー件数
                  - message: 処理結果メッセージ

        Note:
            fetch_all=Trueの場合は全履歴を取得し、
            fetch_all=Falseの場合は差分更新を実行します。
        """
        return await self.fetch_and_save_data(
            symbol=symbol,
            config=self.config,
            limit=limit,
            repository=repository,
            fetch_all=fetch_all,
        )
