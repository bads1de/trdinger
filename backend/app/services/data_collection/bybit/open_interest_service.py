"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.data_collection.bybit.bybit_service import BybitService
from app.services.data_collection.bybit.data_config import (
    get_open_interest_config,
)
from database.repositories.open_interest_repository import (
    OpenInterestRepository,
)

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BybitService):
    """Bybitオープンインタレストサービス"""

    def __init__(self):
        """
        BybitOpenInterestServiceを初期化

        Bybit取引所からのオープンインタレストデータ取得サービスを初期化します。
        親クラスの初期化を実行し、オープンインタレスト専用の設定を読み込みます。

        Note:
            このサービスはBybitServiceを継承し、
            オープンインタレスト専用の設定をself.configに保持します。
        """
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

        指定されたシンボルのオープンインタレスト履歴データをBybitから取得します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            limit: 取得するデータ数（デフォルト: 100）
            since: 開始タイムスタンプ（ミリ秒、オプション）
            interval: データ間隔（デフォルト: '1h'）

        Returns:
            List[Dict[str, Any]]: オープンインタレスト履歴データのリスト

        Raises:
            ValueError: パラメータが無効な場合
        """
        self._validate_parameters(symbol, limit)
        normalized_symbol = self._normalize_symbol_for_ccxt(symbol)
        return await self._handle_ccxt_errors(
            f"オープンインタレスト履歴取得: {normalized_symbol}, limit={limit}",
            self.exchange.fetch_open_interest_history,  # type: ignore[reportAttributeAccessIssue]
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
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
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

        指定されたシンボルのオープンインタレストデータをBybitから取得し、
        データベースに保存します。全履歴の取得または差分更新が可能です。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            limit: 取得件数制限（オプション、デフォルトは設定値）
            repository: OpenInterestRepositoryインスタンス（テスト用、オプション）
            fetch_all: 全履歴を取得するフラグ（デフォルト: False）
            interval: データ間隔（デフォルト: '1h'）

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
            intervalTime=interval,
        )
