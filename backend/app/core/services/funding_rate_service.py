"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import asyncio
import ccxt
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Callable
import logging

from database.connection import SessionLocal
from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.utils.data_converter import FundingRateDataConverter
from app.core.services.base_bybit_service import BaseBybitService

logger = logging.getLogger(__name__)


class BybitFundingRateService(BaseBybitService):
    """Bybitファンディングレートサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()  # BaseBybitServiceの初期化を呼び出し

    def _validate_parameters(self, symbol: str, limit: int):
        """
        パラメータの検証（funding rate専用）

        Args:
            symbol: 取引ペアシンボル
            limit: 取得件数制限

        Raises:
            ValueError: パラメータが無効な場合
        """
        # 基底クラスの検証を使用
        super()._validate_parameters(symbol, limit)

    async def fetch_current_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        現在のファンディングレートを取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）

        Returns:
            現在のファンディングレートデータ

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        # 基底クラスの共通エラーハンドリングを使用
        return await self._handle_ccxt_errors(
            f"現在のファンディングレート取得: {normalized_symbol}",
            self.exchange.fetch_funding_rate,
            normalized_symbol
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

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # パラメータの検証
        self._validate_parameters(symbol, limit)

        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        # 基底クラスの共通エラーハンドリングを使用
        return await self._handle_ccxt_errors(
            f"ファンディングレート履歴取得: {normalized_symbol}, limit={limit}",
            self.exchange.fetch_funding_rate_history,
            normalized_symbol,
            since,
            limit
        )

    async def fetch_all_funding_rate_history(self, symbol: str) -> List[Dict[str, Any]]:
        """
        全期間のファンディングレート履歴を取得（改善版）

        Bybit APIの200件制限を回避するため、逆方向ページネーションを使用します。
        最新データから過去に向かって取得し、差分更新にも対応します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）

        Returns:
            全期間のファンディングレート履歴データのリスト

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        # 差分更新のための最新データ確認
        latest_existing_timestamp = await self._get_latest_funding_rate_timestamp(
            normalized_symbol
        )

        # 基底クラスのページネーション機能を使用
        return await self._fetch_paginated_data(
            fetch_func=self.exchange.fetch_funding_rate_history,
            symbol=normalized_symbol,
            page_limit=200,  # Bybitの実際の制限に合わせる
            max_pages=50,    # 安全のための上限（約10,000件）
            latest_existing_timestamp=latest_existing_timestamp
        )

    async def _fetch_page_data(
        self,
        fetch_func: Callable,
        symbol: str,
        end_time: int,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Funding Rate専用のページデータ取得（基底クラスをオーバーライド）

        Args:
            fetch_func: データ取得関数
            symbol: 正規化されたシンボル
            end_time: 終了時刻（ミリ秒）
            limit: 取得件数
            **kwargs: 追加引数

        Returns:
            1ページ分のデータリスト
        """
        try:
            # 8時間間隔でファンディングレートが設定されるため、
            # limit * 8時間前からend_timeまでの範囲を指定
            hours_back = limit * 8  # 8時間間隔
            since_time = end_time - (hours_back * 60 * 60 * 1000)  # ミリ秒

            # 通常のfetch_funding_rate_historyを使用
            funding_history = await self._handle_ccxt_errors(
                f"ページデータ取得 (limit={limit})",
                fetch_func,
                symbol,
                since_time,
                limit
            )

            # end_time以前のデータのみフィルタリング
            if funding_history:
                funding_history = [
                    item for item in funding_history if item["timestamp"] <= end_time
                ]

                # 新しい順（降順）にソート
                funding_history.sort(key=lambda x: x["timestamp"], reverse=True)

            return funding_history

        except Exception as e:
            logger.error(f"Funding Rate ページデータ取得エラー: {e}")
            # フォールバック: 通常の取得方法
            return await self._handle_ccxt_errors(
                f"フォールバック取得 (limit={limit})",
                fetch_func,
                symbol,
                None,
                limit
            )



    async def _get_latest_funding_rate_timestamp(self, symbol: str) -> Optional[int]:
        """
        データベースから最新のファンディングレートタイムスタンプを取得

        Args:
            symbol: 正規化されたシンボル

        Returns:
            最新のタイムスタンプ（ミリ秒）、データがない場合はNone
        """
        try:
            # データベースから最新のタイムスタンプを取得
            # 実装は後で追加（現在はNoneを返す）
            return None
        except Exception as e:
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            return None

    async def fetch_and_save_funding_rate_data(
        self,
        symbol: str,
        limit: Optional[int] = None,
        repository: Optional[FundingRateRepository] = None,
        fetch_all: bool = False,
    ) -> dict:
        """
        ファンディングレートデータを取得してデータベースに保存

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            limit: 取得するデータ数（1-1000、fetch_all=Trueの場合は無視）
            repository: ファンディングレートリポジトリ（テスト用）
            fetch_all: 全期間のデータを取得するかどうか

        Returns:
            保存結果を含む辞書

        Raises:
            ValueError: パラメータが無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # ファンディングレート履歴を取得
            if fetch_all:
                funding_history = await self.fetch_all_funding_rate_history(symbol)
            else:
                funding_history = await self.fetch_funding_rate_history(
                    symbol, limit or 100
                )

            # データベースに保存
            if repository is None:
                # 基底クラスのデータベースセッション管理を使用
                async def save_with_db(db):
                    repository = FundingRateRepository(db)
                    return await self._save_funding_rate_to_database(
                        funding_history, symbol, repository
                    )

                saved_count = await self._execute_with_db_session(save_with_db)
            else:
                # テスト用のリポジトリを使用
                saved_count = await self._save_funding_rate_to_database(
                    funding_history, symbol, repository
                )

            return {
                "symbol": symbol,
                "fetched_count": len(funding_history),
                "saved_count": saved_count,
                "success": True,
            }

        except Exception as e:
            logger.error(f"ファンディングレートデータ取得・保存エラー: {e}")
            raise

    async def _save_funding_rate_to_database(
        self,
        funding_history: List[Dict[str, Any]],
        symbol: str,
        repository: FundingRateRepository,
    ) -> int:
        """
        ファンディングレートデータをデータベースに保存（内部メソッド）

        Args:
            funding_history: ファンディングレート履歴データ
            symbol: 取引ペアシンボル
            repository: ファンディングレートリポジトリ

        Returns:
            保存された件数
        """
        # ファンディングレートデータを辞書形式に変換
        records = FundingRateDataConverter.ccxt_to_db_format(
            funding_history,
            self.normalize_symbol(symbol)
        )

        # データベースに挿入
        return repository.insert_funding_rate_data(records)
