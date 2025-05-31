"""
Bybitオープンインタレストサービス

ファンディングレートサービスの実装パターンを参考に、
オープンインタレスト（建玉残高）データの取得・保存機能を提供します。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

import ccxt
from database.connection import SessionLocal

# OpenInterestDataは循環インポートを避けるため、必要時にインポート
from database.repositories.open_interest_repository import OpenInterestRepository
from app.core.utils.data_converter import OpenInterestDataConverter
from app.core.services.base_bybit_service import BaseBybitService

logger = logging.getLogger(__name__)


class BybitOpenInterestService(BaseBybitService):
    """Bybitオープンインタレストサービス"""

    def __init__(self):
        """サービスを初期化"""
        super().__init__()  # BaseBybitServiceの初期化を呼び出し

    async def fetch_current_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        現在のオープンインタレストを取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）

        Returns:
            現在のオープンインタレストデータ

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        # 基底クラスの共通エラーハンドリングを使用
        return await self._handle_ccxt_errors(
            f"現在のオープンインタレスト取得: {normalized_symbol}",
            self.exchange.fetch_open_interest,
            normalized_symbol
        )

    async def fetch_open_interest_history(
        self, symbol: str, limit: int = 100, since: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        オープンインタレスト履歴を取得

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            limit: 取得するデータ数（1-1000）
            since: 開始タイムスタンプ（ミリ秒）

        Returns:
            オープンインタレスト履歴データのリスト

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
            f"オープンインタレスト履歴取得: {normalized_symbol}, limit={limit}",
            lambda: self.exchange.fetch_open_interest_history(
                normalized_symbol, since=since, limit=limit
            )
        )

    async def fetch_all_open_interest_history(
        self, symbol: str, interval: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        全期間のオープンインタレスト履歴を取得（改善版）

        ファンディングレートサービスの実装パターンを参考に、
        ページネーション処理で全期間のデータを取得します。

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            interval: データ間隔（"5min", "15min", "30min", "1h", "4h", "1d"）

        Returns:
            全期間のオープンインタレスト履歴データのリスト

        Raises:
            ValueError: パラメータが無効な場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        # シンボルの正規化
        normalized_symbol = self.normalize_symbol(symbol)

        try:
            logger.info(
                f"全期間のオープンインタレスト履歴を取得中: {normalized_symbol}"
            )

            all_open_interest_history = []
            page_limit = 200  # Bybitの実際の制限に合わせる
            page_count = 0
            max_pages = 500  # オープンインタレストは1時間間隔なので多くのページが必要（約100,000件）

            # 最新データから開始
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            end_time = current_time
            cursor = None  # 最初のページではcursorはNone

            # 差分更新のための最新データ確認
            latest_existing_timestamp = await self._get_latest_open_interest_timestamp(
                normalized_symbol
            )

            while page_count < max_pages:
                page_count += 1

                try:
                    # ページごとにデータを取得（cursor使用）
                    open_interest_history, next_cursor = (
                        await self._fetch_open_interest_page_reverse(
                            normalized_symbol, end_time, page_limit, cursor, interval
                        )
                    )

                    if not open_interest_history:
                        logger.info(f"ページ {page_count}: データなし。取得終了")
                        break

                    logger.info(
                        f"ページ {page_count}: {len(open_interest_history)}件取得 (累計: {len(all_open_interest_history) + len(open_interest_history)}件)"
                    )

                    # 重複チェック（タイムスタンプベース）
                    existing_timestamps = {
                        item["timestamp"] for item in all_open_interest_history
                    }
                    new_items = [
                        item
                        for item in open_interest_history
                        if item["timestamp"] not in existing_timestamps
                    ]

                    # 差分更新: 既存データより古いデータのみ追加
                    if latest_existing_timestamp:
                        new_items = [
                            item
                            for item in new_items
                            if item["timestamp"] < latest_existing_timestamp
                        ]

                        if not new_items:
                            logger.info(
                                f"ページ {page_count}: 既存データに到達。差分更新完了"
                            )
                            break

                    all_open_interest_history.extend(new_items)
                    logger.info(
                        f"ページ {page_count}: 新規データ {len(new_items)}件追加 (累計: {len(all_open_interest_history)}件)"
                    )

                    # 次のページのカーソルを設定
                    if next_cursor:
                        cursor = next_cursor
                        logger.info(f"次のページカーソルを設定: {cursor}")
                    else:
                        logger.info(f"ページ {page_count}: カーソルなし。取得終了")
                        break

                    # 次のページの終了時刻を設定（最古のタイムスタンプ）
                    if open_interest_history:
                        oldest_timestamp = min(
                            item["timestamp"] for item in open_interest_history
                        )
                        end_time = oldest_timestamp - 1

                        # データが少ない場合は最後のページ
                        if len(open_interest_history) < page_limit:
                            logger.info(f"ページ {page_count}: 最後のページに到達")
                            break
                    else:
                        break

                    # レート制限対応
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"ページ {page_count} 取得エラー: {e}")
                    # 個別ページのエラーは継続
                    continue

            # データを時系列順（古い順）にソート
            all_open_interest_history.sort(key=lambda x: x["timestamp"])

            logger.info(
                f"全期間のオープンインタレスト履歴取得完了: {len(all_open_interest_history)}件 ({page_count}ページ)"
            )
            return all_open_interest_history

        except Exception as e:
            logger.error(f"全期間オープンインタレスト履歴取得エラー: {e}")
            raise

    async def _fetch_open_interest_page_reverse(
        self,
        symbol: str,
        end_time: int,
        limit: int,
        cursor: Optional[str] = None,
        interval: str = "1h",
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        逆方向ページネーションでオープンインタレスト履歴を取得

        Args:
            symbol: 正規化されたシンボル
            end_time: 終了時刻（ミリ秒）
            limit: 取得件数
            cursor: ページネーション用カーソル
            interval: データ間隔（"5min", "15min", "30min", "1h", "4h", "1d"）

        Returns:
            (オープンインタレスト履歴データのリスト, 次のページのカーソル)
        """
        try:
            logger.info(
                f"逆方向ページネーション: end_time={end_time}, cursor={cursor}, limit={limit}, interval={interval}"
            )

            # CCXTのfetch_open_interest_historyを使用
            # cursorがある場合はparamsで渡す
            if cursor:
                # cursorを使用したページネーション
                open_interest_history = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fetch_open_interest_history(
                        symbol,
                        since=None,
                        limit=limit,
                        params={"cursor": cursor, "intervalTime": interval},
                    ),
                )
            else:
                # 最初のページ（cursorなし）
                open_interest_history = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fetch_open_interest_history(
                        symbol,
                        since=None,
                        limit=limit,
                        params={"intervalTime": interval},
                    ),
                )

            logger.info(
                f"取得データ件数: {len(open_interest_history) if open_interest_history else 0}"
            )

            # CCXTの内部レスポンスからnextPageCursorを取得
            next_cursor = None
            if (
                hasattr(self.exchange, "last_json_response")
                and self.exchange.last_json_response
            ):
                result = self.exchange.last_json_response.get("result", {})
                next_cursor = result.get("nextPageCursor")
                logger.info(f"次のページカーソル: {next_cursor}")

            # end_time以前のデータのみフィルタリング
            if open_interest_history:
                original_count = len(open_interest_history)
                open_interest_history = [
                    item
                    for item in open_interest_history
                    if item["timestamp"] <= end_time
                ]
                filtered_count = len(open_interest_history)

                logger.info(
                    f"フィルタリング結果: {original_count} -> {filtered_count}件"
                )

                # 新しい順（降順）にソート
                open_interest_history.sort(key=lambda x: x["timestamp"], reverse=True)

            return open_interest_history, next_cursor

        except Exception as e:
            logger.error(f"逆方向ページネーション取得エラー: {e}")
            # フォールバック: 通常の取得方法
            fallback_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_open_interest_history(
                    symbol, since=None, limit=limit
                ),
            )
            return fallback_data, None

    async def _get_latest_open_interest_timestamp(self, symbol: str) -> Optional[int]:
        """
        データベースから最新のオープンインタレストタイムスタンプを取得

        Args:
            symbol: 正規化されたシンボル

        Returns:
            最新のタイムスタンプ（ミリ秒）、データがない場合はNone
        """
        try:
            # データベースから最新のタイムスタンプを取得
            db = SessionLocal()
            try:
                repository = OpenInterestRepository(db)
                latest_datetime = repository.get_latest_open_interest_timestamp(symbol)

                if latest_datetime:
                    # datetimeをミリ秒タイムスタンプに変換
                    return int(latest_datetime.timestamp() * 1000)
                else:
                    return None
            finally:
                db.close()
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

        Args:
            symbol: 取引ペアシンボル（例: 'BTC/USDT'）
            limit: 取得するデータ数（1-1000、fetch_all=Trueの場合は無視）
            repository: オープンインタレストリポジトリ（テスト用）
            fetch_all: 全期間のデータを取得するかどうか
            interval: データ間隔（"5min", "15min", "30min", "1h", "4h", "1d"）

        Returns:
            保存結果を含む辞書

        Raises:
            ValueError: パラメータが無効な場合
            Exception: データベースエラーの場合
        """
        try:
            # オープンインタレスト履歴を取得
            if fetch_all:
                open_interest_history = await self.fetch_all_open_interest_history(
                    symbol, interval
                )
            else:
                open_interest_history = await self.fetch_open_interest_history(
                    symbol, limit or 100
                )

            # データベースに保存
            if repository is None:
                # 実際のデータベースセッションを使用
                db = SessionLocal()
                try:
                    repository = OpenInterestRepository(db)
                    saved_count = await self._save_open_interest_to_database(
                        open_interest_history, symbol, repository
                    )
                    db.close()
                except Exception as e:
                    db.close()
                    raise
            else:
                # テスト用のリポジトリを使用
                saved_count = await self._save_open_interest_to_database(
                    open_interest_history, symbol, repository
                )

            return {
                "symbol": symbol,
                "fetched_count": len(open_interest_history),
                "saved_count": saved_count,
                "success": True,
            }

        except Exception as e:
            logger.error(f"オープンインタレストデータ取得・保存エラー: {e}")
            raise

    async def _save_open_interest_to_database(
        self,
        open_interest_history: List[Dict[str, Any]],
        symbol: str,
        repository: OpenInterestRepository,
    ) -> int:
        """
        オープンインタレストデータをデータベースに保存（内部メソッド）

        Args:
            open_interest_history: オープンインタレスト履歴データ
            symbol: 取引ペアシンボル
            repository: オープンインタレストリポジトリ

        Returns:
            保存された件数
        """
        # オープンインタレストデータを辞書形式に変換
        records = OpenInterestDataConverter.ccxt_to_db_format(
            open_interest_history,
            self.normalize_symbol(symbol)
        )

        # データベースに挿入
        return repository.insert_open_interest_data(records)
