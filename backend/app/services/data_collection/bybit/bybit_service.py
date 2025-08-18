"""
Bybitサービスの基底クラス

"""

import asyncio
import logging
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import ccxt

from app.utils.error_handler import (
    DataError,
    ErrorHandler,
)
from database.connection import get_db

logger = logging.getLogger(__name__)


class BybitService(ABC):
    """Bybitサービスの基底クラス"""

    def __init__(self):
        """サービスを初期化"""
        self.exchange = ccxt.bybit(
            {
                "sandbox": False,  # 本番環境を使用（読み取り専用）
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",  # 無期限契約市場を使用
                },
            }
        )

    def normalize_symbol(self, symbol: str) -> str:
        """
        シンボルを正規化（無期限契約形式に変換）

        Args:
            symbol: 入力シンボル（例: 'BTC/USDT' または 'BTC/USDT:USDT'）

        Returns:
            正規化されたシンボル（例: 'BTC/USDT:USDT'）
        """
        # 既に無期限契約形式の場合はそのまま返す
        if ":" in symbol:
            return symbol

        # スポット形式を無期限契約形式に変換
        if symbol.endswith("/USDT"):
            return f"{symbol}:USDT"
        elif symbol.endswith("/USD"):
            return f"{symbol}:USD"
        else:
            # デフォルトはUSDT無期限契約
            return f"{symbol}:USDT"

    def _validate_symbol(self, symbol: str) -> None:
        """
        シンボルの検証

        Args:
            symbol: 取引ペアシンボル

        Raises:
            ValueError: シンボルが無効な場合
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("シンボルは有効な文字列である必要があります")

    def _validate_limit(self, limit: Optional[int] = None) -> None:
        """
        取得件数制限の検証

        Args:
            limit: 取得件数制限

        Raises:
            ValueError: limitが無効な場合
        """
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0 or limit > 1000:
                raise ValueError("limitは1から1000の間の整数である必要があります")

    def _validate_parameters(self, symbol: str, limit: Optional[int] = None) -> None:
        """
        パラメータの検証

        Args:
            symbol: 取引ペアシンボル
            limit: 取得件数制限

        Raises:
            ValueError: パラメータが無効な場合
        """
        self._validate_symbol(symbol)
        self._validate_limit(limit)

    async def _handle_ccxt_errors(
        self, operation_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """
        CCXT操作の共通エラーハンドリング

        Args:
            operation_name: 操作名（ログ用）
            func: 実行する関数
            **args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の実行結果

        Raises:
            ccxt.BadSymbol: 無効なシンボルの場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        return await ErrorHandler.safe_execute(
            func=lambda: self._handle_ccxt_errors_impl(
                operation_name, func, *args, **kwargs
            ),
            default_return=None,
            error_message=f"CCXT操作: {operation_name}",
            log_level="error",
            is_api_call=False,
        )

    async def _handle_ccxt_errors_impl(
        self, operation_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """CCXT操作の実装"""
        logger.info(f"{operation_name}を実行中...")

        # 非同期で実行
        # run_in_executorはキーワード引数を直接渡せないため、lambdaを使用
        try:
            if kwargs:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(*args, **kwargs)
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, func, *args
                )

            logger.info(f"{operation_name}実行成功")
            return result

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {e}")
            raise DataError(f"無効なシンボル: {e}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise DataError(f"ネットワークエラー: {e}") from e
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise DataError(f"取引所エラー: {e}") from e
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise DataError(
                f"{operation_name}中にエラーが発生しました: {e}"
            ) from e

    async def _fetch_paginated_data(
        self,
        fetch_func: Callable,
        symbol: str,
        page_limit: int = 200,
        max_pages: int = 50,
        latest_existing_timestamp: Optional[int] = None,
        pagination_strategy: str = "until",
        **fetch_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        ページネーション処理で全期間データを取得（汎用版）

        Args:
            fetch_func: データ取得関数
            symbol: 正規化されたシンボル
            page_limit: 1ページあたりの取得件数
            max_pages: 最大ページ数
            latest_existing_timestamp: 既存データの最新タイムスタンプ（差分更新用）
            pagination_strategy: ページネーション戦略（"until" または "time_range"）
            **fetch_kwargs: fetch_funcに渡す追加引数

        Returns:
            全期間のデータリスト
        """
        logger.info(f"全期間データ取得開始: {symbol} (strategy: {pagination_strategy})")

        if pagination_strategy == "until":
            return await self._fetch_paginated_data_until(
                fetch_func,
                symbol,
                page_limit,
                max_pages,
                latest_existing_timestamp,
                **fetch_kwargs,
            )
        elif pagination_strategy == "time_range":
            return await self._fetch_paginated_data_time_range(
                fetch_func,
                symbol,
                page_limit,
                max_pages,
                latest_existing_timestamp,
                **fetch_kwargs,
            )
        else:
            raise ValueError(f"未対応のページネーション戦略: {pagination_strategy}")

    async def _fetch_paginated_data_until(
        self,
        fetch_func: Callable,
        symbol: str,
        page_limit: int,
        max_pages: int,
        latest_existing_timestamp: Optional[int],
        **fetch_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        untilパラメータを使用したページネーション（ファンディングレート用）
        """
        all_data = []
        page_count = 0
        # 最新の時刻から開始（Bybit APIは新しいデータから古いデータの順で返す）
        until_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        while page_count < max_pages:
            page_count += 1

            try:
                # ページごとにデータを取得
                # Bybit APIは until パラメータを使用して指定時刻より前のデータを取得
                page_data = await self._handle_ccxt_errors(
                    f"ページデータ取得 (page={page_count})",
                    fetch_func,
                    symbol,
                    None,  # since（開始時刻は指定しない）
                    page_limit,
                    {
                        "until": until_time,
                        **fetch_kwargs,
                    },  # 指定時刻より前のデータを取得
                )

                if not page_data:
                    break

                logger.info(
                    f"ページ {page_count}: {len(page_data)}件取得 "
                    f"(累計: {len(all_data) + len(page_data)}件)"
                )

                # 共通の重複チェックと差分更新処理
                new_items = self._process_page_data(
                    page_data, all_data, latest_existing_timestamp, page_count
                )

                if new_items is None:  # 差分更新完了
                    break

                all_data.extend(new_items)

                # 次のページのために until_time を更新
                if page_data:
                    until_time = min(item["timestamp"] for item in page_data) - 1

                # 取得件数が期待値より少ない場合は最後のページ
                if len(page_data) < page_limit:
                    logger.info(f"ページ {page_count}: 最後のページに到達")
                    break

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"ページ {page_count} 取得エラー: {e}")
                continue

        all_data.sort(key=lambda x: x["timestamp"], reverse=True)  # 新しい順にソート
        logger.info(f"全期間データ取得完了: {len(all_data)}件 ({page_count}ページ)")
        return all_data

    async def _fetch_paginated_data_time_range(
        self,
        fetch_func: Callable,
        symbol: str,
        page_limit: int,
        max_pages: int,
        latest_existing_timestamp: Optional[int],
        interval: str = "1h",
        **fetch_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        時間範囲を使用したページネーション（オープンインタレスト用）
        """
        all_data = []
        page_count = 0
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        # 間隔に応じた時間差を計算（ミリ秒）
        interval_ms = self._get_interval_milliseconds(interval)

        # APIシンボル形式に変換（オープンインタレスト用）
        api_symbol = self._convert_to_api_symbol(symbol)
        category = "linear" if symbol.endswith("USDT") else "inverse"

        while page_count < max_pages:
            page_count += 1

            try:
                # 時間範囲を計算
                start_time = end_time - (page_limit * interval_ms)

                # パラメータを構築
                params = {
                    "category": category,
                    "symbol": api_symbol,
                    "intervalTime": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": page_limit,
                    **fetch_kwargs,
                }

                # ページごとにデータを取得
                page_data = await self._handle_ccxt_errors(
                    f"オープンインタレストページデータ取得 (page={page_count})",
                    fetch_func,
                    api_symbol,
                    since=start_time,
                    limit=page_limit,
                    params=params,
                )

                if not page_data:
                    logger.info(f"ページ {page_count}: データなし。取得終了")
                    break

                logger.info(
                    f"ページ {page_count}: {len(page_data)}件取得 "
                    f"(累計: {len(all_data) + len(page_data)}件)"
                )

                # 共通の重複チェックと差分更新処理（オープンインタレスト用）
                new_items = self._process_page_data(
                    page_data,
                    all_data,
                    latest_existing_timestamp,
                    page_count,
                    "open_interest",
                )

                if new_items is None:  # 差分更新完了
                    break

                all_data.extend(new_items)

                # 取得件数が期待値より少ない場合は最後のページ
                if len(page_data) < page_limit:
                    logger.info(
                        f"ページ {page_count}: 最後のページに到達（件数不足: {len(page_data)} < {page_limit}）"
                    )
                    break

                # 次のページの終了時刻を現在のページの開始時刻に設定
                end_time = start_time - 1

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"ページ {page_count} 取得エラー: {e}")
                break

        logger.info(f"全期間データ取得完了: {len(all_data)}件 ({page_count}ページ)")
        return all_data

    def _process_page_data(
        self,
        page_data: List[Dict[str, Any]],
        all_data: List[Dict[str, Any]],
        latest_existing_timestamp: Optional[int],
        page_count: int,
        data_type: str = "default",
    ) -> Optional[List[Dict[str, Any]]]:
        """
        ページデータの共通処理（重複チェック、差分更新）

        Args:
            page_data: 現在のページのデータ
            all_data: これまでに取得した全データ
            latest_existing_timestamp: 既存データの最新タイムスタンプ
            page_count: 現在のページ番号
            data_type: データタイプ（"open_interest", "funding_rate", "default"）

        Returns:
            新規データのリスト。差分更新完了の場合はNone
        """
        # 重複チェック（タイムスタンプベース）
        existing_timestamps = {item["timestamp"] for item in all_data}
        new_items = [
            item for item in page_data if item["timestamp"] not in existing_timestamps
        ]

        # 差分更新: データタイプに応じた条件で新規データをフィルタ
        if latest_existing_timestamp:
            if data_type == "open_interest":
                # オープンインタレスト: 既存データより新しいデータのみ追加
                new_items = [
                    item
                    for item in new_items
                    if item["timestamp"] > latest_existing_timestamp
                ]
            else:
                # その他のデータ: 既存データより古いデータのみ追加（従来の動作）
                new_items = [
                    item
                    for item in new_items
                    if item["timestamp"] < latest_existing_timestamp
                ]

            if not new_items:
                logger.info(f"ページ {page_count}: 既存データに到達。差分更新完了")
                return None

        logger.info(f"ページ {page_count}: 新規データ {len(new_items)}件追加")

        return new_items

    def _get_interval_milliseconds(self, interval: str) -> int:
        """
        間隔文字列をミリ秒に変換

        Args:
            interval: 間隔文字列（例: "1h", "5min"）

        Returns:
            ミリ秒数
        """
        interval_map = {
            "5min": 5 * 60 * 1000,
            "15min": 15 * 60 * 1000,
            "30min": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return interval_map.get(interval, 60 * 60 * 1000)  # デフォルトは1時間

    def _convert_to_api_symbol(self, symbol: str) -> str:
        """
        シンボルをAPI用形式に変換

        Args:
            symbol: 正規化されたシンボル（例: "BTC/USDT:USDT"）

        Returns:
            API用シンボル（例: "BTCUSDT"）
        """
        # 入力の型を確認
        if not isinstance(symbol, str):
            symbol = str(symbol)

        # スラッシュを削除
        api_symbol = symbol.replace("/", "")
        # コロン以降を削除
        if ":" in api_symbol:
            api_symbol = api_symbol.split(":")[0]
        return api_symbol

    async def _execute_with_db_session(self, func: Callable, **kwargs) -> Any:
        """
        データベースセッションを使用して関数を実行
        リポジトリが引数として渡された場合は、新しいセッションを作成しない。
        """
        if "repository" in kwargs and kwargs.get("repository") is not None:
            # 既存のリポジトリが渡された場合、dbセッションはNoneで呼び出す
            return await func(db=None, **kwargs)
        else:
            # get_db()を使用してセッションを取得
            db = next(get_db())
            try:
                # 新しいdbセッションを渡す
                return await func(db=db, **kwargs)
            finally:
                db.close()

    async def _get_latest_timestamp_from_db(
        self,
        repository_class: Any,
        get_timestamp_method_name: str,
        symbol: str,
        **kwargs,
    ) -> Optional[int]:
        """
        データベースから最新のタイムスタンプを汎用的に取得する

        Args:
            repository_class: リポジトリクラス
            get_timestamp_method_name: タイムスタンプ取得メソッド名
            symbol: シンボル
            **kwargs: タイムスタンプ取得メソッドに渡す追加引数

        Returns:
            最新のタイムスタンプ（ミリ秒）
        """

        async def get_timestamp(db, **_):
            repo = repository_class(db)
            get_timestamp_method = getattr(repo, get_timestamp_method_name)
            latest_datetime = get_timestamp_method(symbol, **kwargs)
            if latest_datetime:
                return int(latest_datetime.timestamp() * 1000)
            return None

        try:
            return await self._execute_with_db_session(
                func=get_timestamp, repository=None
            )
        except Exception as e:
            logger.error(f"最新タイムスタンプの取得中にエラーが発生しました: {e}")
            return None

    async def fetch_incremental_data(
        self,
        symbol: str,
        config: Any,
        repository: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        """
        差分データを取得してデータベースに保存（汎用版）

        Args:
            symbol: 取引ペアシンボル
            config: データサービス設定
            repository: リポジトリインスタンス（テスト用）
            **kwargs: fetch_history_methodに渡す追加引数

        Returns:
            差分更新結果を含む辞書
        """
        normalized_symbol = self.normalize_symbol(symbol)

        # データベースから最新タイムスタンプを取得
        latest_timestamp = await self._get_latest_timestamp_from_db(
            repository_class=config.repository_class,
            get_timestamp_method_name=config.get_timestamp_method_name,
            symbol=normalized_symbol,
        )

        # 履歴取得メソッドを取得
        fetch_history_method = getattr(self.exchange, config.fetch_history_method_name)

        if latest_timestamp:
            logger.info(
                f"{config.log_prefix}差分データ収集開始: {normalized_symbol} (since: {latest_timestamp})"
            )
            # 最新タイムスタンプより新しいデータを取得
            if config.fetch_history_method_name == "fetch_open_interest_history":
                interval = kwargs.get("intervalTime", "1h")
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}差分履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    interval,  # timeframe
                    latest_timestamp,
                    1000,
                    kwargs,
                )
            else:
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}差分履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    latest_timestamp,
                    1000,
                    kwargs,
                )

            # 重複を避けるため、最新タイムスタンプより新しいデータのみフィルタ
            history_data = [
                item for item in history_data if item["timestamp"] > latest_timestamp
            ]
        else:
            logger.info(f"{config.log_prefix}初回データ収集開始: {normalized_symbol}")
            # データがない場合は最新データを取得
            if config.fetch_history_method_name == "fetch_open_interest_history":
                interval = kwargs.get("intervalTime", "1h")
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}初回履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    interval,  # timeframe
                    None,  # since
                    config.default_limit,
                    kwargs,
                )
            else:
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}初回履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    None,
                    config.default_limit,
                    kwargs,
                )

        # データベースに保存
        async def save_with_db(db, repository):
            repo = repository or config.repository_class(db)
            return await self._save_data_to_database(history_data, symbol, repo, config)

        saved_count = await self._execute_with_db_session(
            func=save_with_db, repository=repository
        )

        logger.info(f"{config.log_prefix}差分データ収集完了: {saved_count}件保存")
        return {
            "symbol": normalized_symbol,
            "fetched_count": len(history_data),
            "saved_count": saved_count,
            "success": True,
            "latest_timestamp": latest_timestamp,
        }

    async def fetch_and_save_data(
        self,
        symbol: str,
        config: Any,
        limit: Optional[int] = None,
        repository: Optional[Any] = None,
        fetch_all: bool = False,
        **kwargs,
    ) -> dict:
        """
        データを取得してデータベースに保存（汎用版）

        Args:
            symbol: 取引ペアシンボル
            config: データサービス設定
            limit: 取得件数制限
            repository: リポジトリインスタンス（テスト用）
            fetch_all: 全期間データを取得するかどうか
            **kwargs: fetch_history_methodに渡す追加引数

        Returns:
            取得・保存結果を含む辞書
        """
        normalized_symbol = self.normalize_symbol(symbol)

        if fetch_all:
            # 全期間データを取得
            latest_timestamp = await self._get_latest_timestamp_from_db(
                repository_class=config.repository_class,
                get_timestamp_method_name=config.get_timestamp_method_name,
                symbol=normalized_symbol,
            )

            fetch_history_method = getattr(
                self.exchange, config.fetch_history_method_name
            )
            history_data = await self._fetch_paginated_data(
                fetch_func=fetch_history_method,
                symbol=normalized_symbol,
                page_limit=config.page_limit,
                max_pages=config.max_pages,
                latest_existing_timestamp=latest_timestamp,
                pagination_strategy=config.pagination_strategy,
                **kwargs,
            )
        else:
            # 指定件数のデータを取得
            fetch_history_method = getattr(
                self.exchange, config.fetch_history_method_name
            )

            # オープンインタレストの場合はtimeframeパラメータが必要
            if config.fetch_history_method_name == "fetch_open_interest_history":
                interval = kwargs.get("intervalTime", "1h")
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    interval,  # timeframe
                    None,  # since
                    limit or config.default_limit,
                    kwargs,
                )
            else:
                history_data = await self._handle_ccxt_errors(
                    f"{config.log_prefix}履歴取得",
                    fetch_history_method,
                    normalized_symbol,
                    None,
                    limit or config.default_limit,
                    kwargs,
                )

        # データベースに保存
        async def save_with_db(db, repository):
            repo = repository or config.repository_class(db)
            return await self._save_data_to_database(history_data, symbol, repo, config)

        saved_count = await self._execute_with_db_session(
            func=save_with_db, repository=repository
        )

        return {
            "symbol": normalized_symbol,
            "fetched_count": len(history_data),
            "saved_count": saved_count,
            "success": True,
        }

    async def _save_data_to_database(
        self,
        history_data: List[Dict[str, Any]],
        symbol: str,
        repository: Any,
        config: Any,
    ) -> int:
        """
        データをデータベースに保存（汎用版）

        Args:
            history_data: 履歴データ
            symbol: シンボル
            repository: リポジトリインスタンス
            config: データサービス設定

        Returns:
            保存件数
        """
        logger.info(
            f"{config.log_prefix}データのDB保存開始: {len(history_data)}件のデータを変換中..."
        )

        # データ変換
        converter_method = getattr(
            config.data_converter_class, config.converter_method_name
        )
        records = converter_method(history_data, self.normalize_symbol(symbol))

        logger.info(f"データ変換完了: {len(records)}件のレコードをDB挿入開始...")

        # データベース挿入
        try:
            insert_method = getattr(repository, config.insert_method_name)
            inserted_count = insert_method(records)
            logger.info(
                f"{config.log_prefix}データのDB保存完了: {inserted_count}件挿入"
            )
            return inserted_count
        except Exception as e:
            logger.error(f"{config.log_prefix}データのDB保存エラー: {e}")
            raise
