"""
Bybitサービスの基底クラス

CCXTライブラリを使用したBybitサービスの共通機能を提供します。
重複コードの削減と一貫性のあるエラーハンドリングを実現します。
"""

import asyncio
import ccxt
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from abc import ABC

from database.connection import SessionLocal

logger = logging.getLogger(__name__)


class BaseBybitService(ABC):
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
            *args: 関数の引数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の実行結果

        Raises:
            ccxt.BadSymbol: 無効なシンボルの場合
            ccxt.NetworkError: ネットワークエラーの場合
            ccxt.ExchangeError: 取引所エラーの場合
        """
        try:
            logger.info(f"{operation_name}を実行中...")

            # 非同期で実行
            result = await asyncio.get_event_loop().run_in_executor(
                None, func, *args, **kwargs
            )

            logger.info(f"{operation_name}実行成功")
            return result

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {e}")
            raise ccxt.BadSymbol(f"無効なシンボル: {e}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise ccxt.ExchangeError(
                f"{operation_name}中にエラーが発生しました: {e}"
            ) from e

    async def _fetch_paginated_data(
        self,
        fetch_func: Callable,
        symbol: str,
        page_limit: int = 200,
        max_pages: int = 50,
        latest_existing_timestamp: Optional[int] = None,
        **fetch_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        ページネーション処理で全期間データを取得

        Args:
            fetch_func: データ取得関数
            symbol: 正規化されたシンボル
            page_limit: 1ページあたりの取得件数
            max_pages: 最大ページ数
            latest_existing_timestamp: 既存データの最新タイムスタンプ（差分更新用）
            **fetch_kwargs: fetch_funcに渡す追加引数

        Returns:
            全期間のデータリスト
        """
        logger.info(f"全期間データ取得開始: {symbol}")

        all_data = []
        page_count = 0
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        while page_count < max_pages:
            page_count += 1

            try:
                # ページごとにデータを取得
                params = fetch_kwargs.get("params", {})
                params["end"] = end_time

                page_data = await self._handle_ccxt_errors(
                    f"ページデータ取得 (page={page_count})",
                    fetch_func,
                    symbol,
                    None,  # since
                    page_limit,
                    params,
                )

                if not page_data:
                    logger.info(f"ページ {page_count}: データなし。取得終了")
                    break

                # 最後のデータは次のリクエストのendになるため、重複を避けるために除外
                if len(page_data) > 1:
                    end_time = page_data[-1]["timestamp"]
                    page_data = page_data[:-1]
                else:
                    end_time = page_data[0]["timestamp"] - 1

                logger.info(
                    f"ページ {page_count}: {len(page_data)}件取得 "
                    f"(累計: {len(all_data) + len(page_data)}件)"
                )

                # 重複チェック（タイムスタンプベース）
                existing_timestamps = {item["timestamp"] for item in all_data}
                new_items = [
                    item
                    for item in page_data
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

                all_data.extend(new_items)
                logger.info(
                    f"ページ {page_count}: 新規データ {len(new_items)}件追加 "
                    f"(累計: {len(all_data)}件)"
                )

                if len(page_data) < page_limit - 1:
                    logger.info(f"ページ {page_count}: 最後のページに到達")
                    break

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"ページ {page_count} 取得エラー: {e}")
                continue

        all_data.sort(key=lambda x: x["timestamp"])
        logger.info(f"全期間データ取得完了: {len(all_data)}件 ({page_count}ページ)")
        return all_data

    def _get_database_session(self):
        """
        データベースセッションを取得

        Returns:
            データベースセッション
        """
        return SessionLocal()

    async def _execute_with_db_session(self, func: Callable, **kwargs) -> Any:
        """
        データベースセッションを使用して関数を実行
        リポジトリが引数として渡された場合は、新しいセッションを作成しない。
        """
        if "repository" in kwargs and kwargs.get("repository") is not None:
            # 既存のリポジトリが渡された場合、dbセッションはNoneで呼び出す
            return await func(db=None, **kwargs)
        else:
            db = self._get_database_session()
            try:
                # 新しいdbセッションを渡す
                return await func(db=db, **kwargs)
            finally:
                db.close()
