"""
ファンディングレートサービス

CCXTライブラリを使用してBybitからファンディングレートデータを取得し、
データベースに保存する機能を提供します。
"""

import asyncio
import ccxt
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import logging

from database.connection import SessionLocal
from database.repositories.funding_rate_repository import FundingRateRepository
from app.core.utils.data_converter import FundingRateDataConverter

logger = logging.getLogger(__name__)


class BybitFundingRateService:
    """Bybitファンディングレートサービス"""

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

    def _validate_parameters(self, symbol: str, limit: int):
        """パラメータの検証"""
        if not symbol:
            raise ValueError("シンボルが指定されていません")

        if limit <= 0 or limit > 1000:
            raise ValueError("limitは1-1000の範囲で指定してください")

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

        try:
            logger.info(f"現在のファンディングレートを取得中: {normalized_symbol}")

            # 非同期でファンディングレートを取得
            funding_rate = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_funding_rate, normalized_symbol
            )

            logger.info(f"現在のファンディングレート取得成功: {normalized_symbol}")
            return funding_rate

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {normalized_symbol}")
            raise ccxt.BadSymbol(f"無効なシンボル: {normalized_symbol}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise ccxt.ExchangeError(
                f"ファンディングレート取得中にエラーが発生しました: {e}"
            ) from e

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

        try:
            logger.info(
                f"ファンディングレート履歴を取得中: {normalized_symbol}, limit={limit}"
            )

            # 非同期でファンディングレート履歴を取得
            funding_history = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_funding_rate_history,
                normalized_symbol,
                since,
                limit,
            )

            logger.info(f"ファンディングレート履歴取得成功: {len(funding_history)}件")
            return funding_history

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {normalized_symbol}")
            raise ccxt.BadSymbol(f"無効なシンボル: {normalized_symbol}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise ccxt.ExchangeError(
                f"ファンディングレート履歴取得中にエラーが発生しました: {e}"
            ) from e

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

        try:
            logger.info(
                f"全期間のファンディングレート履歴を取得中（改善版）: {normalized_symbol}"
            )

            all_funding_history = []
            page_limit = 200  # Bybitの実際の制限に合わせる
            page_count = 0
            max_pages = 50  # 安全のための上限（約10,000件）

            # 最新データから開始
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            end_time = current_time

            # 差分更新のための最新データ確認
            latest_existing_timestamp = await self._get_latest_funding_rate_timestamp(
                normalized_symbol
            )

            while page_count < max_pages:
                page_count += 1

                try:
                    # ページごとにデータを取得（逆方向）
                    funding_history = await self._fetch_funding_rate_page_reverse(
                        normalized_symbol, end_time, page_limit
                    )

                    if not funding_history:
                        logger.info(f"ページ {page_count}: データなし。取得終了")
                        break

                    logger.info(
                        f"ページ {page_count}: {len(funding_history)}件取得 (累計: {len(all_funding_history) + len(funding_history)}件)"
                    )

                    # 重複チェック（タイムスタンプベース）
                    existing_timestamps = {
                        item["timestamp"] for item in all_funding_history
                    }
                    new_items = [
                        item
                        for item in funding_history
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

                    all_funding_history.extend(new_items)
                    logger.info(
                        f"ページ {page_count}: 新規データ {len(new_items)}件追加 (累計: {len(all_funding_history)}件)"
                    )

                    # 次のページの終了時刻を設定（最古のタイムスタンプ）
                    if funding_history:
                        oldest_timestamp = min(
                            item["timestamp"] for item in funding_history
                        )
                        end_time = oldest_timestamp - 1

                        # データが少ない場合は最後のページ
                        if len(funding_history) < page_limit:
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
            all_funding_history.sort(key=lambda x: x["timestamp"])

            logger.info(
                f"全期間のファンディングレート履歴取得完了: {len(all_funding_history)}件 ({page_count}ページ)"
            )
            return all_funding_history

        except ccxt.BadSymbol as e:
            logger.error(f"無効なシンボル: {normalized_symbol}")
            raise ccxt.BadSymbol(f"無効なシンボル: {normalized_symbol}") from e
        except ccxt.NetworkError as e:
            logger.error(f"ネットワークエラー: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"取引所エラー: {e}")
            raise
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise ccxt.ExchangeError(
                f"ファンディングレート履歴取得中にエラーが発生しました: {e}"
            ) from e

    async def _fetch_funding_rate_page_reverse(
        self, symbol: str, end_time: int, limit: int
    ) -> List[Dict[str, Any]]:
        """
        逆方向ページネーション用のファンディングレート取得

        Args:
            symbol: 正規化されたシンボル
            end_time: 終了時刻（ミリ秒）
            limit: 取得件数制限

        Returns:
            ファンディングレート履歴データのリスト
        """
        try:
            # 現在のCCXTライブラリでは直接endTimeを指定できないため、
            # 代替手段として時間範囲を計算してsinceを使用

            # 8時間間隔でファンディングレートが設定されるため、
            # limit * 8時間前からend_timeまでの範囲を指定
            hours_back = limit * 8  # 8時間間隔
            since_time = end_time - (hours_back * 60 * 60 * 1000)  # ミリ秒

            # 通常のfetch_funding_rate_historyを使用
            funding_history = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_funding_rate_history,
                symbol,
                since_time,
                limit,
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
            logger.error(f"逆方向ページネーション取得エラー: {e}")
            # フォールバック: 通常の取得方法
            return await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_funding_rate_history, symbol, None, limit
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
                # 実際のデータベースセッションを使用
                db = SessionLocal()
                try:
                    repository = FundingRateRepository(db)
                    saved_count = await self._save_funding_rate_to_database(
                        funding_history, symbol, repository
                    )
                    db.close()
                except Exception as e:
                    db.close()
                    raise
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
