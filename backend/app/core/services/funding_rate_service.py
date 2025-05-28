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
from database.repository import FundingRateRepository

logger = logging.getLogger(__name__)


class BybitFundingRateService:
    """Bybitファンディングレートサービス"""

    def __init__(self):
        """サービスを初期化"""
        self.exchange = ccxt.bybit({
            'sandbox': False,  # 本番環境を使用（読み取り専用）
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # 無期限契約市場を使用
            }
        })

    def normalize_symbol(self, symbol: str) -> str:
        """
        シンボルを正規化（無期限契約形式に変換）

        Args:
            symbol: 入力シンボル（例: 'BTC/USDT' または 'BTC/USDT:USDT'）

        Returns:
            正規化されたシンボル（例: 'BTC/USDT:USDT'）
        """
        # 既に無期限契約形式の場合はそのまま返す
        if ':' in symbol:
            return symbol

        # スポット形式を無期限契約形式に変換
        if symbol.endswith('/USDT'):
            return f"{symbol}:USDT"
        elif symbol.endswith('/USD'):
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
                None,
                self.exchange.fetch_funding_rate,
                normalized_symbol
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
            raise ccxt.ExchangeError(f"ファンディングレート取得中にエラーが発生しました: {e}") from e

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        since: Optional[int] = None
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
            logger.info(f"ファンディングレート履歴を取得中: {normalized_symbol}, limit={limit}")

            # 非同期でファンディングレート履歴を取得
            funding_history = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_funding_rate_history,
                normalized_symbol,
                since,
                limit
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
            raise ccxt.ExchangeError(f"ファンディングレート履歴取得中にエラーが発生しました: {e}") from e

    async def fetch_all_funding_rate_history(
        self,
        symbol: str
    ) -> List[Dict[str, Any]]:
        """
        全期間のファンディングレート履歴を取得

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
            logger.info(f"全期間のファンディングレート履歴を取得中: {normalized_symbol}")

            all_funding_history = []
            since = None  # 最古のデータから開始
            page_limit = 1000  # Bybitの最大制限

            while True:
                # ページごとにデータを取得
                funding_history = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.exchange.fetch_funding_rate_history,
                    normalized_symbol,
                    since,
                    page_limit
                )

                if not funding_history:
                    # データがない場合は終了
                    break

                logger.info(f"取得: {len(funding_history)}件 (累計: {len(all_funding_history) + len(funding_history)}件)")

                # 重複チェック（タイムスタンプベース）
                existing_timestamps = {item['timestamp'] for item in all_funding_history}
                new_items = [item for item in funding_history if item['timestamp'] not in existing_timestamps]

                all_funding_history.extend(new_items)

                # 次のページの開始点を設定
                if len(funding_history) < page_limit:
                    # 取得件数が制限未満の場合は最後のページ
                    break

                # 最後のアイテムのタイムスタンプの次から開始
                since = funding_history[-1]['timestamp'] + 1

                # レート制限対応
                await asyncio.sleep(0.1)

            logger.info(f"全期間のファンディングレート履歴取得完了: {len(all_funding_history)}件")
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
            raise ccxt.ExchangeError(f"ファンディングレート履歴取得中にエラーが発生しました: {e}") from e

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
                funding_history = await self.fetch_funding_rate_history(symbol, limit or 100)

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
        records = []
        for rate_data in funding_history:
            # タイムスタンプをdatetimeに変換
            funding_timestamp = datetime.fromtimestamp(
                rate_data['timestamp'] / 1000, tz=timezone.utc
            )

            # データ取得時刻
            current_timestamp = datetime.now(timezone.utc)

            record = {
                "symbol": self.normalize_symbol(symbol),
                "funding_rate": float(rate_data['fundingRate']),
                "funding_timestamp": funding_timestamp,
                "timestamp": current_timestamp,
                "next_funding_timestamp": None,  # 履歴データには含まれない
                "mark_price": None,  # 履歴データには含まれない場合がある
                "index_price": None,  # 履歴データには含まれない場合がある
            }
            records.append(record)

        # データベースに挿入
        return repository.insert_funding_rate_data(records)
