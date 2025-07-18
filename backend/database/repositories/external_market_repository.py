"""
外部市場データリポジトリ

yfinance APIから取得した外部市場データ（SP500、NASDAQ、DXY、VIXなど）の
データベース操作を管理します。
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session

from database.models import ExternalMarketData
from database.repositories.base_repository import BaseRepository
from app.core.utils.database_utils import DatabaseQueryHelper
from app.core.utils.data_converter import DataValidator

logger = logging.getLogger(__name__)


class ExternalMarketRepository(BaseRepository):
    """外部市場データリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, ExternalMarketData)

    def insert_external_market_data(self, records: List[dict]) -> int:
        """
        外部市場データを一括挿入（重複は無視）

        Args:
            records: 外部市場データのリスト

        Returns:
            挿入された件数
        """
        if not records:
            return 0

        try:
            # データの検証
            if not DataValidator.validate_external_market_data(records):
                raise ValueError(
                    "挿入しようとしている外部市場データに無効なものが含まれています。"
                )

            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                records, ["symbol", "data_timestamp"]
            )

            logger.info(f"外部市場データを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            logger.error(f"外部市場データの挿入中にエラーが発生しました: {e}")
            raise

    def get_external_market_data(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[ExternalMarketData]:
        """
        外部市場データを取得

        Args:
            symbol: シンボル（例: ^GSPC, ^IXIC）
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            外部市場データのリスト
        """
        try:
            filters = {}
            if symbol:
                filters["symbol"] = symbol

            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=ExternalMarketData,
                filters=filters,
                time_range_column="data_timestamp",
                start_time=start_time,
                end_time=end_time,
                order_by_column="data_timestamp",
                order_asc=True,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"外部市場データの取得中にエラーが発生しました: {e}")
            raise

    def get_latest_external_market_data(
        self, symbol: Optional[str] = None, limit: int = 30
    ) -> List[ExternalMarketData]:
        """
        最新の外部市場データを取得

        Args:
            symbol: シンボル（指定しない場合は全シンボル）
            limit: 取得件数制限

        Returns:
            最新の外部市場データのリスト
        """
        try:
            filters = {}
            if symbol:
                filters["symbol"] = symbol

            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=ExternalMarketData,
                filters=filters,
                time_range_column="data_timestamp",
                start_time=None,
                end_time=None,
                order_by_column="data_timestamp",
                order_asc=False,  # 降順で最新データを取得
                limit=limit,
            )

        except Exception as e:
            logger.error(f"最新外部市場データの取得中にエラーが発生しました: {e}")
            raise

    def get_latest_data_timestamp(
        self, symbol: Optional[str] = None
    ) -> Optional[datetime]:
        """
        最新データのタイムスタンプを取得

        Args:
            symbol: シンボル（指定しない場合は全シンボル）

        Returns:
            最新データのタイムスタンプ（データが存在しない場合はNone）
        """
        try:
            query = self.db.query(ExternalMarketData.data_timestamp)

            if symbol:
                query = query.filter(ExternalMarketData.symbol == symbol)

            latest_record = query.order_by(
                ExternalMarketData.data_timestamp.desc()
            ).first()

            return latest_record[0] if latest_record else None

        except Exception as e:
            logger.error(f"最新データタイムスタンプの取得中にエラーが発生しました: {e}")
            raise

    def get_data_range(self, symbol: Optional[str] = None) -> Dict:
        """
        データの範囲情報を取得

        Args:
            symbol: シンボル（指定しない場合は全シンボル）

        Returns:
            データ範囲情報（最古・最新・件数）
        """
        try:
            query = self.db.query(ExternalMarketData)

            if symbol:
                query = query.filter(ExternalMarketData.symbol == symbol)

            # 件数
            count = query.count()

            if count == 0:
                return {
                    "count": 0,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                }

            # 最古・最新のタイムスタンプ
            oldest_record = query.order_by(
                ExternalMarketData.data_timestamp.asc()
            ).first()
            newest_record = query.order_by(
                ExternalMarketData.data_timestamp.desc()
            ).first()

            return {
                "count": count,
                "oldest_timestamp": (
                    oldest_record.data_timestamp.isoformat()
                    if oldest_record is not None
                    and oldest_record.data_timestamp is not None
                    else None
                ),
                "newest_timestamp": (
                    newest_record.data_timestamp.isoformat()
                    if newest_record is not None
                    and newest_record.data_timestamp is not None
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"データ範囲情報の取得中にエラーが発生しました: {e}")
            raise

    def delete_old_data(
        self, cutoff_date: datetime, symbol: Optional[str] = None
    ) -> int:
        """
        古いデータを削除

        Args:
            cutoff_date: 削除基準日（この日より古いデータを削除）
            symbol: シンボル（指定しない場合は全シンボル）

        Returns:
            削除された件数
        """
        try:
            query = self.db.query(ExternalMarketData).filter(
                ExternalMarketData.data_timestamp < cutoff_date
            )

            if symbol:
                query = query.filter(ExternalMarketData.symbol == symbol)

            deleted_count = query.count()
            query.delete(synchronize_session=False)
            self.db.commit()

            logger.info(f"古い外部市場データを {deleted_count} 件削除しました")
            return deleted_count

        except Exception as e:
            logger.error(f"古いデータの削除中にエラーが発生しました: {e}")
            self.db.rollback()
            raise

    def get_symbols(self) -> List[str]:
        """
        データベースに存在するシンボルの一覧を取得

        Returns:
            シンボルのリスト
        """
        try:
            symbols = (
                self.db.query(ExternalMarketData.symbol)
                .distinct()
                .order_by(ExternalMarketData.symbol)
                .all()
            )

            return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"シンボル一覧の取得中にエラーが発生しました: {e}")
            raise

    def get_data_statistics(self, symbol: Optional[str] = None) -> Dict:
        """
        データの統計情報を取得

        Args:
            symbol: シンボル（指定しない場合は全シンボル）

        Returns:
            統計情報
        """
        try:
            query = self.db.query(ExternalMarketData)

            if symbol:
                query = query.filter(ExternalMarketData.symbol == symbol)

            # 基本統計
            count = query.count()

            if count == 0:
                return {
                    "count": 0,
                    "symbols": [],
                    "date_range": None,
                }

            # シンボル別統計
            if symbol:
                symbols = [symbol]
            else:
                symbols = self.get_symbols()

            # 日付範囲
            oldest_record = query.order_by(
                ExternalMarketData.data_timestamp.asc()
            ).first()
            newest_record = query.order_by(
                ExternalMarketData.data_timestamp.desc()
            ).first()

            return {
                "count": count,
                "symbols": symbols,
                "symbol_count": len(symbols),
                "date_range": {
                    "oldest": (
                        oldest_record.data_timestamp.isoformat()
                        if oldest_record is not None
                        and oldest_record.data_timestamp is not None
                        else None
                    ),
                    "newest": (
                        newest_record.data_timestamp.isoformat()
                        if newest_record is not None
                        and newest_record.data_timestamp is not None
                        else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"統計情報の取得中にエラーが発生しました: {e}")
            raise
