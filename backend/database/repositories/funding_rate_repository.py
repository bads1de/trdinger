"""
ファンディングレートデータのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import FundingRateData
from app.core.utils.database_utils import DatabaseQueryHelper

logger = logging.getLogger(__name__)


class FundingRateRepository(BaseRepository):
    """ファンディングレートデータのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FundingRateData)

    def insert_funding_rate_data(self, funding_rate_records: List[dict]) -> int:
        """
        ファンディングレートデータを一括挿入

        Args:
            funding_rate_records: ファンディングレートデータのリスト

        Returns:
            挿入された件数
        """
        if not funding_rate_records:
            logger.warning("挿入するファンディングレートデータがありません")
            return 0

        try:
            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                funding_rate_records, ["symbol", "funding_timestamp"]
            )

            logger.info(f"ファンディングレートデータを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            logger.error(f"ファンディングレートデータ挿入エラー: {e}")
            raise

    def get_funding_rate_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[FundingRateData]:
        """
        ファンディングレートデータを取得

        Args:
            symbol: 取引ペア
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            ファンディングレートデータのリスト
        """
        filters = {"symbol": symbol}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="funding_timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="funding_timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最新ファンディングタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最新のファンディングタイムスタンプ（データがない場合はNone）
        """
        return super().get_latest_timestamp("funding_timestamp", {"symbol": symbol})

    def get_oldest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最古ファンディングタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最古のファンディングタイムスタンプ（データがない場合はNone）
        """
        return super().get_oldest_timestamp("funding_timestamp", {"symbol": symbol})

    def get_funding_rate_count(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータ件数を取得

        Args:
            symbol: 取引ペア

        Returns:
            データ件数
        """
        return super().get_record_count({"symbol": symbol})

    def clear_all_funding_rate_data(self) -> int:
        """
        全てのファンディングレートデータを削除

        Returns:
            削除された件数
        """
        deleted_count = self._delete_all_records()
        logger.info(
            f"全てのファンディングレートデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_funding_rate_data_by_symbol(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータを削除

        Args:
            symbol: 削除対象のシンボル

        Returns:
            削除された件数
        """
        deleted_count = self._delete_records_by_filter("symbol", symbol)
        logger.info(
            f"シンボル '{symbol}' のファンディングレートデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_funding_rate_data_by_date_range(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        指定された期間のファンディングレートデータを削除

        Args:
            symbol: 取引ペア
            start_time: 削除開始時刻
            end_time: 削除終了時刻

        Returns:
            削除された件数
        """
        additional_filters = {"symbol": symbol}
        deleted_count = self.delete_by_date_range(
            timestamp_column="funding_timestamp",
            start_time=start_time,
            end_time=end_time,
            additional_filters=additional_filters,
        )
        logger.info(
            f"期間指定でファンディングレートデータを削除しました ({symbol}): {deleted_count}件"
        )
        return deleted_count

    def get_funding_rate_dataframe(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        ファンディングレートデータをDataFrameとして取得

        Args:
            symbol: 取引ペア
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            ファンディングレートデータのDataFrame
        """
        records = self.get_funding_rate_data(symbol, start_time, end_time, limit)

        column_mapping = {
            "funding_timestamp": "funding_timestamp",
            "funding_rate": "funding_rate",
            "mark_price": "mark_price",
            "index_price": "index_price",
        }

        return self.to_dataframe(
            records=records,
            column_mapping=column_mapping,
            index_column="funding_timestamp",
        )
