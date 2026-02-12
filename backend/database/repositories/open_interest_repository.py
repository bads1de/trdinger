"""
オープンインタレストデータのリポジトリクラス
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from database.models import OpenInterestData

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class OpenInterestRepository(BaseRepository):
    """オープンインタレストデータのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, OpenInterestData)

    def insert_open_interest_data(self, open_interest_records: List[dict]) -> int:
        """
        オープンインタレストデータを一括挿入

        Args:
            open_interest_records: オープンインタレストデータのリスト

        Returns:
            挿入された件数
        """
        if not open_interest_records:
            logger.warning("挿入するオープンインタレストデータがありません。")
            return 0

        from app.utils.error_handler import safe_operation

        @safe_operation(context="オープンインタレストデータ挿入", is_api_call=False)
        def _insert_data():
            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                open_interest_records, ["symbol", "data_timestamp"]
            )

            logger.info(f"オープンインタレストデータを {inserted_count} 件挿入しました")
            return inserted_count

        return _insert_data()

    def get_open_interest_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OpenInterestData]:
        """
        オープンインタレストデータを取得

        Args:
            symbol: 取引ペア
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            オープンインタレストデータのリスト
        """
        filters = {"symbol": symbol}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="data_timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="data_timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_open_interest_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最新オープンインタレストタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最新のデータタイムスタンプ（データがない場合はNone）
        """
        return super().get_latest_timestamp("data_timestamp", {"symbol": symbol})

    def get_oldest_open_interest_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最古オープンインタレストタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最古のデータタイムスタンプ（データがない場合はNone）
        """
        return super().get_oldest_timestamp("data_timestamp", {"symbol": symbol})

    def clear_all_open_interest_data(self) -> int:
        """
        全てのオープンインタレストデータを削除

        Returns:
            削除された件数
        """
        deleted_count = self._delete_all_records()
        logger.info(
            f"全てのオープンインタレストデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_open_interest_data_by_symbol(self, symbol: str) -> int:
        """
        指定されたシンボルのオープンインタレストデータを削除

        Args:
            symbol: 削除対象のシンボル

        Returns:
            削除された件数
        """
        deleted_count = self._delete_records_by_filter("symbol", symbol)
        logger.info(
            f"シンボル '{symbol}' のオープンインタレストデータを削除しました: {deleted_count}件"
        )
        return deleted_count



