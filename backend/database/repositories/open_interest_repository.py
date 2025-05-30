"""
オープンインタレストデータのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import asc
import logging

from .base_repository import BaseRepository
from database.models import OpenInterestData

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
            logger.warning("挿入するオープンインタレストデータがありません")
            return 0

        try:
            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                open_interest_records,
                ["symbol", "data_timestamp"]
            )

            logger.info(f"オープンインタレストデータを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            logger.error(f"オープンインタレストデータ挿入エラー: {e}")
            raise

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
        try:
            query = self.db.query(OpenInterestData).filter(
                OpenInterestData.symbol == symbol
            )

            if start_time:
                query = query.filter(OpenInterestData.data_timestamp >= start_time)

            if end_time:
                query = query.filter(OpenInterestData.data_timestamp <= end_time)

            # 時系列順でソート
            query = query.order_by(asc(OpenInterestData.data_timestamp))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"オープンインタレストデータ取得エラー: {e}")
            raise

    def get_latest_open_interest_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最新オープンインタレストタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最新のデータタイムスタンプ（データがない場合はNone）
        """
        return super().get_latest_timestamp(
            "data_timestamp",
            {"symbol": symbol}
        )

    def get_open_interest_count(self, symbol: str) -> int:
        """
        指定されたシンボルのオープンインタレストデータ件数を取得

        Args:
            symbol: 取引ペア

        Returns:
            データ件数
        """
        return super().get_record_count({"symbol": symbol})
