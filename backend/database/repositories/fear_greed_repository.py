"""
Fear & Greed Index データのリポジトリクラス
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from .base_repository import BaseRepository
from database.models import FearGreedIndexData
from app.utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class FearGreedIndexRepository(BaseRepository):
    """Fear & Greed Index データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FearGreedIndexData)

    def to_dict(self, model_instance: FearGreedIndexData) -> dict:
        """Fear & Greed Index データを辞書に変換

        Args:
            model_instance: 変換するモデルインスタンス

        Returns:
            変換された辞書
        """
        d = super().to_dict(model_instance)
        # UTC保証プロパティを優先
        d["data_timestamp"] = (
            model_instance.data_timestamp_utc.isoformat()
            if model_instance.data_timestamp is not None
            else None
        )
        d["timestamp"] = (
            model_instance.timestamp_utc.isoformat()
            if model_instance.timestamp is not None
            else None
        )
        return d

    def insert_fear_greed_data(self, records: List[dict]) -> int:
        """
        Fear & Greed Index データを一括挿入（重複は無視）

        Args:
            records: Fear & Greed Index データのリスト

        Returns:
            挿入された件数
        """
        if not records:
            return 0

        from app.utils.error_handler import safe_operation

        @safe_operation(context="Fear & Greedデータ挿入", is_api_call=False)
        def _insert_data():
            # データの検証
            if not DataValidator.validate_fear_greed_data(records):
                raise ValueError(
                    "挿入しようとしているFear & Greed Indexデータに無効なものが含まれています。"
                )

            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                records, ["data_timestamp"]
            )

            logger.info(f"Fear & Greed Index データを {inserted_count} 件挿入しました")
            return inserted_count

        return _insert_data()

    def get_fear_greed_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[FearGreedIndexData]:
        """
        Fear & Greed Index データを取得

        Args:
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            Fear & Greed Index データのリスト
        """
        return self.get_filtered_data(
            filters={},
            time_range_column="data_timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="data_timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_fear_greed_data(self, limit: int = 30) -> List[FearGreedIndexData]:
        """
        最新のFear & Greed Index データを取得

        Args:
            limit: 取得件数制限

        Returns:
            最新のFear & Greed Index データのリスト
        """
        return self.get_latest_records(
            filters={},
            timestamp_column="data_timestamp",
            limit=limit,
        )

    def get_latest_data_timestamp(self) -> Optional[datetime]:
        """
        最新のデータタイムスタンプを取得

        Returns:
            最新のデータタイムスタンプ
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="最新データタイムスタンプ取得", is_api_call=False)
        def _get_latest_timestamp():
            return self.get_latest_timestamp("data_timestamp")

        return _get_latest_timestamp()

    def get_data_count(self) -> int:
        """
        保存されているデータ件数を取得

        Returns:
            データ件数
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="データ件数取得", is_api_call=False)
        def _get_data_count():
            return self.db.query(FearGreedIndexData).count()

        return _get_data_count()

    def get_data_range(self) -> dict:
        """
        データの範囲情報を取得

        Returns:
            データ範囲情報（最古・最新のタイムスタンプ、件数）
        """
        from app.utils.error_handler import safe_operation

        @safe_operation(context="データ範囲情報取得", is_api_call=False)
        def _get_data_range():
            from sqlalchemy import func
            from datetime import timezone

            result = self.db.query(
                func.min(FearGreedIndexData.data_timestamp).label("oldest"),
                func.max(FearGreedIndexData.data_timestamp).label("newest"),
                func.count(FearGreedIndexData.id).label("count"),
            ).first()

            if result is None:
                return {
                    "oldest_data": None,
                    "newest_data": None,
                    "total_count": 0,
                }

            # タイムゾーン情報が失われている場合はUTCを設定
            oldest = result.oldest
            newest = result.newest

            if oldest and oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            if newest and newest.tzinfo is None:
                newest = newest.replace(tzinfo=timezone.utc)

            return {
                "oldest_data": oldest.isoformat() if oldest else None,
                "newest_data": newest.isoformat() if newest else None,
                "total_count": result.count or 0,
            }

        return _get_data_range()

    def delete_old_data(
        self,
        timestamp_column: str,
        before_date: datetime,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        指定日時より古いデータを削除

        Args:
            timestamp_column: タイムスタンプカラム名
            before_date: この日時より古いデータを削除
            additional_filters: 追加フィルター条件

        Returns:
            削除された件数
        """
        deleted_count = super().delete_old_data(
            timestamp_column=timestamp_column,
            before_date=before_date,
            additional_filters=additional_filters,
        )
        logger.info(f"古いFear & Greed Indexデータを {deleted_count} 件削除しました")
        return deleted_count
