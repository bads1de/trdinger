"""
Fear & Greed Index データのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from .base_repository import BaseRepository
from database.models import FearGreedIndexData
from app.core.utils.database_utils import DatabaseQueryHelper
from app.core.utils.data_converter import DataValidator

logger = logging.getLogger(__name__)


class FearGreedIndexRepository(BaseRepository):
    """Fear & Greed Index データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FearGreedIndexData)

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

        try:
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

        except Exception as e:
            logger.error(
                f"Fear & Greed Index データの挿入中にエラーが発生しました: {e}"
            )
            raise

    def get_fear_greed_data(
        self, start_time: datetime, end_time: datetime
    ) -> List[FearGreedIndexData]:
        """
        指定期間のFear & Greed Indexデータを取得

        Args:
            start_time: 開始日時
            end_time: 終了日時

        Returns:
            Fear & Greed Indexデータのリスト
        """
        try:
            query = (
                self.db.query(FearGreedIndexData)
                .filter(FearGreedIndexData.data_timestamp >= start_time)
                .filter(FearGreedIndexData.data_timestamp <= end_time)
                .order_by(FearGreedIndexData.data_timestamp)
            )

            results = query.all()
            logger.info(f"Fear & Greed Indexデータを {len(results)} 件取得しました")
            return results

        except Exception as e:
            logger.error(f"Fear & Greed Indexデータの取得中にエラーが発生しました: {e}")
            raise

    def get_latest_fear_greed_data(self) -> Optional[FearGreedIndexData]:
        """
        最新のFear & Greed Indexデータを取得

        Returns:
            最新のFear & Greed Indexデータ（存在しない場合はNone）
        """
        try:
            result = (
                self.db.query(FearGreedIndexData)
                .order_by(FearGreedIndexData.data_timestamp.desc())
                .first()
            )

            if result:
                logger.info(
                    f"最新のFear & Greed Indexデータを取得しました: {result.value}"
                )
            else:
                logger.warning("Fear & Greed Indexデータが見つかりませんでした")

            return result

        except Exception as e:
            logger.error(
                f"最新のFear & Greed Indexデータの取得中にエラーが発生しました: {e}"
            )
            raise

        except Exception as e:
            logger.error(f"Fear & Greed Indexデータの挿入中にエラーが発生しました: {e}")
            raise

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
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=FearGreedIndexData,
                filters={},
                time_range_column="data_timestamp",
                start_time=start_time,
                end_time=end_time,
                order_by_column="data_timestamp",
                order_asc=True,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"Fear & Greed Indexデータの取得中にエラーが発生しました: {e}")
            raise

    def get_latest_fear_greed_data(self, limit: int = 30) -> List[FearGreedIndexData]:
        """
        最新のFear & Greed Index データを取得

        Args:
            limit: 取得件数制限

        Returns:
            最新のFear & Greed Index データのリスト
        """
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=FearGreedIndexData,
                filters={},
                time_range_column="data_timestamp",
                start_time=None,
                end_time=None,
                order_by_column="data_timestamp",
                order_asc=False,  # 降順で最新データを取得
                limit=limit,
            )

        except Exception as e:
            logger.error(
                f"最新Fear & Greed Indexデータの取得中にエラーが発生しました: {e}"
            )
            raise

    def get_latest_data_timestamp(self) -> Optional[datetime]:
        """
        最新のデータタイムスタンプを取得

        Returns:
            最新のデータタイムスタンプ
        """
        try:
            return self.get_latest_timestamp("data_timestamp")

        except Exception as e:
            logger.error(f"最新データタイムスタンプの取得中にエラーが発生しました: {e}")
            raise

    def get_data_count(self) -> int:
        """
        保存されているデータ件数を取得

        Returns:
            データ件数
        """
        try:
            return self.db.query(FearGreedIndexData).count()

        except Exception as e:
            logger.error(f"データ件数の取得中にエラーが発生しました: {e}")
            raise

    def get_data_range(self) -> dict:
        """
        データの範囲情報を取得

        Returns:
            データ範囲情報（最古・最新のタイムスタンプ、件数）
        """
        try:
            from sqlalchemy import func

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

            return {
                "oldest_data": result.oldest.isoformat() if result.oldest else None,
                "newest_data": result.newest.isoformat() if result.newest else None,
                "total_count": result.count or 0,
            }

        except Exception as e:
            logger.error(f"データ範囲情報の取得中にエラーが発生しました: {e}")
            raise

    def delete_old_data(self, before_date: datetime) -> int:
        """
        指定日時より古いデータを削除

        Args:
            before_date: この日時より古いデータを削除

        Returns:
            削除された件数
        """
        try:
            deleted_count = (
                self.db.query(FearGreedIndexData)
                .filter(FearGreedIndexData.data_timestamp < before_date)
                .delete()
            )
            self.db.commit()

            logger.info(
                f"古いFear & Greed Indexデータを {deleted_count} 件削除しました"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"古いデータの削除中にエラーが発生しました: {e}")
            self.db.rollback()
            raise
