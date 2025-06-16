"""
データ収集ログのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from .base_repository import BaseRepository
from database.models import DataCollectionLog
from app.core.utils.database_utils import DatabaseQueryHelper

logger = logging.getLogger(__name__)


class DataCollectionLogRepository(BaseRepository):
    """データ収集ログのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, DataCollectionLog)

    def log_collection(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        records_collected: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> DataCollectionLog:
        """
        データ収集ログを記録

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 収集開始時刻
            end_time: 収集終了時刻
            records_collected: 収集件数
            status: ステータス
            error_message: エラーメッセージ

        Returns:
            作成されたログレコード
        """
        try:
            log_record = DataCollectionLog(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_collected=records_collected,
                status=status,
                error_message=error_message,
            )

            self.db.add(log_record)
            self.db.commit()
            self.db.refresh(log_record)

            logger.info(f"データ収集ログを記録しました: {symbol} {timeframe} {status}")
            return log_record

        except Exception as e:
            self.db.rollback()
            logger.error(f"データ収集ログ記録エラー: {e}")
            raise

    def get_recent_logs(self, limit: int = 100) -> List[DataCollectionLog]:
        """
        最近のデータ収集ログを取得

        Args:
            limit: 取得件数制限

        Returns:
            データ収集ログのリスト
        """
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=DataCollectionLog,
                order_by_column="created_at",
                order_asc=False,
                limit=limit,
            )

        except Exception as e:
            logger.error(f"データ収集ログ取得エラー: {e}")
            raise
