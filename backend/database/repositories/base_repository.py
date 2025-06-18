"""
基底リポジトリクラス
"""

from typing import List, Optional, Type, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.core.utils.database_utils import DatabaseInsertHelper, DatabaseQueryHelper

logger = logging.getLogger(__name__)


class BaseRepository:
    """リポジトリの基底クラス"""

    def __init__(self, db: Session, model_class: Type[Any]):
        self.db = db
        self.model_class = model_class

    def bulk_insert_with_conflict_handling(
        self, records: List[Dict[str, Any]], conflict_columns: List[str]
    ) -> int:
        """
        重複処理付き一括挿入

        Args:
            records: 挿入するレコードのリスト
            conflict_columns: 重複チェック対象のカラム

        Returns:
            挿入された件数
        """
        from database.connection import DATABASE_URL

        return DatabaseInsertHelper.bulk_insert_with_conflict_handling(
            self.db, self.model_class, records, conflict_columns, DATABASE_URL
        )

    def get_latest_timestamp(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最新タイムスタンプを取得

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最新のタイムスタンプ（データがない場合はNone）
        """
        return DatabaseQueryHelper.get_latest_timestamp(
            self.db, self.model_class, timestamp_column, filter_conditions
        )

    def get_oldest_timestamp(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最古タイムスタンプを取得

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最古のタイムスタンプ（データがない場合はNone）
        """
        return DatabaseQueryHelper.get_oldest_timestamp(
            self.db, self.model_class, timestamp_column, filter_conditions
        )

    def get_record_count(
        self, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        レコード数を取得

        Args:
            filter_conditions: フィルター条件

        Returns:
            レコード数
        """
        return DatabaseQueryHelper.get_record_count(
            self.db, self.model_class, filter_conditions
        )

    def get_date_range(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        データ期間を取得

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        return DatabaseQueryHelper.get_date_range(
            self.db, self.model_class, timestamp_column, filter_conditions
        )

    def get_available_symbols(self, symbol_column: str = "symbol") -> List[str]:
        """
        利用可能なシンボルのリストを取得

        Args:
            symbol_column: シンボルカラム名

        Returns:
            シンボルのリスト
        """
        try:
            symbols = (
                self.db.query(getattr(self.model_class, symbol_column)).distinct().all()
            )
            return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"利用可能シンボル取得エラー: {e}")
            raise

    def _delete_all_records(self) -> int:
        """
        全てのレコードを削除する汎用メソッド。
        """
        try:
            deleted_count = self.db.query(self.model_class).delete()
            self.db.commit()
            return deleted_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"全てのレコード削除エラー ({self.model_class.__name__}): {e}")
            raise

    def _delete_records_by_filter(self, filter_column: str, filter_value: Any) -> int:
        """
        指定されたカラムと値に基づいてレコードを削除する汎用メソッド。
        """
        try:
            deleted_count = (
                self.db.query(self.model_class)
                .filter(getattr(self.model_class, filter_column) == filter_value)
                .delete()
            )
            self.db.commit()
            return deleted_count
        except Exception as e:
            self.db.rollback()
            logger.error(
                f"レコード削除エラー ({self.model_class.__name__}) by {filter_column}={filter_value}: {e}"
            )
            raise

    def _handle_delete_error(self, e: Exception, message_prefix: str, **kwargs):
        """
        削除時のエラーを処理し、ログを記録する汎用メソッド。
        """
        self.db.rollback()
        error_message = f"{message_prefix}エラー ({self.model_class.__name__}): {e}"
        if kwargs:
            error_message += f" (詳細: {kwargs})"
        logger.error(error_message)
        raise
