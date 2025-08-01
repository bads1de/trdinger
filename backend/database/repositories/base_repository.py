"""
基底リポジトリクラス
"""

from typing import List, Optional, Type, Dict, Any, TypeVar, Generic, Callable
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd
import logging

from app.utils.database_utils import DatabaseInsertHelper, DatabaseQueryHelper


logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """リポジトリの基底クラス"""

    def __init__(self, db: Session, model_class: Type[T]):
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

    # 汎用データ取得メソッド
    def get_filtered_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        time_range_column: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        order_by_column: Optional[str] = None,
        order_asc: bool = True,
        limit: Optional[int] = None,
    ) -> List[T]:
        """
        汎用的なフィルタリングデータ取得メソッド

        Args:
            filters: フィルター条件
            time_range_column: 時間範囲フィルター用のカラム名
            start_time: 開始時刻
            end_time: 終了時刻
            order_by_column: ソート用カラム名
            order_asc: 昇順ソートかどうか
            limit: 取得件数制限

        Returns:
            フィルタリングされたレコードのリスト
        """
        try:
            return DatabaseQueryHelper.get_filtered_records(
                db=self.db,
                model_class=self.model_class,
                filters=filters or {},
                time_range_column=time_range_column,
                start_time=start_time,
                end_time=end_time,
                order_by_column=order_by_column,
                order_asc=order_asc,
                limit=limit,
            )
        except Exception as e:
            logger.error(
                f"フィルタリングデータ取得エラー ({self.model_class.__name__}): {e}"
            )
            raise

    def get_latest_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
        timestamp_column: str = "timestamp",
        limit: int = 100,
    ) -> List[T]:
        """
        最新レコードを取得する汎用メソッド

        Args:
            filters: フィルター条件
            timestamp_column: タイムスタンプカラム名
            limit: 取得件数制限

        Returns:
            最新レコードのリスト（新しい順）
        """
        return self.get_filtered_data(
            filters=filters,
            time_range_column=timestamp_column,
            order_by_column=timestamp_column,
            order_asc=False,
            limit=limit,
        )

    def get_data_in_range(
        self,
        timestamp_column: str,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[T]:
        """
        期間指定でデータを取得する汎用メソッド

        Args:
            timestamp_column: タイムスタンプカラム名
            start_time: 開始時刻
            end_time: 終了時刻
            filters: 追加フィルター条件
            limit: 取得件数制限

        Returns:
            期間内のレコードのリスト
        """
        return self.get_filtered_data(
            filters=filters,
            time_range_column=timestamp_column,
            start_time=start_time,
            end_time=end_time,
            order_by_column=timestamp_column,
            order_asc=True,
            limit=limit,
        )

    def to_dataframe(
        self,
        records: List[T],
        column_mapping: Optional[Dict[str, str]] = None,
        index_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        レコードをDataFrameに変換する汎用メソッド

        Args:
            records: 変換するレコードのリスト
            column_mapping: カラム名のマッピング（model_attr -> df_column）
            index_column: インデックスに設定するカラム名

        Returns:
            変換されたDataFrame
        """
        if not records:
            # 空のDataFrameを返す
            columns = list(column_mapping.values()) if column_mapping else []
            return pd.DataFrame(columns=columns)

        try:
            data = []
            for record in records:
                row_data = {}
                if column_mapping:
                    # カラムマッピングが指定されている場合
                    for model_attr, df_column in column_mapping.items():
                        row_data[df_column] = getattr(record, model_attr, None)
                else:
                    # マッピングが指定されていない場合、全属性を使用
                    for column in self.model_class.__table__.columns:
                        row_data[column.name] = getattr(record, column.name, None)

                data.append(row_data)

            df = pd.DataFrame(data)

            # インデックスを設定
            if index_column and index_column in df.columns:
                df.set_index(index_column, inplace=True)

            return df

        except Exception as e:
            logger.error(f"DataFrame変換エラー ({self.model_class.__name__}): {e}")
            raise

    def delete_by_date_range(
        self,
        timestamp_column: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        期間指定でレコードを削除する汎用メソッド

        Args:
            timestamp_column: タイムスタンプカラム名
            start_time: 削除開始時刻（この時刻以降を削除）
            end_time: 削除終了時刻（この時刻以前を削除）
            additional_filters: 追加フィルター条件

        Returns:
            削除された件数
        """
        try:
            query = self.db.query(self.model_class)

            # 時間範囲フィルター
            timestamp_attr = getattr(self.model_class, timestamp_column)
            if start_time:
                query = query.filter(timestamp_attr >= start_time)
            if end_time:
                query = query.filter(timestamp_attr <= end_time)

            # 追加フィルター
            if additional_filters:
                for column, value in additional_filters.items():
                    query = query.filter(getattr(self.model_class, column) == value)

            deleted_count = query.delete()
            self.db.commit()

            logger.info(
                f"期間指定削除完了 ({self.model_class.__name__}): {deleted_count}件"
            )
            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"期間指定削除エラー ({self.model_class.__name__}): {e}")
            raise

    def delete_old_data(
        self,
        timestamp_column: str,
        before_date: datetime,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        指定日時より古いデータを削除する汎用メソッド

        Args:
            timestamp_column: タイムスタンプカラム名
            before_date: この日時より古いデータを削除
            additional_filters: 追加フィルター条件

        Returns:
            削除された件数
        """
        return self.delete_by_date_range(
            timestamp_column=timestamp_column,
            end_time=before_date,
            additional_filters=additional_filters,
        )

    def get_data_statistics(
        self,
        timestamp_column: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        データの統計情報を取得する汎用メソッド

        Args:
            timestamp_column: タイムスタンプカラム名
            filters: フィルター条件

        Returns:
            統計情報の辞書
        """
        try:
            query = self.db.query(self.model_class)

            # フィルター適用
            if filters:
                for column, value in filters.items():
                    query = query.filter(getattr(self.model_class, column) == value)

            timestamp_attr = getattr(self.model_class, timestamp_column)

            result = query.with_entities(
                func.count(self.model_class.id).label("total_count"),
                func.min(timestamp_attr).label("oldest_timestamp"),
                func.max(timestamp_attr).label("newest_timestamp"),
            ).first()

            return {
                "total_count": result.total_count or 0,
                "oldest_timestamp": result.oldest_timestamp,
                "newest_timestamp": result.newest_timestamp,
                "date_range_days": (
                    (result.newest_timestamp - result.oldest_timestamp).days
                    if result.oldest_timestamp and result.newest_timestamp
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"統計情報取得エラー ({self.model_class.__name__}): {e}")
            raise

    def validate_records(
        self,
        records: List[Dict[str, Any]],
        required_fields: List[str],
        validation_func: Optional[Callable[[List[Dict[str, Any]]], bool]] = None,
    ) -> bool:
        """
        レコードの妥当性を検証する汎用メソッド

        Args:
            records: 検証するレコードのリスト
            required_fields: 必須フィールドのリスト
            validation_func: カスタム検証関数

        Returns:
            有効な場合True、無効な場合False
        """
        try:
            if not records:
                return True

            # 必須フィールドの検証
            for record in records:
                for field in required_fields:
                    if field not in record or record[field] is None:
                        logger.warning(f"必須フィールド '{field}' が不足しています")
                        return False

            # カスタム検証関数の実行
            if validation_func:
                return validation_func(records)

            return True

        except Exception as e:
            logger.error(f"レコード検証エラー ({self.model_class.__name__}): {e}")
            return False
