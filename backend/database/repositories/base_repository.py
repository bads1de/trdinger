"""
基底リポジトリクラス
"""

from typing import List, Optional, Type, Dict, Any, TypeVar, Generic, Callable
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func, select, insert, delete, asc, desc
import pandas as pd
import logging


logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """リポジトリの基底クラス"""

    def __init__(self, db: Session, model_class: Type[T]):
        self.db = db
        self.model_class = model_class
        
    def to_dict(self, model_instance: T) -> dict:
        """モデルインスタンスを辞書に変換
        
        Args:
            model_instance: 変換するモデルインスタンス
            
        Returns:
            変換された辞書
        """
        result: dict = {}
        # SQLAlchemyカラムを反復処理して値をシリアライズ
        for column in model_instance.__table__.columns:
            val = getattr(model_instance, column.name)
            # datetime -> ISO形式
            if isinstance(val, datetime):
                result[column.name] = val.isoformat()
            else:
                result[column.name] = val
        return result
    
    def to_pydantic_model(self, model_instance: T, pydantic_model_class: Type) -> Any:
        """モデルインスタンスをPydanticモデルに変換
        
        Args:
            model_instance: 変換するモデルインスタンス
            pydantic_model_class: 変換先のPydanticモデルクラス
            
        Returns:
            変換されたPydanticモデルインスタンス
        """
        # 辞書に変換してからPydanticモデルを作成
        data_dict = self.to_dict(model_instance)
        return pydantic_model_class(**data_dict)

    def bulk_insert_with_conflict_handling(
        self, records: List[Dict[str, Any]], conflict_columns: List[str]
    ) -> int:
        """
        重複処理付き一括挿入（SQLAlchemy 2.0 標準API使用）

        Args:
            records: 挿入するレコードのリスト
            conflict_columns: 重複チェック対象のカラム

        Returns:
            挿入された件数
        """
        if not records:
            logger.warning("挿入するレコードが指定されていません。")
            return 0

        try:
            logger.info(f"一括挿入開始: {len(records)}件のデータを処理")

            # SQLAlchemy 2.0の標準的なinsert文を使用
            stmt = insert(self.model_class)

            # PostgreSQLの場合はon_conflict_do_nothingを使用
            # その他のDBの場合は個別に処理（簡易実装）
            try:
                # PostgreSQL用の重複処理
                stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
                result = self.db.execute(stmt, records)
                inserted_count = getattr(result, "rowcount", 0)
            except AttributeError:
                # PostgreSQL以外のDB用の代替処理
                logger.info("PostgreSQL以外のDBを使用中、個別挿入処理を実行")
                inserted_count = 0
                for i, record in enumerate(records):
                    try:
                        result = self.db.execute(stmt, record)
                        # ドライバによってrowcountがない場合があるため、フォールバックを用意
                        # rowcountが-1を返すドライバもあるため、0でないことをもって成功とみなす
                        if getattr(result, "rowcount", 0) != 0:
                            inserted_count += 1
                            logger.debug(f"レコード {i+1} 挿入成功")
                        else:
                            logger.debug(f"レコード {i+1} 挿入失敗（挿入件数0）")
                    except Exception as e:
                        # 重複エラーは無視
                        logger.debug(f"レコード {i+1} 挿入エラー（無視）: {e}")
                        continue

            self.db.commit()
            logger.info(f"一括挿入完了: {inserted_count}/{len(records)}件挿入")
            return inserted_count

        except Exception:
            from app.utils.error_handler import safe_operation

            @safe_operation(context="データ挿入", is_api_call=False)
            def _handle_bulk_insert_error():
                self.db.rollback()
                raise

            _handle_bulk_insert_error()

    def get_latest_timestamp(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最新タイムスタンプを取得（SQLAlchemy 2.0 標準API使用）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最新のタイムスタンプ（データがない場合はNone）
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = select(func.max(getattr(self.model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            result = self.db.scalar(stmt)

            # タイムゾーン情報が失われている場合はUTCを設定
            if result and result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result

        except Exception:
            from app.utils.error_handler import safe_operation

            @safe_operation(context="最新タイムスタンプ取得", is_api_call=False)
            def _handle_latest_timestamp_error():
                raise

            _handle_latest_timestamp_error()

    def get_oldest_timestamp(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最古タイムスタンプを取得（SQLAlchemy 2.0 標準API使用）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最古のタイムスタンプ（データがない場合はNone）
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = select(func.min(getattr(self.model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            result = self.db.scalar(stmt)

            # タイムゾーン情報が失われている場合はUTCを設定
            if result and result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result

        except Exception as e:
            logger.error(f"最古タイムスタンプの取得中にエラーが発生しました: {e}")
            raise

    def get_record_count(
        self, filter_conditions: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        レコード数を取得（SQLAlchemy 2.0 標準API使用）

        Args:
            filter_conditions: フィルター条件

        Returns:
            レコード数
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = select(func.count()).select_from(self.model_class)

            if filter_conditions:
                for column, value in filter_conditions.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            return self.db.scalar(stmt) or 0

        except Exception:
            from app.utils.error_handler import safe_operation

            @safe_operation(context="レコード数取得", is_api_call=False)
            def _handle_record_count_error():
                raise

            _handle_record_count_error()

    def get_date_range(
        self, timestamp_column: str, filter_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        データ期間を取得（SQLAlchemy 2.0 標準API使用）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        try:
            oldest = self.get_oldest_timestamp(timestamp_column, filter_conditions)
            latest = self.get_latest_timestamp(timestamp_column, filter_conditions)
            return oldest, latest

        except Exception as e:
            logger.error(f"データ期間の取得中にエラーが発生しました: {e}")
            raise

    def get_available_symbols(self, symbol_column: str = "symbol") -> List[str]:
        """
        利用可能なシンボルのリストを取得

        Args:
            symbol_column: シンボルカラム名

        Returns:
            シンボルのリスト
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = select(getattr(self.model_class, symbol_column)).distinct()
            symbols = self.db.scalars(stmt).all()
            return list(symbols)

        except Exception as e:
            logger.error(f"利用可能シンボル取得エラー: {e}")
            raise

    def _delete_all_records(self) -> int:
        """
        全てのレコードを削除する汎用メソッド（SQLAlchemy 2.0 標準API使用）
        """
        try:
            # SQLAlchemy 2.0の標準的なdelete文を使用
            stmt = delete(self.model_class)
            result = self.db.execute(stmt)
            deleted_count = getattr(result, "rowcount", 0)
            self.db.commit()
            return deleted_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"全てのレコード削除エラー ({self.model_class.__name__}): {e}")
            raise

    def _delete_records_by_filter(self, filter_column: str, filter_value: Any) -> int:
        """
        指定されたカラムと値に基づいてレコードを削除する汎用メソッド（SQLAlchemy 2.0 標準API使用）
        """
        try:
            # SQLAlchemy 2.0の標準的なdelete文を使用
            stmt = delete(self.model_class).where(
                getattr(self.model_class, filter_column) == filter_value
            )
            result = self.db.execute(stmt)
            deleted_count = getattr(result, "rowcount", 0)
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
        offset: Optional[int] = None,
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
            offset: オフセット（ページネーション用）

        Returns:
            フィルタリングされたレコードのリスト
        """
        try:
            # SQLAlchemy 2.0の標準的なselect文を使用
            stmt = select(self.model_class)

            # フィルター条件を適用
            if filters:
                for column, value in filters.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            # 時間範囲フィルターを適用
            if time_range_column:
                if start_time:
                    stmt = stmt.where(
                        getattr(self.model_class, time_range_column) >= start_time
                    )
                if end_time:
                    stmt = stmt.where(
                        getattr(self.model_class, time_range_column) <= end_time
                    )

            # ソート条件を適用
            if order_by_column:
                if order_asc:
                    stmt = stmt.order_by(
                        asc(getattr(self.model_class, order_by_column))
                    )
                else:
                    stmt = stmt.order_by(
                        desc(getattr(self.model_class, order_by_column))
                    )

            # ページネーション
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            return list(self.db.scalars(stmt).all())

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

        except Exception:
            from app.utils.error_handler import safe_operation

            @safe_operation(context="DataFrame変換", is_api_call=False)
            def _handle_dataframe_error():
                raise

            _handle_dataframe_error()

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
            # SQLAlchemy 2.0の標準的なdelete文を使用
            stmt = delete(self.model_class)

            # 時間範囲フィルター
            timestamp_attr = getattr(self.model_class, timestamp_column)
            if start_time:
                stmt = stmt.where(timestamp_attr >= start_time)
            if end_time:
                stmt = stmt.where(timestamp_attr <= end_time)

            # 追加フィルター
            if additional_filters:
                for column, value in additional_filters.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            result = self.db.execute(stmt)
            deleted_count = getattr(result, "rowcount", 0)
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
            # SQLAlchemy 2.0の標準的なselect文を使用
            timestamp_attr = getattr(self.model_class, timestamp_column)

            stmt = select(
                func.count(self.model_class.id).label("total_count"),
                func.min(timestamp_attr).label("oldest_timestamp"),
                func.max(timestamp_attr).label("newest_timestamp"),
            ).select_from(self.model_class)

            # フィルター適用
            if filters:
                for column, value in filters.items():
                    stmt = stmt.where(getattr(self.model_class, column) == value)

            result = self.db.execute(stmt).first()

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
