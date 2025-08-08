"""
データベース操作の共通ユーティリティ
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DatabaseInsertHelper:
    """データベース挿入処理の共通ヘルパークラス"""

    @staticmethod
    def bulk_insert_with_conflict_handling(
        db: Session,
        model_class: Type,
        records: List[dict],
        conflict_columns: List[str],
        database_url: str,
    ) -> int:
        """
        データベースタイプに応じた重複処理付き一括挿入

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            records: 挿入するレコードのリスト
            conflict_columns: 重複チェック対象のカラム
            database_url: データベースURL

        Returns:
            挿入された件数
        """
        if not records:
            logger.warning("挿入するレコードが指定されていません。")
            return 0

        try:
            if "sqlite" in database_url.lower():
                # SQLiteの場合は一件ずつINSERT OR IGNOREで処理
                # SQLiteの場合は一件ずつINSERT OR IGNOREで処理
                return DatabaseInsertHelper._sqlite_insert_with_ignore(
                    db, model_class, records, conflict_columns
                )
            else:
                # PostgreSQL の ON CONFLICT を使用して重複を無視
                return DatabaseInsertHelper._postgresql_insert_with_conflict(
                    db, model_class, records, conflict_columns
                )

        except Exception as e:
            db.rollback()
            logger.error(f"データベースへのデータ挿入中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def _sqlite_insert_with_ignore(
        db: Session,
        model_class: Type,
        records: List[dict],
        conflict_columns: List[str],
    ) -> int:
        """SQLite用の重複無視挿入（SQLAlchemyのon_conflict_do_nothingを使用）"""
        if not records:
            return 0

        logger.info(f"SQLite一括挿入開始: {len(records)}件のデータを処理")

        try:
            stmt = sqlite_insert(model_class).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
            result = db.execute(stmt)
            db.commit()
            inserted_count = result.rowcount
            logger.info(
                f"SQLite一括挿入完了: {inserted_count}/{len(records)}件挿入"
            )
            return inserted_count
        except Exception as e:
            db.rollback()
            logger.error(f"SQLite一括挿入エラー: {e}")
            # エラーが発生した場合は、元の個別処理にフォールバックすることも検討できる
            # ここではシンプルにエラーをraiseする
            raise

    @staticmethod
    def _postgresql_insert_with_conflict(
        db: Session, model_class: Type, records: List[dict], conflict_columns: List[str]
    ) -> int:
        """PostgreSQL用の重複無視挿入"""
        stmt = postgresql_insert(model_class).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
        result = db.execute(stmt)
        db.commit()
        return result.rowcount


class DatabaseQueryHelper:
    """データベースクエリの共通ヘルパークラス"""

    @staticmethod
    def get_latest_timestamp(
        db: Session,
        model_class: Type,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ):
        """
        最新タイムスタンプを取得

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最新のタイムスタンプ（データがない場合はNone）
        """
        from datetime import timezone

        from sqlalchemy import func

        try:
            query = db.query(func.max(getattr(model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            result = query.scalar()

            # タイムゾーン情報が失われている場合はUTCを設定
            if result and result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result

        except Exception as e:
            logger.error(f"最新タイムスタンプの取得中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def get_oldest_timestamp(
        db: Session,
        model_class: Type,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ):
        """
        最古タイムスタンプを取得

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最古のタイムスタンプ（データがない場合はNone）
        """
        from datetime import timezone

        from sqlalchemy import func

        try:
            query = db.query(func.min(getattr(model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            result = query.scalar()

            # タイムゾーン情報が失われている場合はUTCを設定
            if result and result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result

        except Exception as e:
            logger.error(f"最古タイムスタンプの取得中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def get_record_count(
        db: Session,
        model_class: Type,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        レコード数を取得

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            filter_conditions: フィルター条件

        Returns:
            レコード数
        """
        try:
            query = db.query(model_class)

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            return query.count()

        except Exception as e:
            logger.error(f"レコード数の取得中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def get_date_range(
        db: Session,
        model_class: Type,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ):
        """
        データ期間を取得

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        from datetime import timezone

        from sqlalchemy import func

        try:
            query = db.query(
                func.min(getattr(model_class, timestamp_column)),
                func.max(getattr(model_class, timestamp_column)),
            )

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            result = query.first()

            if result and result[0] and result[1]:
                # タイムゾーン情報が失われている場合はUTCを設定
                oldest = result[0]
                newest = result[1]

                if oldest.tzinfo is None:
                    oldest = oldest.replace(tzinfo=timezone.utc)
                if newest.tzinfo is None:
                    newest = newest.replace(tzinfo=timezone.utc)

                return (oldest, newest)

            return (None, None)

        except Exception as e:
            logger.error(f"データ期間の取得中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def get_filtered_records(
        db: Session,
        model_class: Type,
        filters: Optional[Dict[str, Any]] = None,
        time_range_column: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        order_by_column: Optional[str] = None,
        order_asc: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Any]:
        """
        指定された条件でレコードを取得

        Args:
            db: データベースセッション
            model_class: SQLAlchemyモデルクラス
            filters: 等価フィルター条件（例: {"symbol": "BTC/USDT"}）
            time_range_column: タイムスタンプ範囲の対象カラム名
            start_time: タイムスタンプの開始時刻
            end_time: タイムスタンプの終了時刻
            order_by_column: ソート対象のカラム名
            order_asc: 昇順ソートの場合True、降順の場合False
            limit: 取得件数制限
            offset: オフセット（ページネーション用）

        Returns:
            条件に一致するレコードのリスト
        """
        from sqlalchemy import asc, desc

        try:
            query = db.query(model_class)

            if filters:
                for column, value in filters.items():
                    query = query.filter(getattr(model_class, column) == value)

            if time_range_column:
                if start_time:
                    query = query.filter(
                        getattr(model_class, time_range_column) >= start_time
                    )
                if end_time:
                    query = query.filter(
                        getattr(model_class, time_range_column) <= end_time
                    )

            if order_by_column:
                if order_asc:
                    query = query.order_by(asc(getattr(model_class, order_by_column)))
                else:
                    query = query.order_by(desc(getattr(model_class, order_by_column)))

            if offset:
                query = query.offset(offset)

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"指定された条件でのレコード取得中にエラーが発生しました: {e}")
            raise
