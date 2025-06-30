"""
データベース操作の共通ユーティリティ
"""

from typing import List, Type, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
import logging
from datetime import datetime

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
                return DatabaseInsertHelper._sqlite_insert_with_ignore(
                    db, model_class, records
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
        db: Session, model_class: Type, records: List[dict]
    ) -> int:
        """SQLite用の重複無視挿入（バッチ処理対応）"""
        if not records:
            return 0

        inserted_count = 0
        batch_size = 100  # バッチサイズを設定
        total_records = len(records)

        logger.info(
            f"SQLite一括挿入開始: {total_records}件のデータを{batch_size}件ずつ処理"
        )

        # レコードをバッチに分割して処理
        for i in range(0, total_records, batch_size):
            batch = records[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_records + batch_size - 1) // batch_size

            logger.info(f"バッチ {batch_num}/{total_batches}: {len(batch)}件処理中...")

            try:
                # バッチ内の各レコードを追加
                batch_inserted = 0
                batch_skipped = 0

                for record in batch:
                    try:
                        obj = model_class(**record)
                        db.add(obj)
                        batch_inserted += 1
                    except Exception as e:
                        # 重複エラーなどの場合は該当レコードをスキップ
                        batch_skipped += 1
                        logger.debug(f"レコードスキップ: {e}")
                        continue

                # バッチ単位でコミット
                try:
                    if batch_inserted > 0:
                        db.commit()
                        inserted_count += batch_inserted
                        logger.info(
                            f"バッチ {batch_num}: {batch_inserted}件挿入完了 (スキップ: {batch_skipped}件)"
                        )
                    else:
                        logger.info(
                            f"バッチ {batch_num}: 新規挿入データなし (スキップ: {batch_skipped}件)"
                        )
                except Exception as commit_error:
                    # コミット時のエラー（重複制約など）
                    db.rollback()
                    logger.warning(
                        f"バッチ {batch_num} コミットエラー、個別処理に切り替え: {commit_error}"
                    )

                    # 個別処理で重複を回避
                    individual_inserted = 0
                    for record in batch:
                        try:
                            obj = model_class(**record)
                            db.add(obj)
                            db.commit()
                            individual_inserted += 1
                        except Exception:
                            db.rollback()
                            continue

                    inserted_count += individual_inserted
                    logger.info(
                        f"バッチ {batch_num}: 個別処理で {individual_inserted}件挿入完了"
                    )

            except Exception as e:
                # バッチ全体でエラーが発生した場合はロールバック
                db.rollback()
                logger.warning(f"バッチ {batch_num} でエラー発生、ロールバック: {e}")
                continue

        logger.info(f"SQLite一括挿入完了: {inserted_count}/{total_records}件挿入")
        return inserted_count

    @staticmethod
    def _postgresql_insert_with_conflict(
        db: Session, model_class: Type, records: List[dict], conflict_columns: List[str]
    ) -> int:
        """PostgreSQL用の重複無視挿入"""
        stmt = insert(model_class).values(records)
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
        from sqlalchemy import func

        try:
            query = db.query(func.max(getattr(model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            return query.scalar()

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
        from sqlalchemy import func

        try:
            query = db.query(func.min(getattr(model_class, timestamp_column)))

            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)

            return query.scalar()

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
            return result if result else (None, None)

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

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"指定された条件でのレコード取得中にエラーが発生しました: {e}")
            raise
