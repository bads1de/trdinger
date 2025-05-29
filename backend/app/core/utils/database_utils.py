"""
データベース操作の共通ユーティリティ
"""

from typing import List, Any, Type
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Table
import logging

logger = logging.getLogger(__name__)


class DatabaseInsertHelper:
    """データベース挿入処理の共通ヘルパークラス"""
    
    @staticmethod
    def bulk_insert_with_conflict_handling(
        db: Session,
        model_class: Type,
        records: List[dict],
        conflict_columns: List[str],
        database_url: str
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
            logger.warning("挿入するデータがありません")
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
            logger.error(f"データ挿入エラー: {e}")
            raise
    
    @staticmethod
    def _sqlite_insert_with_ignore(
        db: Session,
        model_class: Type,
        records: List[dict]
    ) -> int:
        """SQLite用の重複無視挿入"""
        inserted_count = 0
        for record in records:
            try:
                obj = model_class(**record)
                db.add(obj)
                db.commit()
                inserted_count += 1
            except Exception:
                # 重複エラーの場合はロールバックして続行
                db.rollback()
                continue
        return inserted_count
    
    @staticmethod
    def _postgresql_insert_with_conflict(
        db: Session,
        model_class: Type,
        records: List[dict],
        conflict_columns: List[str]
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
        filter_conditions: dict = None
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
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            raise
    
    @staticmethod
    def get_oldest_timestamp(
        db: Session,
        model_class: Type,
        timestamp_column: str,
        filter_conditions: dict = None
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
            logger.error(f"最古タイムスタンプ取得エラー: {e}")
            raise
    
    @staticmethod
    def get_record_count(
        db: Session,
        model_class: Type,
        filter_conditions: dict = None
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
            logger.error(f"レコード数取得エラー: {e}")
            raise
    
    @staticmethod
    def get_date_range(
        db: Session,
        model_class: Type,
        timestamp_column: str,
        filter_conditions: dict = None
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
                func.max(getattr(model_class, timestamp_column))
            )
            
            if filter_conditions:
                for column, value in filter_conditions.items():
                    query = query.filter(getattr(model_class, column) == value)
            
            result = query.first()
            return result if result else (None, None)
            
        except Exception as e:
            logger.error(f"データ期間取得エラー: {e}")
            raise
