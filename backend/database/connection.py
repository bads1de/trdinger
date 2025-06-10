"""
データベース接続管理
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)

# データベース設定
DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./trdinger.db"  # 開発環境用にSQLiteを使用
)

# SQLAlchemy エンジンの作成
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,  # SQLログを出力する場合はTrue
)

# セッションファクトリの作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ベースクラス
Base = declarative_base()


def get_db():
    """
    データベースセッションを取得

    Yields:
        Session: データベースセッション
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    データベースを初期化
    """
    try:
        # モデルクラスをインポートしてメタデータに登録
        from .models import (
            OHLCVData,
            FundingRateData,
            OpenInterestData,
            DataCollectionLog,
            TechnicalIndicatorData,
            GAExperiment,
            GeneratedStrategy,
            StrategyShowcase,
        )

        # テーブルを作成
        Base.metadata.create_all(bind=engine)
        logger.info("データベースの初期化が完了しました")
    except Exception as e:
        logger.error(f"データベース初期化エラー: {e}")
        raise


def test_connection():
    """
    データベース接続をテスト

    Returns:
        bool: 接続成功の場合True
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("データベース接続テスト成功")
            return True
    except Exception as e:
        logger.error(f"データベース接続テストエラー: {e}")
        return False


def check_db_initialized():
    """
    データベースが初期化されているかチェック

    Returns:
        bool: 初期化済みの場合True
    """
    try:
        with engine.connect() as connection:
            # データベースタイプに応じてテーブル存在確認クエリを切り替え
            if "sqlite" in DATABASE_URL.lower():
                # SQLite用クエリ
                result = connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data'"
                    )
                )
            else:
                # PostgreSQL用クエリ
                result = connection.execute(
                    text("SELECT tablename FROM pg_tables WHERE tablename='ohlcv_data'")
                )

            table_exists = result.fetchone() is not None

            if table_exists:
                logger.info("データベースは既に初期化されています")
                return True
            else:
                logger.info("データベースは初期化されていません")
                return False

    except Exception as e:
        logger.error(f"データベース初期化チェックエラー: {e}")
        return False


def ensure_db_initialized():
    """
    データベースが初期化されていることを保証
    初期化されていない場合は自動的に初期化を実行

    Returns:
        bool: 初期化成功の場合True
    """
    try:
        # 接続テスト
        if not test_connection():
            logger.error("データベース接続に失敗しました")
            return False

        # 初期化チェック
        if check_db_initialized():
            return True

        # 初期化実行
        logger.info("データベースを自動初期化します")
        init_db()

        # 初期化確認
        if check_db_initialized():
            logger.info("データベースの自動初期化が完了しました")
            return True
        else:
            logger.error("データベースの自動初期化に失敗しました")
            return False

    except Exception as e:
        logger.error(f"データベース初期化保証エラー: {e}")
        return False
