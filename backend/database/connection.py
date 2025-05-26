"""
データベース接続管理
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)

# データベース設定
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/trdinger"
)

# SQLAlchemy エンジンの作成
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False  # SQLログを出力する場合はTrue
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
            result = connection.execute("SELECT 1")
            logger.info("データベース接続テスト成功")
            return True
    except Exception as e:
        logger.error(f"データベース接続テストエラー: {e}")
        return False
