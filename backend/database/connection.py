"""
データベース接続管理
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)

# .env の自動読み込み（backend直下の.envを優先）
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# データベース設定
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./trdinger.db",  # 開発環境用にSQLiteを使用（デフォルトは相対パス）
)

# データベースの種類に応じてエンジン設定を最適化
# SQLiteはコネクションプーリングに対応しておらず、並列書き込み時に
# "database is locked" エラーが発生するため NullPool を使用する
if DATABASE_URL.lower().startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    logger.info("SQLiteデータベースを使用しています（NullPool設定）")
else:
    # PostgreSQLなど本番環境用
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )
    logger.info("PostgreSQLデータベースを使用しています（QueuePool設定）")

# セッションファクトリの作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ベースクラス
Base = declarative_base()


def get_db():
    """
    データベースセッションを生成・管理するジェネレーターです。

    FastAPI の `Depends(get_db)` を通じて使用されることを想定しています。
    リクエストごとに新しいセッションを作成し、処理完了後に自動的にクローズします。

    Yields:
        Session: SQLAlchemy のデータベースセッション。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session():
    """
    スクリプト用のデータベースセッションを取得

    Returns:
        Session: データベースセッション
    """
    return SessionLocal()


def init_db():
    """
    アプリケーション起動時にデータベースの初期化（テーブル作成）を保証します。

    この関数は以下の順序で動作します：
    1. `test_connection()` によりデータベースサーバー（またはファイル）への接続を確認。
    2. `check_db_initialized()` により `ohlcv_data` テーブルの存在を確認。
    3. 未初期化の場合、`Base.metadata.create_all()` を実行して定義済みの全テーブルを作成。

    Returns:
        bool: 全てのステップが正常に完了し、データベースが利用可能な状態であれば True。
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
        Base.metadata.create_all(bind=engine)

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


def test_connection():
    """データベース接続をテストします。

    Returns:
        bool: 接続成功の場合True。
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            logger.info("データベース接続テスト成功")
            return True
    except Exception as e:
        logger.error(f"データベース接続テストエラー: {e}")
        return False


def check_db_initialized():
    """データベースが初期化されているかチェックします。

    Returns:
        bool: 初期化済みの場合True。
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



