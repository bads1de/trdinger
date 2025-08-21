
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import Mock
from backend.database.connection import (
    Base,
    get_db,
    init_db,
    test_connection,
    check_db_initialized,
)
from backend.database.models import OHLCVData # check_db_initializedで参照されるため

# テスト用のインメモリSQLiteデータベース設定
@pytest.fixture(name="test_engine")
def test_engine_fixture():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture(name="test_session")
def test_session_fixture(test_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine) # 各テスト後にテーブルを削除


@pytest.fixture(name="mock_get_db")
def mock_get_db_fixture(test_session):
    def override_get_db():
        yield test_session

    return override_get_db


class TestConnectionFunctions:
    """
    connection.py内の関数群のテストクラス
    """

    def test_test_connection_success(self, test_engine, monkeypatch):
        """データベース接続テスト成功のテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        assert test_connection() is True

    def test_test_connection_failure(self, monkeypatch):
        """データベース接続テスト失敗のテスト"""
        # 意図的に接続を失敗させるモック
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        monkeypatch.setattr("backend.database.connection.engine", mock_engine)
        assert test_connection() is False

    def test_init_db(self, test_engine, monkeypatch):
        """データベース初期化のテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        monkeypatch.setattr("backend.database.connection.Base", Base)
        # テーブルが存在しないことを確認
        inspector = inspect(test_engine)
        assert not inspector.has_table(OHLCVData.__tablename__)
        init_db()
        # テーブルが作成されたことを確認
        assert inspector.has_table(OHLCVData.__tablename__)

    def test_check_db_initialized_true(self, test_engine, monkeypatch):
        """データベースが初期化済みの場合のテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        monkeypatch.setattr("backend.database.connection.DATABASE_URL", "sqlite:///:memory:")
        Base.metadata.create_all(bind=test_engine) # テーブルを作成
        inspector = inspect(test_engine)
        assert inspector.has_table(OHLCVData.__tablename__)
        assert check_db_initialized() is True

    def test_check_db_initialized_false(self, test_engine, monkeypatch):
        """データベースが未初期化の場合のテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        monkeypatch.setattr("backend.database.connection.DATABASE_URL", "sqlite:///:memory:")
        # テーブルは作成しない
        inspector = inspect(test_engine)
        assert not inspector.has_table(OHLCVData.__tablename__)
        assert check_db_initialized() is False

    def test_init_db_already_initialized(self, test_engine, monkeypatch):
        """init_db: 既に初期化済みの場合のテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        monkeypatch.setattr("backend.database.connection.DATABASE_URL", "sqlite:///:memory:")
        Base.metadata.create_all(bind=test_engine) # 事前にテーブルを作成
        assert init_db() is True

    def test_init_db_not_initialized(self, test_engine, monkeypatch):
        """init_db: 未初期化の場合に初期化されるテスト"""
        monkeypatch.setattr("backend.database.connection.engine", test_engine)
        monkeypatch.setattr("backend.database.connection.DATABASE_URL", "sqlite:///:memory:")
        # テーブルは作成しない
        inspector = inspect(test_engine)
        assert not inspector.has_table(OHLCVData.__tablename__)
        assert init_db() is True
        # 初期化されたことを確認
        assert inspector.has_table(OHLCVData.__tablename__)

    def test_init_db_connection_failure(self, monkeypatch):
        """init_db: 接続失敗の場合のテスト"""
        mock_test_connection = Mock(return_value=False)
        monkeypatch.setattr("backend.database.connection.test_connection", mock_test_connection)
        assert init_db() is False
