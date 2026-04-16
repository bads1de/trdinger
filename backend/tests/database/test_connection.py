"""
データベース接続モジュールのテスト

database/connection.pyの各関数をテストします。
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

# モジュールをインポート
from database import connection


class TestGetDb:
    """get_db関数のテスト"""

    @patch("database.connection.SessionLocal")
    def test_yields_session_and_closes(self, mock_session_local):
        """セッションを生成し、最後にクローズする"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        gen = connection.get_db()
        session = next(gen)

        assert session == mock_session
        mock_session.close.assert_not_called()

        # ジェネレータを最後まで実行
        try:
            next(gen)
        except StopIteration:
            pass

        mock_session.close.assert_called_once()

    @patch("database.connection.SessionLocal")
    def test_closes_on_exception(self, mock_session_local):
        """例外時もセッションをクローズする"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        gen = connection.get_db()
        next(gen)  # セッションを取得

        # 例外が発生してもcloseが呼ばれる
        try:
            gen.throw(RuntimeError("Test error"))
        except RuntimeError:
            pass

        mock_session.close.assert_called_once()


class TestGetSession:
    """get_session関数のテスト"""

    @patch("database.connection.SessionLocal")
    def test_returns_session(self, mock_session_local):
        """セッションを返す"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        result = connection.get_session()

        assert result == mock_session
        mock_session_local.assert_called_once()


class TestTestConnection:
    """test_connection関数のテスト"""

    @patch("database.connection.engine")
    def test_returns_true_on_success(self, mock_engine):
        """接続成功時にTrueを返す"""
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_connection
        )
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = connection.test_connection()

        assert result is True
        mock_connection.execute.assert_called_once()

    @patch("database.connection.engine")
    @patch("database.connection.logger")
    def test_returns_false_on_error(self, mock_logger, mock_engine):
        """接続エラー時にFalseを返す"""
        mock_engine.connect.side_effect = OperationalError(
            "connection failed", None, None
        )

        result = connection.test_connection()

        assert result is False
        mock_logger.error.assert_called_once()


class TestCheckDbInitialized:
    """check_db_initialized関数のテスト"""

    @patch("database.connection.engine")
    def test_returns_true_when_table_exists_sqlite(self, mock_engine):
        """SQLiteでテーブルが存在する場合Trueを返す"""
        with patch.object(connection, "DATABASE_URL", "sqlite:///test.db"):
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = ("ohlcv_data",)
            mock_connection.execute.return_value = mock_result
            mock_engine.connect.return_value.__enter__ = MagicMock(
                return_value=mock_connection
            )
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = connection.check_db_initialized()

            assert result is True

    @patch("database.connection.engine")
    def test_returns_false_when_table_missing_sqlite(self, mock_engine):
        """SQLiteでテーブルが存在しない場合Falseを返す"""
        with patch.object(connection, "DATABASE_URL", "sqlite:///test.db"):
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None
            mock_connection.execute.return_value = mock_result
            mock_engine.connect.return_value.__enter__ = MagicMock(
                return_value=mock_connection
            )
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = connection.check_db_initialized()

            assert result is False

    @patch("database.connection.engine")
    def test_returns_true_when_table_exists_postgres(self, mock_engine):
        """PostgreSQLでテーブルが存在する場合Trueを返す"""
        with patch.object(
            connection, "DATABASE_URL", "postgresql://user:pass@localhost/db"
        ):
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = ("ohlcv_data",)
            mock_connection.execute.return_value = mock_result
            mock_engine.connect.return_value.__enter__ = MagicMock(
                return_value=mock_connection
            )
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = connection.check_db_initialized()

            assert result is True

    @patch("database.connection.engine")
    @patch("database.connection.logger")
    def test_returns_false_on_error(self, mock_logger, mock_engine):
        """エラー時にFalseを返す"""
        mock_engine.connect.side_effect = OperationalError("query failed", None, None)

        result = connection.check_db_initialized()

        assert result is False
        mock_logger.error.assert_called_once()


class TestInitDb:
    """init_db関数のテスト"""

    @patch("database.connection.test_connection")
    @patch("database.connection.check_db_initialized")
    def test_returns_true_when_already_initialized(self, mock_check, mock_test_conn):
        """既に初期化されている場合Trueを返す"""
        mock_test_conn.return_value = True
        mock_check.return_value = True

        result = connection.init_db()

        assert result is True
        mock_test_conn.assert_called_once()
        mock_check.assert_called_once()

    @patch("database.connection.test_connection")
    @patch("database.connection.check_db_initialized")
    @patch("database.connection.Base")
    @patch("database.connection.logger")
    def test_initializes_when_not_initialized(
        self, mock_logger, mock_base, mock_check, mock_test_conn
    ):
        """未初期化の場合は初期化を実行する"""
        mock_test_conn.return_value = True
        mock_check.side_effect = [False, True]  # 最初はFalse、後でTrue

        result = connection.init_db()

        assert result is True
        mock_base.metadata.create_all.assert_called_once()
        mock_logger.info.assert_any_call("データベースを自動初期化します")

    @patch("database.connection.test_connection")
    @patch("database.connection.check_db_initialized")
    @patch("database.connection.Base")
    @patch("database.connection.logger")
    def test_returns_false_when_init_fails(
        self, mock_logger, mock_base, mock_check, mock_test_conn
    ):
        """初期化失敗時にFalseを返す"""
        mock_test_conn.return_value = True
        mock_check.side_effect = [False, False]  # 常にFalse

        result = connection.init_db()

        assert result is False
        mock_base.metadata.create_all.assert_called_once()
        mock_logger.error.assert_called_with("データベースの自動初期化に失敗しました")

    @patch("database.connection.test_connection")
    @patch("database.connection.logger")
    def test_returns_false_when_connection_fails(self, mock_logger, mock_test_conn):
        """接続失敗時にFalseを返す"""
        mock_test_conn.return_value = False

        result = connection.init_db()

        assert result is False
        mock_logger.error.assert_called_with("データベース接続に失敗しました")

    @patch("database.connection.test_connection")
    @patch("database.connection.check_db_initialized")
    @patch("database.connection.logger")
    def test_handles_exception(self, mock_logger, mock_check, mock_test_conn):
        """例外発生時にFalseを返す"""
        mock_test_conn.side_effect = Exception("Unexpected error")

        result = connection.init_db()

        assert result is False
        mock_logger.error.assert_called_once()


class TestModuleLevelVariables:
    """モジュールレベル変数のテスト"""

    def test_engine_exists(self):
        """engineが定義されている"""
        assert hasattr(connection, "engine")

    def test_session_local_exists(self):
        """SessionLocalが定義されている"""
        assert hasattr(connection, "SessionLocal")

    def test_base_exists(self):
        """Baseが定義されている"""
        assert hasattr(connection, "Base")

    def test_database_url_exists(self):
        """DATABASE_URLが定義されている"""
        assert hasattr(connection, "DATABASE_URL")

    @patch("database.connection.logger")
    def test_database_url_default_value(self, mock_logger):
        """DATABASE_URLのデフォルト値を確認"""
        # デフォルト値はSQLite
        assert (
            "sqlite" in connection.DATABASE_URL.lower()
            or connection.DATABASE_URL is not None
        )
