"""共有ロギングユーティリティ（app/utils/logging.py）のテスト。"""

import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

import pytest

from app.utils.logging import (
    AUTO_STRATEGY_LOGGER_NAME,
    _resolve_level,
    configure_logging,
    get_log_dir,
)


@pytest.fixture
def _restore_root_logger():
    """テスト前後のルートロガーハンドラー状態を退避・復元する。"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


class TestGetLogDir:
    """get_log_dir のテスト。"""

    def test_returns_backend_logs_dir(self):
        log_dir = get_log_dir()
        assert log_dir.name == "logs"
        assert log_dir.is_dir()


class TestResolveLevel:
    """_resolve_level のテスト。"""

    def test_resolves_named_level(self):
        assert _resolve_level("DEBUG") == logging.DEBUG
        assert _resolve_level("ERROR") == logging.ERROR

    def test_invalid_level_falls_back_to_info(self):
        assert _resolve_level("NOT_A_LEVEL") == logging.INFO

    def test_none_uses_config_default(self):
        resolved = _resolve_level(None)
        assert isinstance(resolved, int)


class TestConfigureLogging:
    """configure_logging のテスト。"""

    def test_adds_console_and_file_handlers(self, tmp_path, _restore_root_logger):
        with patch("app.utils.logging.get_log_dir", return_value=tmp_path):
            log_path = configure_logging(log_file="test.log", use_console=True)

        assert log_path == tmp_path / "test.log"
        assert log_path.exists()

        handler_types = [type(h) for h in logging.getLogger().handlers]
        assert logging.StreamHandler in handler_types
        assert RotatingFileHandler in handler_types

        # オートストラテジー専用ロガーが設定される
        auto_logger = logging.getLogger(AUTO_STRATEGY_LOGGER_NAME)
        assert auto_logger.level != logging.NOTSET

    def test_no_file_handler_when_use_file_false(self, tmp_path, _restore_root_logger):
        with patch("app.utils.logging.get_log_dir", return_value=tmp_path):
            log_path = configure_logging(use_file=False)

        assert log_path is None
        handler_types = [type(h) for h in logging.getLogger().handlers]
        assert RotatingFileHandler not in handler_types

    def test_clears_existing_handlers_to_avoid_duplication(
        self, tmp_path, _restore_root_logger
    ):
        # 事前にハンドラーを追加しても、再設定後に重複しないこと
        logging.getLogger().addHandler(logging.NullHandler())

        with patch("app.utils.logging.get_log_dir", return_value=tmp_path):
            configure_logging(log_file="test.log")

        handlers = logging.getLogger().handlers
        handler_types = [type(h) for h in handlers]
        # NullHandler はクリアされ、RotatingFileHandler は1つだけ
        assert logging.NullHandler not in handler_types
        assert handler_types.count(RotatingFileHandler) == 1
