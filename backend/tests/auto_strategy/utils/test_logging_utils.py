import pytest
from unittest.mock import patch

# テスト対象モジュールをインポート
from backend.app.services.auto_strategy.utils.logging_utils import LoggingUtils


class TestLoggingUtils:
    """LoggingUtilsクラスのテスト"""

    @pytest.fixture
    def mock_logger(self):
        """ロガーのモックを提供"""
        with patch('backend.app.services.auto_strategy.utils.logging_utils.logger') as mock:
            yield mock

    def test_log_performance_logs_correct_message(self, mock_logger):
        """log_performanceメソッドが正しいメッセージをログに出力するかテスト"""
        LoggingUtils.log_performance("test_operation", 1.234, metric1=10, metric2="value")

        # 期待されるメッセージ
        expected_message = "[PERF] test_operation: 1.234s, metric1=10, metric2=value"

        mock_logger.info.assert_called_once_with(expected_message)

    def test_log_performance_with_no_metrics(self, mock_logger):
        """メトリクスがない場合のログ出力テスト"""
        LoggingUtils.log_performance("simple_operation", 0.5)

        expected_message = "[PERF] simple_operation: 0.500s, "

        mock_logger.info.assert_called_once_with(expected_message)

    def test_log_performance_formats_duration(self, mock_logger):
        """durationが正しく3桁小数でフォーマットされるかテスト"""
        LoggingUtils.log_performance("format_test", 0.123456)

        expected_message = "[PERF] format_test: 0.123s, "

        mock_logger.info.assert_called_once_with(expected_message)
