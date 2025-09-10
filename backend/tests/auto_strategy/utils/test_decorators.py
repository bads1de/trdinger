"""decorators関数のテストモジュール"""

import logging
import pytest
from unittest.mock import patch
from app.services.auto_strategy.utils.decorators import auto_strategy_operation, with_metrics_tracking

class TestAutoStrategyOperation:
    """auto_strategy_operationデコレータのテスト"""

    def test_auto_strategy_operation_success(self):
        """正常な操作テスト"""
        @auto_strategy_operation(context="Test context", enable_debug_logging=True)
        def sample_func(x, y):
            return x + y

        result = sample_func(1, 2)
        assert result == 3

    def test_auto_strategy_operation_with_exception_non_api(self):
        """非API呼び出しでの例外処理テスト"""
        @auto_strategy_operation(context="Test context", default_return="fallback", is_api_call=False)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "fallback"

    def test_auto_strategy_operation_with_exception_api(self):
        """API呼び出しでの例外処理テスト（例外発生を期待）"""
        from app.utils.error_handler import ErrorHandler

        @auto_strategy_operation(context="Test context", is_api_call=True)
        def failing_func():
            raise ValueError("Test error")

        with patch.object(ErrorHandler, 'handle_api_error') as mock_error:
            mock_error.side_effect = Exception("HTTP Error")
            with pytest.raises(Exception) as exc_info:
                failing_func()
            assert "HTTP Error" in str(exc_info.value)

    def test_auto_strategy_operation_log_level_warning(self):
        """ログレベルwarningのテスト"""
        @auto_strategy_operation(context="Test context", log_level="warning", default_return="ok")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "ok"

    def test_auto_strategy_operation_log_level_info(self):
        """ログレベルinfoのテスト"""
        @auto_strategy_operation(context="Test context", log_level="info", default_return="ok")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "ok"

    def test_auto_strategy_operation_invalid_log_level(self):
        """無効なログレベルのテスト（デフォルトerrorを使用）"""
        @auto_strategy_operation(context="Test context", log_level="invalid", default_return="ok")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "ok"


class TestWithMetricsTracking:
    """with_metrics_trackingデコレータのテスト"""

    def test_with_metrics_tracking_success_without_memory(self):
        """メモリ追跡なしの成功テスト"""
        @with_metrics_tracking("test_operation", track_memory=False)
        def sample_func():
            return "success"

        result = sample_func()
        assert result == "success"

    def test_with_metrics_tracking_success_with_memory_psutil_available(self):
        """psutil利用可能な場合のメモリ追跡テスト"""
        from unittest.mock import Mock, patch

        mock_psutil = Mock()
        mock_psutil.Process.return_value.memory_info.return_value.rss = 1000
        mock_os = Mock()
        mock_os.getpid.return_value = 123

        with patch.dict('sys.modules', {'psutil': mock_psutil, 'os': mock_os}):
            @with_metrics_tracking("test_operation", track_memory=True)
            def sample_func():
                return "success"

            result = sample_func()
            assert result == "success"

    def test_with_metrics_tracking_psutil_import_error(self):
        """psutil importエラー時のエレー際テスト（バグ発見目的）"""
        with patch.dict('sys.modules', {'psutil': None}):
            @with_metrics_tracking("test_operation", track_memory=True)
            def sample_func():
                return "success"

            # psutilが使えない場合も正常動作するかテスト
            result = sample_func()
            assert result == "success"