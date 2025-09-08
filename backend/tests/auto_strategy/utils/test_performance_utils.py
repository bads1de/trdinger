"""PerformanceUtilsのユニットテスト"""

import pytest
import time
from unittest.mock import patch, MagicMock
from app.services.auto_strategy.utils.performance_utils import PerformanceUtils


class TestPerformanceUtils:
    """PerformanceUtilsクラスのテスト"""

    @pytest.fixture
    def mock_logging_utils(self):
        """LoggingUtilsのモック"""
        with patch("app.services.auto_strategy.utils.performance_utils.LoggingUtils") as mock:
            yield mock

    @pytest.fixture
    def sample_func(self):
        """テスト用のサンプル関数"""
        def test_func(x, y=10):
            time.sleep(0.1)  # シミュレーション
            return x + y
        return test_func

    def test_time_function_normal_execution(self, mock_logging_utils, sample_func):
        """time_functionデコレータが正常実行時に時間を測定するテスト"""
        # デコレータ適用
        decorated_func = PerformanceUtils.time_function(sample_func)
        
        # モック設定
        mock_logging_utils.log_performance = MagicMock()
        
        # 実行
        result = decorated_func(5, y=3)
        
        # 検証
        assert result == 8  # 関数結果が正しい
        assert mock_logging_utils.log_performance.called
        # 呼び出し引数を検証
        call_args = mock_logging_utils.log_performance.call_args
        assert call_args[0][0] == "test_func"
        # 時間は0.1秒以上かかるはず
        duration = call_args[0][1]
        assert duration > 0
        assert duration >= 0.1  # スリープ時間以上

    def test_time_function_exception_execution(self, mock_logging_utils):
        """time_functionデコレータが例外発生時に時間を測定するテスト"""
        # 例外を投げる関数
        @PerformanceUtils.time_function
        def failing_func():
            time.sleep(0.05)
            raise ValueError("Test exception")
        
        # モック設定
        mock_logging_utils.log_performance = MagicMock()
        
        # 実行（例外が発生するはず）
        with pytest.raises(ValueError, match="Test exception"):
            failing_func()
        
        # 検証
        assert mock_logging_utils.log_performance.called
        call_args = mock_logging_utils.log_performance.call_args
        func_name = call_args[0][0]
        duration = call_args[0][1]
        
        assert "(ERROR)" in func_name
        assert duration > 0
        assert duration >= 0.05

    def test_time_function_preserves_metadata(self, mock_logging_utils):
        """time_functionデコレータが関数メタデータを保存するテスト"""
        def original_func():
            """This is a docstring"""
            pass
        
        decorated = PerformanceUtils.time_function(original_func)
        
        # functools.wrapsによってメタデータが保存されるはず
        assert decorated.__name__ == "original_func"
        assert decorated.__doc__ == "This is a docstring"

    def test_time_function_with_mock_timer(self, mock_logging_utils):
        """モックを使って時間を制御するテスト"""
        # time.timeをモック化
        with patch("app.services.auto_strategy.utils.performance_utils.time") as mock_time:
            mock_time.time.side_effect = [1.0, 1.5]  # 開始1秒、終了1.5秒
            
            @PerformanceUtils.time_function
            def quick_func():
                return "done"
            
            mock_logging_utils.log_performance = MagicMock()
            
            result = quick_func()
            
            assert result == "done"
            mock_logging_utils.log_performance.assert_called_once_with("quick_func", 0.5)