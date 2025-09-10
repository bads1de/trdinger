"""performance_utils.py のテストモジュール"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.auto_strategy.utils.performance_utils import PerformanceUtils


class TestPerformanceUtils:
    """PerformanceUtilsクラスのテスト"""

    def test_time_function_decorator_normal_execution(self):
        """正常な関数実行でのパフォーマンス測定テスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            # テスト関数を定義
            @PerformanceUtils.time_function
            def test_function(x, y):
                return x + y

            # 関数実行
            result = test_function(2, 3)

            # 戻り値確認
            assert result == 5

            # LoggingUtils.log_performanceが1回呼び出されたことを確認
            mock_logging_utils.log_performance.assert_called_once()
            args, kwargs = mock_logging_utils.log_performance.call_args
            assert args[0] == "test_function"  # 関数名

            # 測定時間がfloatであることを確認
            assert isinstance(args[1], float)
            assert args[1] >= 0.0

    def test_time_function_decorator_with_exception(self):
        """例外発生時のパフォーマンス測定テスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def failing_function():
                raise ValueError("Test exception")

            # 例外が再生成されることを確認
            with pytest.raises(ValueError, match="Test exception"):
                failing_function()

            # LoggingUtils.log_performanceがエラー用に1回呼び出されたことを確認
            mock_logging_utils.log_performance.assert_called_once()
            args, kwargs = mock_logging_utils.log_performance.call_args
            assert args[0] == "failing_function (ERROR)"

            # 測定時間がfloatであることを確認
            assert isinstance(args[1], float)
            assert args[1] >= 0.0

    def test_time_function_preserves_function_metadata(self):
        """time_functionデコレータが関数のメタデータを保存するかテスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def sample_function():
                """テスト関数"""
                pass

            # __name__ 属性が保存されていること
            assert sample_function.__name__ == "sample_function"

            # ドックストリングが保存されていること
            assert sample_function.__doc__ == "テスト関数"

    def test_time_function_with_args_and_kwargs(self):
        """引数とキーワード引数を持つ関数でのテスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def func_with_args(a, b=None):
                return a + (b or 0)

            # 位置引数のみ
            assert func_with_args(10, 20) == 30

            # キーワード引数含む
            assert func_with_args(10, b=5) == 15

            # log_performanceが2回呼ばれたことを確認
            assert mock_logging_utils.log_performance.call_count == 2

    def test_time_function_execution_time_measurement(self):
        """実行時間の測定が正しいかテスト"""
        import time
        import threading

        results = {"duration": None, "start_time": None, "end_time": None}

        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def delay_function(seconds):
                time.sleep(seconds)
                return "completed"

            # 関数実行
            delay_function(0.01)  # 0.01秒待つ

            # log_performanceの呼び出しを取得
            call_args = mock_logging_utils.log_performance.call_args
            duration = call_args[0][1]  # 2番目の引数（duration）

            # 測定時間が合理的な範囲にあるかチェック
            assert duration >= 0.01  # 最低でもsleep時間
            assert duration < 0.1  # 現実的に1秒未満

    def test_time_function_multiple_calls(self):
        """複数回の関数呼び出しでのテスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def counter_function():
                return 42

            # 3回呼び出し
            for i in range(3):
                assert counter_function() == 42

            # log_performanceが3回呼ばれたことを確認
            assert mock_logging_utils.log_performance.call_count == 3

    # バグ発見用のテスト
    def test_time_function_zero_division_error(self):
        """ZeroDivisionErrorが発生する場合のテスト（潜在的なバグ発見）"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def division_by_zero():
                return 1 / 0

            with pytest.raises(ZeroDivisionError):
                division_by_zero()

            # エラーログが記録されたことを確認
            mock_logging_utils.log_performance.assert_called_once()
            args, kwargs = mock_logging_utils.log_performance.call_args
            assert "division_by_zero (ERROR)" == args[0]

    def test_time_function_very_quick_execution(self):
        """非常に高速な関数実行でのテスト（0除算など潜在的なバグ発見）"""
        import math

        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def very_fast_function():
                return math.sqrt(4)  # 非常に高速な計算

            result = very_fast_function()
            assert result == 2.0

            # log_performanceが呼ばれたことを確認
            mock_logging_utils.log_performance.assert_called_once()

    @patch('time.time')
    def test_time_function_time_reversal_detection(self, mock_time):
        """異常な時間経過の検知（バグ発見用）"""
        # 時間経過が負になる場合のシミュレーション

        class ReversibleTime:
            def __init__(self):
                self.call_count = 0

            def __call__(self):
                self.call_count += 1
                if self.call_count == 1:
                    return 100.0
                else:
                    return 99.0  # 時間が逆行

        mock_time.side_effect = ReversibleTime()

        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def test_func():
                return "test"

            result = test_func()
            assert result == "test"

            # 負の時間が記録されることを確認（これはバグのヒント）
            mock_logging_utils.log_performance.assert_called_once()
            call_args = mock_logging_utils.log_performance.call_args
            duration = call_args[0][1]
            assert duration == -1.0  # これは潜在的な問題を示す

    def test_time_function_with_complex_return_values(self):
        """複雑な戻り値を持つ関数でのテスト"""
        with patch('app.services.auto_strategy.utils.performance_utils.LoggingUtils', autospec=True) as mock_logging_utils:
            @PerformanceUtils.time_function
            def complex_return():
                return {"result": [1, 2, 3], "status": "success", "nested": {"value": 42}}

            result = complex_return()
            assert result["status"] == "success"
            assert result["nested"]["value"] == 42

            mock_logging_utils.log_performance.assert_called_once()