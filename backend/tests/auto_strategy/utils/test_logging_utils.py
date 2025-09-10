"""logging_utils.py のテストモジュール"""

import logging
import pytest
from unittest.mock import patch, MagicMock
from app.services.auto_strategy.utils.logging_utils import LoggingUtils


class TestLoggingUtils:
    """LoggingUtilsクラスのテスト"""

    def test_log_performance_without_metrics(self):
        """パフォーマンスログテスト（メトリクスなし）"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            LoggingUtils.log_performance("test_operation", 1.23)

            mock_logger.info.assert_called_once_with("[PERF] test_operation: 1.230s, ")

    def test_log_performance_with_metrics(self):
        """パフォーマンスログテスト（メトリクスあり）"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            metrics = {"db_queries": 45, "cache_hits": 0.85, "memory_used": 128.5}
            LoggingUtils.log_performance("complex_calulation", 2.567, **metrics)

            expected_msg = "[PERF] complex_calulation: 2.567s, cache_hits=0.85, db_queries=45, memory_used=128.5"
            # メトリクス順序がアルファベット順にソートされる: cache_hits, db_queries, memory_used
            mock_logger.info.assert_called_once_with(expected_msg.replace("cache_hits=0.85, db_queries=45, memory_used=128.5",
                                                                          "cache_hits=0.85, db_queries=45, memory_used=128.5"))

    def test_log_performance_float_formatting(self):
        """パフォーマンスログのパフォーマンス値のフォーマットテスト"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            # 小数点以下の表示テスト
            LoggingUtils.log_performance("test_operation", 0.00123)

            mock_logger.info.assert_called_once_with("[PERF] test_operation: 0.001s, ")

            # 整数値のテスト
            mock_logger.reset_mock()
            LoggingUtils.log_performance("test_operation", 5.0)

            mock_logger.info.assert_called_once_with("[PERF] test_operation: 5.000s, ")

    def test_log_performance_with_empty_metrics(self):
        """空のメトリクス辞書でのテスト"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            LoggingUtils.log_performance("empty_metrics_test", 1.0, **{})

            mock_logger.info.assert_called_once_with("[PERF] empty_metrics_test: 1.000s, ")

    def test_log_performance_with_large_numbers(self):
        """大きな数値でのテスト"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            LoggingUtils.log_performance("large_test", 999.9999, very_large_metric=1000000, small_metric=0.000001)

            mock_logger.info.assert_called_once_with("[PERF] large_test: 1000.000s, small_metric=1e-06, very_large_metric=1000000")

    def test_log_performance_zero_duration(self):
        """実行時間が0の場合のテスト"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            LoggingUtils.log_performance("instant_operation", 0.0, result="success")

            mock_logger.info.assert_called_once_with("[PERF] instant_operation: 0.000s, result=success")

    def test_log_performance_negative_duration(self):
        """実行時間が負の値の場合のテスト（エッジケース）"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            # 負の値もフォーマットされるか確認
            LoggingUtils.log_performance("negative_duration_operation", -1.5, error="timing_error")

            mock_logger.info.assert_called_once_with("[PERF] negative_duration_operation: -1.500s, error=timing_error")

    def test_log_performance_with_string_values(self):
        """メトリクスに文字列値を含めるテスト"""
        with patch('app.services.auto_strategy.utils.logging_utils.logger', autospec=True) as mock_logger:
            mock_logger.info = MagicMock()

            LoggingUtils.log_performance("mixed_metrics", 3.14, operation_type="calculation", status="completed")

            mock_logger.info.assert_called_once_with("[PERF] mixed_metrics: 3.140s, operation_type=calculation, status=completed")

    def test_log_performance_logger_instantiation(self):
        """ロガーのインスタンス確認テスト"""
        # importのテスト - ロガーが正しく設定されているか確認
        from app.services.auto_strategy.utils.logging_utils import logger

        assert isinstance(logger, logging.Logger)
        assert logger.name == "app.services.auto_strategy.utils.logging_utils"

    # バグ発見用のテスト
    @patch('app.services.auto_strategy.utils.logging_utils.logger')
    def test_log_performance_with_none_duration(self, mock_logger):
        """durationがNoneの場合のテスト（バグ発見用）"""

        # もし興味があれば実装にNone処理を追加できる
        try:
            LoggingUtils.log_performance("none_duration", None)  # type: ignore
            # 実装によってはTypeErrorが出るはず - バグ発見のヒント
        except TypeError:
            pytest.fail("log_performance が None 値を適切に処理していない")

    @patch('app.services.auto_strategy.utils.logging_utils.logger')
    def test_log_performance_with_non_numeric_duration(self, mock_logger):
        """durationが文字列の場合のテスト（バグ発見用）"""

        # 非数値を渡してどうなるか確認
        try:
            LoggingUtils.log_performance("string_duration", "invalid")  # type: ignore
            # 実装によってはValueErrorやTypeErrorが出るはず
        except (TypeError, ValueError):
            # 例外が出るのは予期される動作
            pass

    @patch('app.services.auto_strategy.utils.logging_utils.logger')
    def test_log_performance_logger_error_handling(self, mock_logger):
        """ロガーが例外を投げる場合のテスト（バグ発見用）"""

        # ロガーが例外を投げるシミュレーション
        mock_logger.info.side_effect = Exception("Logging failed")

        # この場合、関数は内部例外を処理すべきか、
        # または例外が伝播するか - 実装の堅牢性をテスト
        try:
            LoggingUtils.log_performance("error_test", 1.0)
            # 例外が無視されているならここを通る
            # もし伝播するなら except 節でキャッチされる
        except Exception:
            pytest.fail("log_performance がロガー例外を適切に処理していない")

    @patch('app.services.auto_strategy.utils.logging_utils.logger')
    def test_log_performance_with_missing_metrics_keys(self, mock_logger):
        """メトリクスに特殊な値を含む場合のテスト"""

        # **metrics展開時にNone値を渡すテスト
        try:
            LoggingUtils.log_performance("missing_keys", 1.0, None)  # キーワード引数としてNone
            # **metrics がキーワード引数として渡されるため、TypeErrorが発生するはず
        except TypeError:
            # このテストは実装の限界を示す（バグ discovery）
            pass
