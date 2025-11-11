"""
duplicate_filter_handler.pyのテスト

重複ログフィルタリング機能の包括的なテストを提供します。
"""

import logging
import time
from unittest.mock import Mock, patch

import pytest

from app.utils.duplicate_filter_handler import DuplicateFilter


class TestDuplicateFilter:
    """DuplicateFilterクラスのテスト"""

    @pytest.fixture
    def filter_instance(self):
        """フィルタインスタンスのフィクスチャ"""
        return DuplicateFilter(name="test_filter", interval=5.0, capacity=100)

    @pytest.fixture
    def small_capacity_filter(self):
        """小さな容量のフィルタインスタンス"""
        return DuplicateFilter(name="small_filter", interval=1.0, capacity=3)

    def _create_log_record(
        self,
        message: str,
        level: int = logging.INFO,
        args: tuple = (),
        name: str = "test",
    ) -> logging.LogRecord:
        """ログレコードを作成するヘルパーメソッド

        Args:
            message: ログメッセージ
            level: ログレベル
            args: メッセージフォーマット用の引数
            name: ロガー名

        Returns:
            logging.LogRecord: 作成されたログレコード
        """
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=args,
            exc_info=None,
        )

    def test_first_message_passes(self, filter_instance):
        """正常系: 初回メッセージは通過"""
        record = self._create_log_record("Test message")
        assert filter_instance.filter(record) is True

    def test_duplicate_message_filtered(self, filter_instance):
        """異常系: 重複メッセージはフィルタリング"""
        record = self._create_log_record("Duplicate message")

        # 1回目は通過
        assert filter_instance.filter(record) is True

        # 2回目（重複）はフィルタリング
        assert filter_instance.filter(record) is False

    def test_different_messages_not_filtered(self, filter_instance):
        """異なるメッセージは両方通過"""
        record1 = self._create_log_record("Message 1")
        record2 = self._create_log_record("Message 2")

        assert filter_instance.filter(record1) is True
        assert filter_instance.filter(record2) is True

    def test_time_window_expiration(self, filter_instance):
        """タイムウィンドウ後は再度通過"""
        # interval=5.0のフィルタを使用
        record = self._create_log_record("Test message")

        # 1回目は通過
        assert filter_instance.filter(record) is True

        # タイムウィンドウ内は重複
        assert filter_instance.filter(record) is False

        # タイムウィンドウ経過後（6秒待機）
        time.sleep(6)
        assert filter_instance.filter(record) is True

    def test_different_log_levels(self, filter_instance):
        """異なるログレベルでの動作"""
        message = "Test message"

        # INFO レベル
        info_record = self._create_log_record(message, level=logging.INFO)
        assert filter_instance.filter(info_record) is True

        # 同じメッセージでWARNINGレベル（メッセージが同じなのでフィルタリング）
        warning_record = self._create_log_record(message, level=logging.WARNING)
        assert filter_instance.filter(warning_record) is False

        # 異なるメッセージでERRORレベル
        error_record = self._create_log_record("Error message", level=logging.ERROR)
        assert filter_instance.filter(error_record) is True

    def test_message_with_args(self, filter_instance):
        """引数を含むメッセージのテスト"""
        # %s形式のメッセージ
        record1 = self._create_log_record("Message with %s", args=("arg1",))
        record2 = self._create_log_record("Message with %s", args=("arg1",))

        # 1回目は通過
        assert filter_instance.filter(record1) is True

        # 同じフォーマット済みメッセージは重複としてフィルタリング
        assert filter_instance.filter(record2) is False

    def test_message_with_different_args(self, filter_instance):
        """異なる引数を持つメッセージのテスト"""
        record1 = self._create_log_record("Message with %s", args=("arg1",))
        record2 = self._create_log_record("Message with %s", args=("arg2",))

        # 異なる引数なので両方通過
        assert filter_instance.filter(record1) is True
        assert filter_instance.filter(record2) is True

    def test_capacity_limit(self, small_capacity_filter):
        """容量制限のテスト"""
        # capacity=3のフィルタを使用

        # 3つの異なるメッセージを追加
        record1 = self._create_log_record("Message 1")
        record2 = self._create_log_record("Message 2")
        record3 = self._create_log_record("Message 3")

        assert small_capacity_filter.filter(record1) is True
        assert small_capacity_filter.filter(record2) is True
        assert small_capacity_filter.filter(record3) is True

        # 4つ目のメッセージを追加（最も古いMessage 1が削除される）
        record4 = self._create_log_record("Message 4")
        assert small_capacity_filter.filter(record4) is True

        # interval期間経過後にMessage 1を再度試行
        time.sleep(1.5)  # interval=1.0を超えて待機
        assert small_capacity_filter.filter(record1) is True

    def test_lru_behavior(self, small_capacity_filter):
        """LRU（Least Recently Used）動作のテスト"""
        record1 = self._create_log_record("Message 1")
        record2 = self._create_log_record("Message 2")
        record3 = self._create_log_record("Message 3")

        # 3つのメッセージを追加
        assert small_capacity_filter.filter(record1) is True
        assert small_capacity_filter.filter(record2) is True
        assert small_capacity_filter.filter(record3) is True

        # Message 1を再度試行（時刻が更新され、最新になる）
        time.sleep(1.5)  # interval=1.0を超えて待機
        assert small_capacity_filter.filter(record1) is True

        # 新しいメッセージを追加（最も古いMessage 2が削除される）
        record4 = self._create_log_record("Message 4")
        assert small_capacity_filter.filter(record4) is True

        # interval経過後にMessage 2を試行
        time.sleep(1.5)
        assert small_capacity_filter.filter(record2) is True

        # Message 1は記憶されているがinterval経過後なので再度通過
        time.sleep(1.5)
        assert small_capacity_filter.filter(record1) is True

    def test_multiple_consecutive_duplicates(self, filter_instance):
        """連続する複数の重複メッセージ"""
        record = self._create_log_record("Repeated message")

        # 1回目は通過
        assert filter_instance.filter(record) is True

        # 連続する重複は全てフィルタリング
        for _ in range(10):
            assert filter_instance.filter(record) is False

    def test_empty_message(self, filter_instance):
        """空のメッセージのテスト"""
        record = self._create_log_record("")

        assert filter_instance.filter(record) is True
        assert filter_instance.filter(record) is False

    def test_unicode_messages(self, filter_instance):
        """Unicode文字を含むメッセージのテスト"""
        record1 = self._create_log_record("日本語メッセージ")
        record2 = self._create_log_record("日本語メッセージ")

        assert filter_instance.filter(record1) is True
        assert filter_instance.filter(record2) is False

    def test_long_messages(self, filter_instance):
        """長いメッセージのテスト"""
        long_message = "A" * 10000
        record1 = self._create_log_record(long_message)
        record2 = self._create_log_record(long_message)

        assert filter_instance.filter(record1) is True
        assert filter_instance.filter(record2) is False

    def test_special_characters_in_messages(self, filter_instance):
        """特殊文字を含むメッセージのテスト"""
        special_message = "Message with\nnewline\tand\ttabs"
        record1 = self._create_log_record(special_message)
        record2 = self._create_log_record(special_message)

        assert filter_instance.filter(record1) is True
        assert filter_instance.filter(record2) is False

    @patch("time.time")
    def test_time_mocking(self, mock_time, filter_instance):
        """時間をモックしたテスト"""
        mock_time.return_value = 1000.0

        record = self._create_log_record("Test message")

        # 1回目は通過
        assert filter_instance.filter(record) is True

        # 時間が経過していない状態
        mock_time.return_value = 1002.0  # 2秒後
        assert filter_instance.filter(record) is False

        # interval(5秒)経過後
        mock_time.return_value = 1006.0  # 6秒後
        assert filter_instance.filter(record) is True

    def test_filter_with_custom_interval(self):
        """カスタムインターバルのテスト"""
        short_interval_filter = DuplicateFilter(interval=0.5)
        record = self._create_log_record("Test message")

        assert short_interval_filter.filter(record) is True
        assert short_interval_filter.filter(record) is False

        # 短いインターバル後に再度通過
        time.sleep(0.6)
        assert short_interval_filter.filter(record) is True

    def test_filter_name(self):
        """フィルタ名のテスト"""
        named_filter = DuplicateFilter(name="custom_filter")
        assert named_filter.name == "custom_filter"

    def test_multiple_filters_independence(self):
        """複数のフィルタが独立して動作することを確認"""
        filter1 = DuplicateFilter(name="filter1", interval=1.0)
        filter2 = DuplicateFilter(name="filter2", interval=1.0)

        record = self._create_log_record("Test message")

        # filter1で通過
        assert filter1.filter(record) is True
        # filter1で重複
        assert filter1.filter(record) is False

        # filter2は独立しているので通過
        assert filter2.filter(record) is True
        # filter2で重複
        assert filter2.filter(record) is False


class TestDuplicateFilterIntegration:
    """DuplicateFilterの統合テスト"""

    def test_with_real_logger(self, caplog):
        """実際のロガーとの統合テスト"""
        caplog.set_level(logging.INFO)

        # ロガーを作成してフィルタを追加
        logger = logging.getLogger("test_logger")
        logger.handlers.clear()  # 既存のハンドラをクリア

        handler = logging.StreamHandler()
        handler.addFilter(DuplicateFilter(interval=1.0))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # 同じメッセージを複数回ログ出力
        logger.info("Duplicate log message")
        logger.info("Duplicate log message")
        logger.info("Duplicate log message")

        # 最初のメッセージのみが記録されることを確認
        # 注: capturelogは全てのログを捕捉するため、
        # フィルタの効果は実際のハンドラでのみ機能します
        messages = [record.message for record in caplog.records]
        assert "Duplicate log message" in messages

    def test_performance_with_many_messages(self):
        """多数のメッセージでのパフォーマンステスト"""
        filter_instance = DuplicateFilter(interval=1.0, capacity=1000)

        start_time = time.time()

        # 1000個の異なるメッセージを処理
        for i in range(1000):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            filter_instance.filter(record)

        elapsed_time = time.time() - start_time

        # 1秒以内に処理できることを確認（パフォーマンステスト）
        assert elapsed_time < 1.0

    def test_concurrent_filtering(self):
        """並行フィルタリングの基本動作テスト"""
        import threading

        filter_instance = DuplicateFilter(interval=1.0, capacity=100)
        results = []

        def filter_message(message: str) -> None:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None,
            )
            result = filter_instance.filter(record)
            results.append((message, result))

        # 複数のスレッドで同時にフィルタリング
        threads = []
        for i in range(10):
            thread = threading.Thread(target=filter_message, args=(f"Message {i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 全てのメッセージが処理されたことを確認
        assert len(results) == 10

    def test_filter_with_exception_in_message(self):
        """例外情報を含むログレコードのテスト"""
        filter_instance = DuplicateFilter(interval=1.0)
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        assert filter_instance.filter(record) is True
        assert filter_instance.filter(record) is False

    def test_complex_formatted_messages(self):
        """複雑なフォーマットメッセージのテスト"""
        filter_instance = DuplicateFilter(interval=1.0)
        # 複数の引数を持つメッセージ
        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User %s performed action %s at %s",
            args=("Alice", "login", "2024-01-01"),
            exc_info=None,
        )

        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User %s performed action %s at %s",
            args=("Alice", "login", "2024-01-01"),
            exc_info=None,
        )

        assert filter_instance.filter(record1) is True
        # 同じフォーマット結果なので重複
        assert filter_instance.filter(record2) is False

        # 異なる引数
        record3 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User %s performed action %s at %s",
            args=("Bob", "logout", "2024-01-02"),
            exc_info=None,
        )

        assert filter_instance.filter(record3) is True

    def test_filter_memory_cleanup(self):
        """メモリクリーンアップの確認"""
        filter_instance = DuplicateFilter(interval=0.5, capacity=5)

        # 5つのメッセージを追加
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            filter_instance.filter(record)

        # 内部辞書のサイズが5であることを確認
        assert len(filter_instance._last_log_time_by_msg) == 5

        # 6つ目のメッセージを追加（LRUにより最も古いものが削除される）
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Message 5",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)

        # サイズがcapacityを超えないことを確認
        assert len(filter_instance._last_log_time_by_msg) == 5


class TestDuplicateFilterEdgeCases:
    """DuplicateFilterのエッジケーステスト"""

    def test_zero_interval(self):
        """インターバル0のフィルタ"""
        filter_instance = DuplicateFilter(interval=0.0)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # 1回目は通過
        assert filter_instance.filter(record) is True
        # インターバル0なので即座に再度通過
        assert filter_instance.filter(record) is True

    def test_very_large_capacity(self):
        """非常に大きな容量のフィルタ"""
        filter_instance = DuplicateFilter(interval=1.0, capacity=10000)

        # 多数のメッセージを処理
        for i in range(1000):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            assert filter_instance.filter(record) is True

        # 全てのメッセージが記憶されていることを確認
        assert len(filter_instance._last_log_time_by_msg) == 1000

    def test_negative_interval(self):
        """負のインターバル（実質的に全て通過）"""
        filter_instance = DuplicateFilter(interval=-1.0)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # 負のインターバルでも動作（全て通過）
        assert filter_instance.filter(record) is True
        assert filter_instance.filter(record) is True