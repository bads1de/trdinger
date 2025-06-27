"""
重複ログフィルターのテスト
"""

import logging
import time
import pytest
from io import StringIO
from app.core.utils.DuplicateFilterHandler import DuplicateFilterHandler


class TestDuplicateFilterHandler:
    """DuplicateFilterHandlerのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.filter = DuplicateFilterHandler(capacity=10, interval=1.0)
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        
        # 既存のハンドラーをクリア
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # StringIOハンドラーを作成してテスト用に使用
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.addFilter(self.filter)
        self.logger.addHandler(self.handler)

    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def test_first_message_passes(self):
        """最初のメッセージは通ることを確認"""
        self.logger.info("Test message")
        output = self.log_stream.getvalue()
        assert "Test message" in output

    def test_duplicate_message_filtered(self):
        """重複メッセージがフィルタリングされることを確認"""
        self.logger.info("Duplicate message")
        self.logger.info("Duplicate message")
        
        output = self.log_stream.getvalue()
        # メッセージが1回だけ出力されることを確認
        assert output.count("Duplicate message") == 1

    def test_different_messages_pass(self):
        """異なるメッセージは通ることを確認"""
        self.logger.info("Message 1")
        self.logger.info("Message 2")
        
        output = self.log_stream.getvalue()
        assert "Message 1" in output
        assert "Message 2" in output

    def test_message_passes_after_interval(self):
        """一定時間経過後は同じメッセージが通ることを確認"""
        # 短いインターバルでテスト
        short_filter = DuplicateFilterHandler(capacity=10, interval=0.1)
        self.handler.removeFilter(self.filter)
        self.handler.addFilter(short_filter)
        
        self.logger.info("Interval test message")
        time.sleep(0.2)  # インターバルより長く待機
        self.logger.info("Interval test message")
        
        output = self.log_stream.getvalue()
        # メッセージが2回出力されることを確認
        assert output.count("Interval test message") == 2

    def test_capacity_limit(self):
        """キャパシティ制限が機能することを確認"""
        small_filter = DuplicateFilterHandler(capacity=2, interval=10.0)
        self.handler.removeFilter(self.filter)
        self.handler.addFilter(small_filter)
        
        # キャパシティを超えるメッセージを送信
        self.logger.info("Message 1")
        self.logger.info("Message 2")
        self.logger.info("Message 3")  # これでキャパシティを超える
        
        # 最初のメッセージが削除されているはずなので、再度送信すると通る
        self.logger.info("Message 1")
        
        output = self.log_stream.getvalue()
        assert output.count("Message 1") == 2

    def test_get_stats(self):
        """統計情報の取得をテスト"""
        self.logger.info("Stats test message 1")
        self.logger.info("Stats test message 2")
        
        stats = self.filter.get_stats()
        assert stats["tracked_messages"] == 2
        assert stats["capacity"] == 10

    def test_clear_cache(self):
        """キャッシュクリア機能をテスト"""
        self.logger.info("Cache test message")
        self.logger.info("Cache test message")  # 重複でフィルタリング
        
        # キャッシュをクリア
        self.filter.clear_cache()
        
        # 同じメッセージが再度通るはず
        self.logger.info("Cache test message")
        
        output = self.log_stream.getvalue()
        assert output.count("Cache test message") == 2

    def test_exception_handling(self):
        """例外処理のテスト"""
        # 不正なレコードでも例外が発生しないことを確認
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=None,  # 不正なメッセージ
            args=(),
            exc_info=None
        )
        
        # 例外が発生せずにTrueが返されることを確認
        result = self.filter.filter(record)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
