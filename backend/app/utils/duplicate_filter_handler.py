import logging
import time
from collections import OrderedDict


class DuplicateFilter(logging.Filter):
    """
    一定期間内の重複ログメッセージをフィルタリングする。
    """

    def __init__(self, interval: float = 1.0, capacity: int = 100):
        """
        Args:
            interval (float): 同じメッセージが再度許可されるまでの最小時間（秒）。
            capacity (int): 記憶する最新のログメッセージの最大数。
        """
        super().__init__()
        self.interval = interval
        self.capacity = capacity
        self.last_logs: "OrderedDict[str, float]" = OrderedDict()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        ログレコードをフィルタリングする。

        Args:
            record: ログレコード

        Returns:
            bool: ログを出力する場合はTrue、スキップする場合はFalse
        """
        try:
            message = record.getMessage()
        except Exception:
            return True

        current_time = time.time()

        if message in self.last_logs:
            last_time = self.last_logs[message]
            if current_time - last_time < self.interval:
                return False  # 抑制
            # Move to end to mark as most recently used
            self.last_logs.move_to_end(message)

        self.last_logs[message] = current_time

        # LRUキャッシュのサイズを制限
        if len(self.last_logs) > self.capacity:
            self.last_logs.popitem(last=False)

        return True
