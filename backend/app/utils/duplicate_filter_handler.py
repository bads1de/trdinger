import logging
import time
from collections import OrderedDict


class DuplicateFilter(logging.Filter):
    """
    一定期間内の重複ログメッセージをフィルタリングする。
    標準のlogging.Filterを使い、よりシンプルな実装にします。
    """

    def __init__(self, name: str = "", interval: float = 1.0, capacity: int = 200):
        """
        Args:
            name (str): フィルター名。
            interval (float): 同じメッセージが再度許可されるまでの最小時間（秒）。
            capacity (int): 記憶するメッセージの最大数。
        """
        super().__init__(name)
        self.interval = interval
        self.capacity = capacity
        self._last_log_time_by_msg: OrderedDict[str, float] = OrderedDict()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        ログレコードをフィルタリングする。

        Args:
            record: ログレコード

        Returns:
            bool: ログを出力する場合はTrue、スキップする場合はFalse
        """
        message = record.getMessage()
        current_time = time.time()

        last_time = self._last_log_time_by_msg.get(message)

        if last_time and (current_time - last_time) < self.interval:
            return False  # 期間内なので抑制

        # メッセージを記録（LRU方式で容量制限）
        if message in self._last_log_time_by_msg:
            # 既存のメッセージを最新に移動
            self._last_log_time_by_msg.move_to_end(message)
        else:
            # 容量制限チェック
            if len(self._last_log_time_by_msg) >= self.capacity:
                # 最も古いメッセージを削除
                self._last_log_time_by_msg.popitem(last=False)

        self._last_log_time_by_msg[message] = current_time
        return True



