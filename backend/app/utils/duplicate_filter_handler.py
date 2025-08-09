import logging
import time


class DuplicateFilter(logging.Filter):
    """
    一定期間内の重複ログメッセージをフィルタリングする。
    標準のlogging.Filterを使い、よりシンプルな実装にします。
    """

    def __init__(self, name: str = "", interval: float = 1.0):
        """
        Args:
            name (str): フィルター名。
            interval (float): 同じメッセージが再度許可されるまでの最小時間（秒）。
        """
        super().__init__(name)
        self.interval = interval
        self._last_log_time_by_msg: dict[str, float] = {}

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

        self._last_log_time_by_msg[message] = current_time
        return True