import logging
import time
from typing import Dict, Tuple


class DuplicateFilterHandler(logging.Filter):
    """
    重複するログメッセージをフィルタリングするカスタムログフィルター。
    同じメッセージが一定期間内または連続して出力されるのを防ぎます。
    """

    def __init__(self, capacity: int = 100, interval: float = 1.0):
        """
        コンストラクタ。

        Args:
            capacity (int): 記憶する最新のログメッセージの最大数。
            interval (float): 同じメッセージが再度許可されるまでの最小時間（秒）。
        """
        super().__init__()
        self.capacity = capacity
        self.interval = interval
        self.last_logs: Dict[str, Tuple[float, int]] = (
            {}
        )  # {message: (timestamp, count)}

    def filter(self, record: logging.LogRecord) -> bool:
        """
        ログレコードをフィルタリングします。
        Trueを返すとログが出力され、Falseを返すとスキップされます。

        Args:
            record: ログレコード

        Returns:
            bool: ログを出力する場合True、スキップする場合False
        """
        # ログメッセージを取得
        try:
            message = record.getMessage()
        except Exception:
            # メッセージの取得に失敗した場合は通す
            return True

        current_time = time.time()

        # メッセージの重複チェック
        if message in self.last_logs:
            last_time, count = self.last_logs[message]
            if current_time - last_time < self.interval:
                # 一定期間内の重複メッセージはスキップ
                self.last_logs[message] = (current_time, count + 1)
                return False
            else:
                # 一定期間経過したらカウントをリセットして許可
                self.last_logs[message] = (current_time, 1)
                return True
        else:
            # 新しいメッセージなので許可
            self.last_logs[message] = (current_time, 1)

        # 記憶するログの数を制限（LRU的な動作）
        if len(self.last_logs) > self.capacity:
            # 最も古いエントリを削除
            oldest_message = min(self.last_logs, key=lambda k: self.last_logs[k][0])
            del self.last_logs[oldest_message]

        return True

    def get_stats(self) -> Dict[str, int]:
        """
        フィルターの統計情報を取得

        Returns:
            Dict[str, int]: 統計情報（記録中のメッセージ数など）
        """
        return {
            "tracked_messages": len(self.last_logs),
            "capacity": self.capacity,
        }

    def clear_cache(self) -> None:
        """
        キャッシュをクリア
        """
        self.last_logs.clear()
