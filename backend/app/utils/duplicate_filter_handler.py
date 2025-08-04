import logging
import time
from typing import Dict, Tuple, Optional, Any


class DuplicateFilterHandler(logging.Filter):
    """
    重複するログメッセージをフィルタリングするカスタムログフィルター。
    同じメッセージが一定期間内または連続して出力されるのを防ぎます。

    拡張点:
      - サマリーログ（抑制統計）のオプトイン出力
      - 容量超過時（LRU）削除のデバッグ出力（オプトイン）
    """

    def __init__(
        self,
        capacity: int = 100,
        interval: float = 1.0,
        *,
        enable_summary: bool = False,
        summary_threshold: int = 10,
        summary_interval: float = 30.0,
        enable_eviction_debug: bool = False,
        logger_name: Optional[str] = None,
    ):
        """
        コンストラクタ。

        Args:
            capacity (int): 記憶する最新のログメッセージの最大数。
            interval (float): 同じメッセージが再度許可されるまでの最小時間（秒）。
            enable_summary (bool): サマリー出力を有効化。
            summary_threshold (int): interval 内抑制回数のしきい値（これを超えるとサマリー候補）。
            summary_interval (float): サマリーログを出す最小間隔（秒）。
            enable_eviction_debug (bool): 容量超過時の LRU 削除を debug 出力。
            logger_name (Optional[str]): 出力に使用するロガー名（未指定は __name__）。
        """
        super().__init__()
        self.capacity = capacity
        self.interval = interval

        # 重複判定用（最後の許可時刻、interval 内カウント）
        self.last_logs: Dict[str, Tuple[float, int]] = {}

        # サマリー関連
        self.enable_summary = enable_summary
        self.summary_threshold = summary_threshold
        self.summary_interval = summary_interval
        # message -> {"first_suppress_ts": float, "last_summary_ts": float, "suppressed_count_accum": int}
        self._summary_stats: Dict[str, Dict[str, Any]] = {}

        # LRU 削除 debug
        self.enable_eviction_debug = enable_eviction_debug

        # 出力ロガー
        self._logger = logging.getLogger(logger_name or __name__)

    def _maybe_log_summary(self, message: str, now: float) -> None:
        """サマリー出力条件を満たす場合にログを出力"""
        if not self.enable_summary:
            return

        stats = self._summary_stats.setdefault(
            message,
            {
                "first_suppress_ts": now,
                "last_summary_ts": 0.0,
                "suppressed_count_accum": 0,
            },
        )

        # 間隔チェック
        if (
            stats["suppressed_count_accum"] >= self.summary_threshold
            and (now - stats["last_summary_ts"]) >= self.summary_interval
        ):
            elapsed = now - stats["first_suppress_ts"]
            count = stats["suppressed_count_accum"]
            # 情報レベルでサマリー出力
            self._logger.info(
                "message=%r suppressed %d times in last %.1f sec",
                message,
                count,
                elapsed,
            )
            # 状態更新（次のサマリーまでの間隔管理）
            stats["last_summary_ts"] = now
            stats["first_suppress_ts"] = now
            stats["suppressed_count_accum"] = 0

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
                # 一定期間内の重複メッセージはスキップ（抑制）
                self.last_logs[message] = (current_time, count + 1)

                # サマリー統計の更新
                if self.enable_summary:
                    stats = self._summary_stats.setdefault(
                        message,
                        {
                            "first_suppress_ts": current_time,
                            "last_summary_ts": 0.0,
                            "suppressed_count_accum": 0,
                        },
                    )
                    # 抑制分を累積
                    stats["suppressed_count_accum"] += 1
                    # first_suppress_ts は最初の抑制時刻を保つ（初期セット済み）
                    self._maybe_log_summary(message, current_time)

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

            if self.enable_eviction_debug:
                # 抑制統計があれば付加情報も出す
                stats = self._summary_stats.get(
                    oldest_message,
                    {"suppressed_count_accum": 0, "last_summary_ts": 0.0},
                )
                self._logger.debug(
                    "evicted message key=%r due to capacity LRU (capacity=%d, size=%d, suppressed_accum=%d)",
                    oldest_message,
                    self.capacity,
                    len(self.last_logs),
                    int(stats.get("suppressed_count_accum", 0)),
                )

            # 実際の削除（last_logs と summary も掃除）
            del self.last_logs[oldest_message]
            self._summary_stats.pop(oldest_message, None)

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
        self._summary_stats.clear()
