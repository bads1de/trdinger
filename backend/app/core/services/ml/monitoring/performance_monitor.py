"""
パフォーマンス監視システム

戦略の実運用パフォーマンスを監視し、劣化検出アラートを提供します。
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """アラートレベル"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""

    strategy_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    strategy_id: str
    timestamp: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    long_trades: int
    short_trades: int
    long_pnl: float
    short_pnl: float
    balance_score: float


class PerformanceMonitor:
    """
    パフォーマンス監視クラス

    戦略の実運用パフォーマンスを継続的に監視し、
    劣化検出やアラート機能を提供します。
    """

    def __init__(self):
        """初期化"""
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.alerts: List[PerformanceAlert] = []
        self.monitoring_config = {
            "min_sharpe_ratio": 1.0,
            "max_drawdown_threshold": 0.2,
            "min_win_rate": 0.45,
            "min_balance_score": 0.3,
            "performance_window": 30,  # 日数
            "alert_cooldown": 3600,  # 秒
        }
        self.last_alert_times: Dict[str, datetime] = {}

    def add_performance_record(
        self, strategy_id: str, performance_data: Dict[str, Any]
    ) -> None:
        """
        パフォーマンス記録を追加

        Args:
            strategy_id: 戦略ID
            performance_data: パフォーマンスデータ
        """
        try:
            # パフォーマンス指標を作成
            metrics = PerformanceMetrics(
                strategy_id=strategy_id,
                timestamp=datetime.now(),
                total_return=performance_data.get("total_return", 0.0),
                sharpe_ratio=performance_data.get("sharpe_ratio", 0.0),
                max_drawdown=performance_data.get("max_drawdown", 0.0),
                win_rate=performance_data.get("win_rate", 0.0),
                total_trades=performance_data.get("total_trades", 0),
                long_trades=performance_data.get("long_trades", 0),
                short_trades=performance_data.get("short_trades", 0),
                long_pnl=performance_data.get("long_pnl", 0.0),
                short_pnl=performance_data.get("short_pnl", 0.0),
                balance_score=performance_data.get("balance_score", 0.0),
            )

            # 履歴に追加
            if strategy_id not in self.performance_history:
                self.performance_history[strategy_id] = []

            self.performance_history[strategy_id].append(metrics)

            # 古い記録を削除（メモリ管理）
            self._cleanup_old_records(strategy_id)

            # パフォーマンス劣化をチェック
            self._check_performance_degradation(strategy_id)

            logger.debug(f"パフォーマンス記録を追加: {strategy_id}")

        except Exception as e:
            logger.error(f"パフォーマンス記録追加エラー: {e}")

    def get_strategy_performance(
        self, strategy_id: str, days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        戦略のパフォーマンス統計を取得

        Args:
            strategy_id: 戦略ID
            days: 取得期間（日数）

        Returns:
            パフォーマンス統計
        """
        try:
            if strategy_id not in self.performance_history:
                return None

            # 指定期間のデータを取得
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_records = [
                record
                for record in self.performance_history[strategy_id]
                if record.timestamp >= cutoff_date
            ]

            if not recent_records:
                return None

            # 統計を計算
            returns = [r.total_return for r in recent_records]
            sharpe_ratios = [r.sharpe_ratio for r in recent_records]
            drawdowns = [r.max_drawdown for r in recent_records]
            win_rates = [r.win_rate for r in recent_records]
            balance_scores = [r.balance_score for r in recent_records]

            return {
                "strategy_id": strategy_id,
                "period_days": days,
                "record_count": len(recent_records),
                "avg_return": np.mean(returns),
                "avg_sharpe_ratio": np.mean(sharpe_ratios),
                "avg_drawdown": np.mean(drawdowns),
                "avg_win_rate": np.mean(win_rates),
                "avg_balance_score": np.mean(balance_scores),
                "return_volatility": np.std(returns),
                "latest_record": recent_records[-1],
                "performance_trend": self._calculate_trend(recent_records),
            }

        except Exception as e:
            logger.error(f"戦略パフォーマンス取得エラー: {e}")
            return None

    def detect_performance_degradation(self, strategy_id: str) -> bool:
        """
        パフォーマンス劣化を検出

        Args:
            strategy_id: 戦略ID

        Returns:
            劣化検出フラグ
        """
        try:
            if strategy_id not in self.performance_history:
                return False

            recent_performance = self.get_strategy_performance(strategy_id, days=7)
            if not recent_performance:
                return False

            # 劣化条件をチェック
            degradation_flags = []

            # シャープレシオの劣化
            if (
                recent_performance["avg_sharpe_ratio"]
                < self.monitoring_config["min_sharpe_ratio"]
            ):
                degradation_flags.append("low_sharpe_ratio")

            # ドローダウンの悪化
            if (
                recent_performance["avg_drawdown"]
                > self.monitoring_config["max_drawdown_threshold"]
            ):
                degradation_flags.append("high_drawdown")

            # 勝率の低下
            if (
                recent_performance["avg_win_rate"]
                < self.monitoring_config["min_win_rate"]
            ):
                degradation_flags.append("low_win_rate")

            # バランススコアの悪化
            if (
                recent_performance["avg_balance_score"]
                < self.monitoring_config["min_balance_score"]
            ):
                degradation_flags.append("poor_balance")

            # トレンドの悪化
            if recent_performance["performance_trend"] < -0.1:
                degradation_flags.append("negative_trend")

            return len(degradation_flags) > 0

        except Exception as e:
            logger.error(f"パフォーマンス劣化検出エラー: {e}")
            return False

    def get_alerts(
        self, strategy_id: Optional[str] = None, level: Optional[AlertLevel] = None
    ) -> List[PerformanceAlert]:
        """
        アラートを取得

        Args:
            strategy_id: 戦略ID（オプション）
            level: アラートレベル（オプション）

        Returns:
            アラートリスト
        """
        alerts = self.alerts

        if strategy_id:
            alerts = [alert for alert in alerts if alert.strategy_id == strategy_id]

        if level:
            alerts = [alert for alert in alerts if alert.level == level]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def clear_alerts(self, strategy_id: Optional[str] = None) -> None:
        """
        アラートをクリア

        Args:
            strategy_id: 戦略ID（オプション、指定時はその戦略のみクリア）
        """
        if strategy_id:
            self.alerts = [
                alert for alert in self.alerts if alert.strategy_id != strategy_id
            ]
        else:
            self.alerts.clear()

        logger.info(f"アラートをクリア: {strategy_id or 'all'}")

    def _check_performance_degradation(self, strategy_id: str) -> None:
        """パフォーマンス劣化をチェックしてアラートを生成"""
        try:
            # アラートクールダウンチェック
            if self._is_alert_cooldown(strategy_id):
                return

            if self.detect_performance_degradation(strategy_id):
                recent_performance = self.get_strategy_performance(strategy_id, days=7)

                # アラートメトリクスを定義
                alert_metrics: Dict[str, Any]
                if recent_performance is not None:
                    alert_metrics = recent_performance
                else:
                    alert_metrics = {}

                alert = PerformanceAlert(
                    strategy_id=strategy_id,
                    alert_type="performance_degradation",
                    level=AlertLevel.WARNING,
                    message=f"戦略 {strategy_id} のパフォーマンス劣化を検出",
                    timestamp=datetime.now(),
                    metrics=alert_metrics,
                )

                self.alerts.append(alert)
                self.last_alert_times[strategy_id] = datetime.now()

                logger.warning(f"パフォーマンス劣化アラート: {strategy_id}")

        except Exception as e:
            logger.error(f"パフォーマンス劣化チェックエラー: {e}")

    def _cleanup_old_records(self, strategy_id: str) -> None:
        """古い記録を削除"""
        try:
            if strategy_id not in self.performance_history:
                return

            # 90日より古い記録を削除
            cutoff_date = datetime.now() - timedelta(days=90)
            self.performance_history[strategy_id] = [
                record
                for record in self.performance_history[strategy_id]
                if record.timestamp >= cutoff_date
            ]

        except Exception as e:
            logger.error(f"古い記録削除エラー: {e}")

    def _calculate_trend(self, records: List[PerformanceMetrics]) -> float:
        """パフォーマンストレンドを計算"""
        try:
            if len(records) < 2:
                return 0.0

            # 直近のリターンの傾向を計算
            returns = [r.total_return for r in records[-10:]]  # 最新10件

            if len(returns) < 2:
                return 0.0

            # 線形回帰の傾き
            x = np.arange(len(returns))
            slope = np.polyfit(x, returns, 1)[0]

            return float(slope)

        except Exception as e:
            logger.error(f"トレンド計算エラー: {e}")
            return 0.0

    def _is_alert_cooldown(self, strategy_id: str) -> bool:
        """アラートクールダウン中かチェック"""
        if strategy_id not in self.last_alert_times:
            return False

        cooldown_period = timedelta(seconds=self.monitoring_config["alert_cooldown"])
        return datetime.now() - self.last_alert_times[strategy_id] < cooldown_period
