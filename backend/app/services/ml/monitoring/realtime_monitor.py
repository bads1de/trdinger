"""
リアルタイム監視システム

システム状態、モデル性能、データ品質を継続的に監視し、
異常検出時に自動アラートを発行します。
"""

import logging
import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

from .data_drift_detector import DataDriftDetector, DriftType
from ..model_manager import ModelManager
from ....utils.unified_error_handler import safe_ml_operation

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: float


@dataclass
class ModelMetrics:
    """モデルメトリクス"""
    timestamp: datetime
    model_name: str
    prediction_count: int
    average_latency_ms: float
    error_rate: float
    accuracy: Optional[float] = None
    drift_score: Optional[float] = None


@dataclass
class Alert:
    """アラート"""
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class MonitoringConfig:
    """監視設定"""
    # システム監視
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 90.0
    disk_critical_threshold: float = 95.0
    
    # モデル監視
    latency_warning_threshold: float = 1000.0  # ms
    latency_critical_threshold: float = 5000.0  # ms
    error_rate_warning_threshold: float = 0.05
    error_rate_critical_threshold: float = 0.15
    accuracy_warning_threshold: float = 0.05  # 5%低下
    
    # 監視間隔
    system_check_interval: int = 30  # seconds
    model_check_interval: int = 60   # seconds
    drift_check_interval: int = 300  # seconds
    
    # データ保持
    max_metrics_history: int = 1000
    max_alerts_history: int = 500


class RealtimeMonitor:
    """
    リアルタイム監視システム
    
    システムリソース、モデル性能、データドリフトを
    継続的に監視し、異常時にアラートを発行します。
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        初期化
        
        Args:
            config: 監視設定
        """
        self.config = config or MonitoringConfig()
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # メトリクス履歴
        self.system_metrics_history: deque = deque(maxlen=self.config.max_metrics_history)
        self.model_metrics_history: deque = deque(maxlen=self.config.max_metrics_history)
        self.alerts_history: deque = deque(maxlen=self.config.max_alerts_history)
        
        # 外部サービス
        self.drift_detector = DataDriftDetector()
        self.model_manager = ModelManager()
        
        # アラートコールバック
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # パフォーマンス追跡
        self.prediction_counts: Dict[str, int] = {}
        self.latency_measurements: Dict[str, deque] = {}
        self.error_counts: Dict[str, int] = {}
        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """アラートコールバックを追加"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """監視を開始"""
        if self.is_running:
            logger.warning("監視は既に実行中です")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("リアルタイム監視を開始しました")
    
    def stop_monitoring(self):
        """監視を停止"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("リアルタイム監視を停止しました")
    
    def _monitoring_loop(self):
        """監視ループ"""
        last_system_check = 0
        last_model_check = 0
        last_drift_check = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # システム監視
                if current_time - last_system_check >= self.config.system_check_interval:
                    self._check_system_metrics()
                    last_system_check = current_time
                
                # モデル監視
                if current_time - last_model_check >= self.config.model_check_interval:
                    self._check_model_metrics()
                    last_model_check = current_time
                
                # ドリフト監視
                if current_time - last_drift_check >= self.config.drift_check_interval:
                    self._check_data_drift()
                    last_drift_check = current_time
                
                time.sleep(1)  # 1秒間隔でチェック
                
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(5)  # エラー時は5秒待機
    
    def _check_system_metrics(self):
        """システムメトリクスをチェック"""
        try:
            # システム情報を取得
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())
            
            # 負荷平均（Windowsでは利用不可の場合がある）
            try:
                load_average = psutil.getloadavg()[0]
            except AttributeError:
                load_average = cpu_percent / 100.0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                process_count=process_count,
                load_average=load_average
            )
            
            self.system_metrics_history.append(metrics)
            
            # アラートチェック
            self._check_system_alerts(metrics)
            
        except Exception as e:
            logger.error(f"システムメトリクス取得エラー: {e}")
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """システムアラートをチェック"""
        # CPU使用率チェック
        if metrics.cpu_percent >= self.config.cpu_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL, "system", 
                f"CPU使用率が危険レベルに達しました: {metrics.cpu_percent:.1f}%",
                {"cpu_percent": metrics.cpu_percent, "threshold": self.config.cpu_critical_threshold}
            )
        elif metrics.cpu_percent >= self.config.cpu_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING, "system",
                f"CPU使用率が高くなっています: {metrics.cpu_percent:.1f}%",
                {"cpu_percent": metrics.cpu_percent, "threshold": self.config.cpu_warning_threshold}
            )
        
        # メモリ使用率チェック
        if metrics.memory_percent >= self.config.memory_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL, "system",
                f"メモリ使用率が危険レベルに達しました: {metrics.memory_percent:.1f}%",
                {"memory_percent": metrics.memory_percent, "threshold": self.config.memory_critical_threshold}
            )
        elif metrics.memory_percent >= self.config.memory_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING, "system",
                f"メモリ使用率が高くなっています: {metrics.memory_percent:.1f}%",
                {"memory_percent": metrics.memory_percent, "threshold": self.config.memory_warning_threshold}
            )
        
        # ディスク使用率チェック
        if metrics.disk_percent >= self.config.disk_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL, "system",
                f"ディスク使用率が危険レベルに達しました: {metrics.disk_percent:.1f}%",
                {"disk_percent": metrics.disk_percent, "threshold": self.config.disk_critical_threshold}
            )
        elif metrics.disk_percent >= self.config.disk_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING, "system",
                f"ディスク使用率が高くなっています: {metrics.disk_percent:.1f}%",
                {"disk_percent": metrics.disk_percent, "threshold": self.config.disk_warning_threshold}
            )
    
    def _check_model_metrics(self):
        """モデルメトリクスをチェック"""
        try:
            # 各モデルのメトリクスを確認
            for model_name in self.prediction_counts.keys():
                # レイテンシ計算
                latencies = self.latency_measurements.get(model_name, deque())
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                
                # エラー率計算
                total_predictions = self.prediction_counts.get(model_name, 0)
                error_count = self.error_counts.get(model_name, 0)
                error_rate = error_count / total_predictions if total_predictions > 0 else 0
                
                metrics = ModelMetrics(
                    timestamp=datetime.now(),
                    model_name=model_name,
                    prediction_count=total_predictions,
                    average_latency_ms=avg_latency,
                    error_rate=error_rate
                )
                
                self.model_metrics_history.append(metrics)
                
                # アラートチェック
                self._check_model_alerts(metrics)
                
        except Exception as e:
            logger.error(f"モデルメトリクス取得エラー: {e}")
    
    def _check_model_alerts(self, metrics: ModelMetrics):
        """モデルアラートをチェック"""
        # レイテンシチェック
        if metrics.average_latency_ms >= self.config.latency_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL, "model",
                f"モデル {metrics.model_name} のレイテンシが危険レベルです: {metrics.average_latency_ms:.1f}ms",
                {"model_name": metrics.model_name, "latency_ms": metrics.average_latency_ms}
            )
        elif metrics.average_latency_ms >= self.config.latency_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING, "model",
                f"モデル {metrics.model_name} のレイテンシが高くなっています: {metrics.average_latency_ms:.1f}ms",
                {"model_name": metrics.model_name, "latency_ms": metrics.average_latency_ms}
            )
        
        # エラー率チェック
        if metrics.error_rate >= self.config.error_rate_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL, "model",
                f"モデル {metrics.model_name} のエラー率が危険レベルです: {metrics.error_rate:.1%}",
                {"model_name": metrics.model_name, "error_rate": metrics.error_rate}
            )
        elif metrics.error_rate >= self.config.error_rate_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING, "model",
                f"モデル {metrics.model_name} のエラー率が高くなっています: {metrics.error_rate:.1%}",
                {"model_name": metrics.model_name, "error_rate": metrics.error_rate}
            )
    
    def _check_data_drift(self):
        """データドリフトをチェック"""
        try:
            # ドリフト検出サマリーを取得
            drift_summary = self.drift_detector.get_drift_summary(hours=1)
            
            if drift_summary.get("requires_attention", False):
                severe_features = drift_summary.get("severe_drift_features", [])
                self._create_alert(
                    AlertLevel.ERROR, "data_drift",
                    f"深刻なデータドリフトが検出されました: {', '.join(severe_features)}",
                    {"drift_summary": drift_summary}
                )
                
        except Exception as e:
            logger.error(f"データドリフトチェックエラー: {e}")
    
    def _create_alert(self, level: AlertLevel, category: str, message: str, details: Dict[str, Any]):
        """アラートを作成"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.alerts_history.append(alert)
        
        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"アラートコールバックエラー: {e}")
        
        # ログ出力
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(level, logger.info)
        
        log_level(f"[{category.upper()}] {message}")
    
    def record_prediction(self, model_name: str, latency_ms: float, success: bool = True):
        """予測実行を記録"""
        # 予測回数
        self.prediction_counts[model_name] = self.prediction_counts.get(model_name, 0) + 1
        
        # レイテンシ
        if model_name not in self.latency_measurements:
            self.latency_measurements[model_name] = deque(maxlen=100)
        self.latency_measurements[model_name].append(latency_ms)
        
        # エラー回数
        if not success:
            self.error_counts[model_name] = self.error_counts.get(model_name, 0) + 1
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態を取得"""
        recent_alerts = [
            alert for alert in self.alerts_history 
            if alert.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "is_running": self.is_running,
            "system_metrics_count": len(self.system_metrics_history),
            "model_metrics_count": len(self.model_metrics_history),
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts_count": sum(1 for a in recent_alerts if a.level == AlertLevel.CRITICAL),
            "last_system_check": self.system_metrics_history[-1].timestamp.isoformat() if self.system_metrics_history else None,
            "monitored_models": list(self.prediction_counts.keys())
        }
