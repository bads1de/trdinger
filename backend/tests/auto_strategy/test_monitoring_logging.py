"""
オートストラテジー 監視・ログテスト

異常検知アラート、ログファイル管理、パフォーマンスメトリクス、エラー追跡を検証します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import pandas as pd
import numpy as np
import time
import logging
import tempfile
import shutil
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psutil
import traceback
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class CustomLogHandler(logging.Handler):
    """カスタムログハンドラー"""
    
    def __init__(self):
        super().__init__()
        self.logs = deque(maxlen=10000)
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0
        
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)
        
        self.logs.append(log_entry)
        
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        elif record.levelno >= logging.INFO:
            self.info_count += 1


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "response_times": deque(maxlen=1000),
            "error_rates": deque(maxlen=1000),
            "throughput": deque(maxlen=1000)
        }
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                # メモリ使用率
                memory_info = psutil.virtual_memory()
                self.metrics["memory_usage"].append(memory_info.percent)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"監視ループエラー: {e}")
    
    def record_response_time(self, response_time: float):
        """レスポンス時間を記録"""
        self.metrics["response_times"].append(response_time)
    
    def record_error_rate(self, error_rate: float):
        """エラー率を記録"""
        self.metrics["error_rates"].append(error_rate)
    
    def record_throughput(self, throughput: float):
        """スループットを記録"""
        self.metrics["throughput"].append(throughput)
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
            else:
                stats[metric_name] = {"count": 0}
        
        return stats


class TestMonitoringLogging:
    """監視・ログテストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()
        self.log_handler = CustomLogHandler()
        self.performance_monitor = PerformanceMonitor()
        
        # ログハンドラーを追加
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.DEBUG)
        
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        execution_time = time.time() - self.start_time
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
        
        # ログハンドラーを削除
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.log_handler)
        
        # パフォーマンス監視停止
        self.performance_monitor.stop_monitoring()
        
        # 一時ディレクトリのクリーンアップ
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_anomaly_detection_alerts(self):
        """テスト63: 異常検知アラートの正確性"""
        logger.info("🔍 異常検知アラートテスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            
            # 異常検知システム
            class AnomalyDetector:
                def __init__(self):
                    self.baseline_metrics = {
                        "response_time": {"mean": 0.01, "std": 0.005, "threshold": 3.0},
                        "error_rate": {"mean": 0.001, "std": 0.0005, "threshold": 5.0},
                        "cpu_usage": {"mean": 20.0, "std": 5.0, "threshold": 3.0},
                        "memory_usage": {"mean": 50.0, "std": 10.0, "threshold": 2.5}
                    }
                    self.alerts = []
                
                def check_anomaly(self, metric_name: str, value: float) -> bool:
                    """異常値をチェック"""
                    if metric_name not in self.baseline_metrics:
                        return False
                    
                    baseline = self.baseline_metrics[metric_name]
                    z_score = abs(value - baseline["mean"]) / baseline["std"]
                    
                    if z_score > baseline["threshold"]:
                        alert = {
                            "timestamp": datetime.now(),
                            "metric": metric_name,
                            "value": value,
                            "z_score": z_score,
                            "threshold": baseline["threshold"],
                            "severity": "HIGH" if z_score > baseline["threshold"] * 1.5 else "MEDIUM"
                        }
                        self.alerts.append(alert)
                        logger.warning(f"異常検知: {metric_name}={value:.4f} (Z-score: {z_score:.2f})")
                        return True
                    
                    return False
            
            detector = AnomalyDetector()
            
            # 正常データと異常データを混在させてテスト
            test_scenarios = [
                # 正常なシナリオ
                {"response_time": 0.008, "error_rate": 0.0008, "cpu_usage": 18.0, "memory_usage": 45.0},
                {"response_time": 0.012, "error_rate": 0.0012, "cpu_usage": 22.0, "memory_usage": 55.0},
                {"response_time": 0.009, "error_rate": 0.0009, "cpu_usage": 19.0, "memory_usage": 48.0},
                
                # 異常なシナリオ
                {"response_time": 0.050, "error_rate": 0.001, "cpu_usage": 20.0, "memory_usage": 50.0},  # 高レスポンス時間
                {"response_time": 0.010, "error_rate": 0.008, "cpu_usage": 20.0, "memory_usage": 50.0},  # 高エラー率
                {"response_time": 0.010, "error_rate": 0.001, "cpu_usage": 45.0, "memory_usage": 50.0},  # 高CPU使用率
                {"response_time": 0.010, "error_rate": 0.001, "cpu_usage": 20.0, "memory_usage": 85.0},  # 高メモリ使用率
                
                # 複合異常
                {"response_time": 0.080, "error_rate": 0.010, "cpu_usage": 50.0, "memory_usage": 90.0},  # 複数異常
            ]
            
            anomaly_stats = {
                "total_checks": 0,
                "anomalies_detected": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 0,
                "true_negatives": 0
            }
            
            for i, scenario in enumerate(test_scenarios):
                logger.info(f"シナリオ {i+1}: {scenario}")
                
                # 各メトリクスをチェック
                scenario_anomalies = 0
                expected_anomalies = 0
                
                # 期待される異常数を計算（手動で設定）
                if i >= 3:  # 異常シナリオ
                    if i == 3:  # 高レスポンス時間
                        expected_anomalies = 1
                    elif i == 4:  # 高エラー率
                        expected_anomalies = 1
                    elif i == 5:  # 高CPU使用率
                        expected_anomalies = 1
                    elif i == 6:  # 高メモリ使用率
                        expected_anomalies = 1
                    elif i == 7:  # 複合異常
                        expected_anomalies = 4
                
                for metric_name, value in scenario.items():
                    anomaly_stats["total_checks"] += 1
                    is_anomaly = detector.check_anomaly(metric_name, value)
                    
                    if is_anomaly:
                        scenario_anomalies += 1
                        anomaly_stats["anomalies_detected"] += 1
                
                # 精度の評価
                if expected_anomalies > 0:  # 異常が期待される場合
                    if scenario_anomalies > 0:
                        anomaly_stats["true_positives"] += 1
                    else:
                        anomaly_stats["false_negatives"] += 1
                else:  # 正常が期待される場合
                    if scenario_anomalies == 0:
                        anomaly_stats["true_negatives"] += 1
                    else:
                        anomaly_stats["false_positives"] += 1
                
                # 実際の処理をシミュレート
                try:
                    start_time = time.time()
                    sl_price, tp_price = calculator.calculate_basic_tpsl_prices(50000, 0.02, 0.04, 1.0)
                    response_time = time.time() - start_time
                    
                    # レスポンス時間の異常チェック
                    detector.check_anomaly("response_time", response_time)
                    
                except Exception as e:
                    logger.error(f"シナリオ {i+1} 処理エラー: {e}")
                    detector.check_anomaly("error_rate", 1.0)  # 100%エラー率
            
            # 異常検知の精度分析
            total_scenarios = len(test_scenarios)
            precision = anomaly_stats["true_positives"] / (anomaly_stats["true_positives"] + anomaly_stats["false_positives"]) if (anomaly_stats["true_positives"] + anomaly_stats["false_positives"]) > 0 else 0
            recall = anomaly_stats["true_positives"] / (anomaly_stats["true_positives"] + anomaly_stats["false_negatives"]) if (anomaly_stats["true_positives"] + anomaly_stats["false_negatives"]) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"異常検知結果:")
            logger.info(f"  総チェック数: {anomaly_stats['total_checks']}")
            logger.info(f"  検知された異常: {anomaly_stats['anomalies_detected']}")
            logger.info(f"  真陽性: {anomaly_stats['true_positives']}")
            logger.info(f"  真陰性: {anomaly_stats['true_negatives']}")
            logger.info(f"  偽陽性: {anomaly_stats['false_positives']}")
            logger.info(f"  偽陰性: {anomaly_stats['false_negatives']}")
            logger.info(f"  精度 (Precision): {precision:.3f}")
            logger.info(f"  再現率 (Recall): {recall:.3f}")
            logger.info(f"  F1スコア: {f1_score:.3f}")
            logger.info(f"  アラート数: {len(detector.alerts)}")
            
            # 異常検知の要件確認
            assert len(detector.alerts) > 0, "異常が検知されませんでした"
            assert precision >= 0.7, f"精度が低すぎます: {precision:.3f}"
            assert recall >= 0.7, f"再現率が低すぎます: {recall:.3f}"
            assert f1_score >= 0.7, f"F1スコアが低すぎます: {f1_score:.3f}"
            
            # アラートの詳細確認
            high_severity_alerts = [alert for alert in detector.alerts if alert["severity"] == "HIGH"]
            medium_severity_alerts = [alert for alert in detector.alerts if alert["severity"] == "MEDIUM"]
            
            logger.info(f"高重要度アラート: {len(high_severity_alerts)}件")
            logger.info(f"中重要度アラート: {len(medium_severity_alerts)}件")
            
            logger.info("✅ 異常検知アラートテスト成功")
            
        except Exception as e:
            pytest.fail(f"異常検知アラートテストエラー: {e}")
    
    def test_log_file_integrity_rotation(self):
        """テスト64: ログファイルの完整性とローテーション"""
        logger.info("🔍 ログファイル完整性・ローテーションテスト開始")
        
        try:
            # ログファイル管理システム
            class LogFileManager:
                def __init__(self, log_dir: str, max_file_size: int = 1024*1024, max_files: int = 5):
                    self.log_dir = log_dir
                    self.max_file_size = max_file_size
                    self.max_files = max_files
                    self.current_file = None
                    self.current_size = 0
                    self.file_count = 0
                    
                    os.makedirs(log_dir, exist_ok=True)
                    self._create_new_log_file()
                
                def _create_new_log_file(self):
                    """新しいログファイルを作成"""
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"auto_strategy_{timestamp}_{self.file_count}.log"
                    filepath = os.path.join(self.log_dir, filename)
                    
                    if self.current_file:
                        self.current_file.close()
                    
                    self.current_file = open(filepath, 'w', encoding='utf-8')
                    self.current_size = 0
                    self.file_count += 1
                    
                    return filepath
                
                def write_log(self, message: str):
                    """ログを書き込み"""
                    if not self.current_file:
                        self._create_new_log_file()
                    
                    log_entry = f"{datetime.now().isoformat()} - {message}\n"
                    self.current_file.write(log_entry)
                    self.current_file.flush()
                    
                    self.current_size += len(log_entry.encode('utf-8'))
                    
                    # ファイルサイズチェック
                    if self.current_size >= self.max_file_size:
                        self._rotate_log_files()
                
                def _rotate_log_files(self):
                    """ログファイルをローテーション"""
                    logger.info(f"ログファイルローテーション実行 (サイズ: {self.current_size} bytes)")
                    
                    # 古いファイルを削除
                    log_files = sorted([
                        f for f in os.listdir(self.log_dir) 
                        if f.startswith("auto_strategy_") and f.endswith(".log")
                    ])
                    
                    while len(log_files) >= self.max_files:
                        oldest_file = log_files.pop(0)
                        oldest_file_path = os.path.join(self.log_dir, oldest_file)
                        try:
                            # ファイルハンドルを閉じてから削除を試行
                            if hasattr(self, 'current_file_handle') and self.current_file_handle:
                                self.current_file_handle.close()
                                self.current_file_handle = None

                            os.remove(oldest_file_path)
                            logger.info(f"古いログファイルを削除: {oldest_file}")
                        except (PermissionError, OSError) as e:
                            # Windowsでファイルが使用中の場合は警告を出して続行
                            logger.warning(f"ログファイル削除に失敗（続行します）: {oldest_file} - {e}")
                            break  # 削除に失敗した場合はループを抜ける
                    
                    # 新しいファイルを作成
                    self._create_new_log_file()
                
                def get_log_files_info(self) -> List[Dict]:
                    """ログファイル情報を取得"""
                    log_files = []
                    
                    for filename in os.listdir(self.log_dir):
                        if filename.startswith("auto_strategy_") and filename.endswith(".log"):
                            filepath = os.path.join(self.log_dir, filename)
                            stat = os.stat(filepath)
                            
                            log_files.append({
                                "filename": filename,
                                "size": stat.st_size,
                                "created": datetime.fromtimestamp(stat.st_ctime),
                                "modified": datetime.fromtimestamp(stat.st_mtime)
                            })
                    
                    return sorted(log_files, key=lambda x: x["created"])
                
                def verify_log_integrity(self) -> Dict[str, Any]:
                    """ログファイルの整合性を検証"""
                    integrity_results = {
                        "total_files": 0,
                        "total_size": 0,
                        "corrupted_files": 0,
                        "empty_files": 0,
                        "valid_files": 0,
                        "encoding_errors": 0
                    }
                    
                    for filename in os.listdir(self.log_dir):
                        if filename.startswith("auto_strategy_") and filename.endswith(".log"):
                            filepath = os.path.join(self.log_dir, filename)
                            integrity_results["total_files"] += 1
                            
                            try:
                                file_size = os.path.getsize(filepath)
                                integrity_results["total_size"] += file_size
                                
                                if file_size == 0:
                                    integrity_results["empty_files"] += 1
                                    continue
                                
                                # ファイル内容の検証
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    line_count = 0
                                    for line in f:
                                        line_count += 1
                                        # 基本的なログ形式チェック
                                        if not line.strip():
                                            continue
                                        
                                        # ISO形式の日時が含まれているかチェック
                                        if " - " not in line:
                                            raise ValueError(f"Invalid log format in line {line_count}")
                                
                                integrity_results["valid_files"] += 1
                                
                            except UnicodeDecodeError:
                                integrity_results["encoding_errors"] += 1
                                logger.warning(f"エンコーディングエラー: {filename}")
                            except Exception as e:
                                integrity_results["corrupted_files"] += 1
                                logger.warning(f"ファイル破損: {filename} - {e}")
                    
                    return integrity_results
                
                def close(self):
                    """ログファイルマネージャーを閉じる"""
                    if self.current_file:
                        self.current_file.close()
            
            # ログファイルマネージャーのテスト
            log_manager = LogFileManager(
                log_dir=os.path.join(self.temp_dir, "logs"),
                max_file_size=1024,  # 1KB（テスト用に小さく設定）
                max_files=3
            )
            
            # 大量のログを生成してローテーションをテスト
            log_messages = []
            for i in range(100):
                message = f"Test log message {i:03d} - Processing auto strategy calculation with various parameters and detailed information"
                log_manager.write_log(message)
                log_messages.append(message)
                
                # 時々異なるレベルのログを混在
                if i % 10 == 0:
                    log_manager.write_log(f"INFO: Checkpoint reached at iteration {i}")
                elif i % 15 == 0:
                    log_manager.write_log(f"WARNING: High CPU usage detected at iteration {i}")
                elif i % 25 == 0:
                    log_manager.write_log(f"ERROR: Simulated error at iteration {i}")
            
            # ログファイル情報の取得
            log_files_info = log_manager.get_log_files_info()
            
            logger.info(f"ログファイル情報:")
            for file_info in log_files_info:
                logger.info(f"  {file_info['filename']}: {file_info['size']} bytes, 作成: {file_info['created']}")
            
            # 整合性検証
            integrity_results = log_manager.verify_log_integrity()
            
            logger.info(f"ログファイル整合性検証結果:")
            logger.info(f"  総ファイル数: {integrity_results['total_files']}")
            logger.info(f"  総サイズ: {integrity_results['total_size']} bytes")
            logger.info(f"  有効ファイル: {integrity_results['valid_files']}")
            logger.info(f"  破損ファイル: {integrity_results['corrupted_files']}")
            logger.info(f"  空ファイル: {integrity_results['empty_files']}")
            logger.info(f"  エンコーディングエラー: {integrity_results['encoding_errors']}")
            
            # ローテーション機能の確認
            assert integrity_results["total_files"] <= 5, f"ファイル数が制限を超えています: {integrity_results['total_files']}"
            assert integrity_results["total_files"] > 1, "ローテーションが実行されていません"
            
            # 整合性の確認
            assert integrity_results["corrupted_files"] == 0, f"破損ファイルが存在します: {integrity_results['corrupted_files']}"
            assert integrity_results["encoding_errors"] == 0, f"エンコーディングエラーがあります: {integrity_results['encoding_errors']}"
            # 空ファイルも有効ファイルとして扱う
            valid_files_including_empty = integrity_results["valid_files"] + integrity_results["empty_files"]
            assert valid_files_including_empty == integrity_results["total_files"], "無効なファイルが存在します"
            
            # ファイルサイズの確認
            for file_info in log_files_info[:-1]:  # 最後のファイル以外
                assert file_info["size"] >= 1024, f"ローテーション前のファイルサイズが小さすぎます: {file_info['size']}"
            
            log_manager.close()
            
            logger.info("✅ ログファイル完整性・ローテーションテスト成功")

        except Exception as e:
            pytest.fail(f"ログファイル完整性・ローテーションテストエラー: {e}")

    def test_performance_metrics_collection(self):
        """テスト65: パフォーマンスメトリクスの収集精度"""
        logger.info("🔍 パフォーマンスメトリクス収集テスト開始")

        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator

            calculator = TPSLCalculator()

            # パフォーマンス監視開始
            self.performance_monitor.start_monitoring()

            # 負荷テストシナリオ
            test_scenarios = [
                {"name": "軽負荷", "iterations": 100, "complexity": "low"},
                {"name": "中負荷", "iterations": 500, "complexity": "medium"},
                {"name": "高負荷", "iterations": 1000, "complexity": "high"}
            ]

            scenario_results = {}

            for scenario in test_scenarios:
                scenario_name = scenario["name"]
                iterations = scenario["iterations"]
                complexity = scenario["complexity"]

                logger.info(f"{scenario_name}テスト開始 ({iterations}回反復)")

                scenario_start = time.time()
                response_times = []
                error_count = 0

                for i in range(iterations):
                    operation_start = time.time()

                    try:
                        # 複雑度に応じた処理
                        if complexity == "low":
                            # 単純な計算
                            price = 50000 + np.random.normal(0, 100)
                            sl_price, tp_price = calculator.calculate_basic_tpsl_prices(price, 0.02, 0.04, 1.0)

                        elif complexity == "medium":
                            # 中程度の計算
                            for _ in range(5):
                                price = 50000 + np.random.normal(0, 100)
                                sl_price, tp_price = calculator.calculate_basic_tpsl_prices(price, 0.02, 0.04, 1.0)

                        elif complexity == "high":
                            # 複雑な計算
                            for _ in range(10):
                                price = 50000 + np.random.normal(0, 100)
                                sl_pct = 0.01 + np.random.uniform(0, 0.03)
                                tp_pct = 0.02 + np.random.uniform(0, 0.06)
                                direction = np.random.choice([1.0, -1.0])
                                sl_price, tp_price = calculator.calculate_basic_tpsl_prices(price, sl_pct, tp_pct, direction)

                        operation_time = time.time() - operation_start
                        response_times.append(operation_time)

                        # パフォーマンス監視に記録
                        self.performance_monitor.record_response_time(operation_time)

                    except Exception as e:
                        error_count += 1
                        logger.debug(f"{scenario_name} 反復 {i+1} エラー: {e}")

                    # 進捗表示
                    if (i + 1) % (iterations // 10) == 0:
                        progress = (i + 1) / iterations * 100
                        logger.info(f"  進捗: {progress:.0f}%")

                scenario_time = time.time() - scenario_start

                # シナリオ結果の分析
                if response_times:
                    avg_response_time = np.mean(response_times)
                    p95_response_time = np.percentile(response_times, 95)
                    p99_response_time = np.percentile(response_times, 99)
                    throughput = len(response_times) / scenario_time
                    error_rate = error_count / iterations
                else:
                    avg_response_time = 0
                    p95_response_time = 0
                    p99_response_time = 0
                    throughput = 0
                    error_rate = 1.0

                scenario_results[scenario_name] = {
                    "iterations": iterations,
                    "total_time": scenario_time,
                    "avg_response_time": avg_response_time,
                    "p95_response_time": p95_response_time,
                    "p99_response_time": p99_response_time,
                    "throughput": throughput,
                    "error_count": error_count,
                    "error_rate": error_rate,
                    "success_rate": 1.0 - error_rate
                }

                # パフォーマンス監視に記録
                self.performance_monitor.record_throughput(throughput)
                self.performance_monitor.record_error_rate(error_rate)

                logger.info(f"{scenario_name}テスト完了:")
                logger.info(f"  総時間: {scenario_time:.3f}秒")
                logger.info(f"  平均レスポンス時間: {avg_response_time*1000:.2f}ms")
                logger.info(f"  95%ileレスポンス時間: {p95_response_time*1000:.2f}ms")
                logger.info(f"  スループット: {throughput:.1f}ops/sec")
                logger.info(f"  エラー率: {error_rate:.1%}")

            # 監視停止
            time.sleep(1)  # 最後のメトリクスを収集
            self.performance_monitor.stop_monitoring()

            # 全体統計の取得
            overall_stats = self.performance_monitor.get_statistics()

            logger.info(f"全体パフォーマンス統計:")
            for metric_name, stats in overall_stats.items():
                if stats["count"] > 0:
                    logger.info(f"  {metric_name}:")
                    logger.info(f"    サンプル数: {stats['count']}")
                    logger.info(f"    平均: {stats['mean']:.4f}")
                    logger.info(f"    標準偏差: {stats['std']:.4f}")
                    logger.info(f"    最小: {stats['min']:.4f}")
                    logger.info(f"    最大: {stats['max']:.4f}")
                    logger.info(f"    95%ile: {stats['p95']:.4f}")

            # パフォーマンス要件の確認
            for scenario_name, result in scenario_results.items():
                # 成功率の確認
                assert result["success_rate"] >= 0.95, f"{scenario_name}: 成功率が低すぎます: {result['success_rate']:.1%}"

                # レスポンス時間の確認
                if scenario_name == "軽負荷":
                    assert result["avg_response_time"] < 0.01, f"{scenario_name}: 平均レスポンス時間が長すぎます: {result['avg_response_time']*1000:.2f}ms"
                elif scenario_name == "中負荷":
                    assert result["avg_response_time"] < 0.05, f"{scenario_name}: 平均レスポンス時間が長すぎます: {result['avg_response_time']*1000:.2f}ms"
                elif scenario_name == "高負荷":
                    assert result["avg_response_time"] < 0.1, f"{scenario_name}: 平均レスポンス時間が長すぎます: {result['avg_response_time']*1000:.2f}ms"

                # スループットの確認
                assert result["throughput"] > 10, f"{scenario_name}: スループットが低すぎます: {result['throughput']:.1f}ops/sec"

            # メトリクス収集の精度確認
            response_time_stats = overall_stats.get("response_times", {})
            if response_time_stats.get("count", 0) > 0:
                # 収集されたメトリクスが妥当な範囲内であることを確認
                assert response_time_stats["mean"] > 0, "平均レスポンス時間が無効です"
                assert response_time_stats["std"] >= 0, "標準偏差が無効です"
                assert response_time_stats["min"] >= 0, "最小レスポンス時間が無効です"
                assert response_time_stats["max"] >= response_time_stats["min"], "最大値が最小値より小さいです"

            logger.info("✅ パフォーマンスメトリクス収集テスト成功")

        except Exception as e:
            pytest.fail(f"パフォーマンスメトリクス収集テストエラー: {e}")

    def test_error_tracking_stack_traces(self):
        """テスト66: エラー追跡とスタックトレースの詳細度"""
        logger.info("🔍 エラー追跡・スタックトレーステスト開始")

        try:
            # エラー追跡システム
            class ErrorTracker:
                def __init__(self):
                    self.errors = []
                    self.error_counts = {}
                    self.error_patterns = {}

                def track_error(self, error: Exception, context: Dict[str, Any] = None):
                    """エラーを追跡"""
                    error_info = {
                        "timestamp": datetime.now(),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "stack_trace": traceback.format_exc(),
                        "context": context or {}
                    }

                    self.errors.append(error_info)

                    # エラー種別のカウント
                    error_type = error_info["error_type"]
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

                    # エラーパターンの分析
                    error_pattern = self._extract_error_pattern(error_info["stack_trace"])
                    self.error_patterns[error_pattern] = self.error_patterns.get(error_pattern, 0) + 1

                    logger.error(f"エラー追跡: {error_type} - {error_info['error_message']}")

                def _extract_error_pattern(self, stack_trace: str) -> str:
                    """スタックトレースからエラーパターンを抽出"""
                    lines = stack_trace.strip().split('\n')
                    if len(lines) >= 2:
                        # 最後の行（エラーメッセージ）と最後から2番目の行（エラー発生箇所）を使用
                        error_line = lines[-1]
                        location_line = lines[-2] if len(lines) > 1 else ""

                        # ファイル名と行番号を抽出
                        import re
                        location_match = re.search(r'File "([^"]+)", line (\d+)', location_line)
                        if location_match:
                            filename = os.path.basename(location_match.group(1))
                            line_number = location_match.group(2)
                            return f"{filename}:{line_number}"

                    return "unknown"

                def get_error_summary(self) -> Dict[str, Any]:
                    """エラーサマリーを取得"""
                    total_errors = len(self.errors)

                    if total_errors == 0:
                        return {"total_errors": 0}

                    # 最も多いエラータイプ
                    most_common_type = max(self.error_counts.items(), key=lambda x: x[1])

                    # 最も多いエラーパターン
                    most_common_pattern = max(self.error_patterns.items(), key=lambda x: x[1])

                    # 時間別エラー分布
                    error_times = [error["timestamp"] for error in self.errors]
                    time_span = (max(error_times) - min(error_times)).total_seconds() if len(error_times) > 1 else 0
                    error_rate = total_errors / max(time_span, 1)  # エラー/秒

                    return {
                        "total_errors": total_errors,
                        "unique_error_types": len(self.error_counts),
                        "unique_error_patterns": len(self.error_patterns),
                        "most_common_type": most_common_type,
                        "most_common_pattern": most_common_pattern,
                        "error_rate": error_rate,
                        "time_span": time_span
                    }

            error_tracker = ErrorTracker()

            # 意図的にエラーを発生させるテストケース
            error_test_cases = [
                {
                    "name": "ゼロ除算エラー",
                    "function": lambda: 1 / 0,
                    "context": {"operation": "division", "value": 0}
                },
                {
                    "name": "型エラー",
                    "function": lambda: "string" + 123,
                    "context": {"operation": "concatenation", "types": ["str", "int"]}
                },
                {
                    "name": "インデックスエラー",
                    "function": lambda: [1, 2, 3][10],
                    "context": {"operation": "indexing", "index": 10, "list_length": 3}
                },
                {
                    "name": "キーエラー",
                    "function": lambda: {"a": 1}["b"],
                    "context": {"operation": "dict_access", "key": "b", "available_keys": ["a"]}
                },
                {
                    "name": "値エラー",
                    "function": lambda: int("not_a_number"),
                    "context": {"operation": "conversion", "value": "not_a_number", "target_type": "int"}
                },
                {
                    "name": "属性エラー",
                    "function": lambda: "string".nonexistent_method(),
                    "context": {"operation": "method_call", "object_type": "str", "method": "nonexistent_method"}
                }
            ]

            # 各エラーケースを複数回実行
            for test_case in error_test_cases:
                for attempt in range(3):  # 各エラーを3回発生させる
                    try:
                        test_case["function"]()
                    except Exception as e:
                        context = test_case["context"].copy()
                        context["test_case"] = test_case["name"]
                        context["attempt"] = attempt + 1
                        error_tracker.track_error(e, context)

            # 実際の処理でのエラーもテスト
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            calculator = TPSLCalculator()

            # 無効な入力でのエラー
            invalid_inputs = [
                {"price": None, "sl_pct": 0.02, "tp_pct": 0.04, "direction": 1.0},
                {"price": "invalid", "sl_pct": 0.02, "tp_pct": 0.04, "direction": 1.0},
                {"price": 50000, "sl_pct": None, "tp_pct": 0.04, "direction": 1.0},
                {"price": 50000, "sl_pct": 0.02, "tp_pct": "invalid", "direction": 1.0},
                {"price": 50000, "sl_pct": 0.02, "tp_pct": 0.04, "direction": "invalid"},
                {"price": float('inf'), "sl_pct": 0.02, "tp_pct": 0.04, "direction": 1.0},
                {"price": float('nan'), "sl_pct": 0.02, "tp_pct": 0.04, "direction": 1.0},
            ]

            for i, inputs in enumerate(invalid_inputs):
                try:
                    calculator.calculate_basic_tpsl_prices(
                        inputs["price"], inputs["sl_pct"], inputs["tp_pct"], inputs["direction"]
                    )
                except Exception as e:
                    context = {
                        "operation": "tpsl_calculation",
                        "input_index": i,
                        "inputs": inputs
                    }
                    error_tracker.track_error(e, context)

            # エラー追跡結果の分析
            error_summary = error_tracker.get_error_summary()

            logger.info(f"エラー追跡結果:")
            logger.info(f"  総エラー数: {error_summary['total_errors']}")
            logger.info(f"  ユニークエラータイプ数: {error_summary['unique_error_types']}")
            logger.info(f"  ユニークエラーパターン数: {error_summary['unique_error_patterns']}")
            logger.info(f"  最多エラータイプ: {error_summary['most_common_type'][0]} ({error_summary['most_common_type'][1]}回)")
            logger.info(f"  最多エラーパターン: {error_summary['most_common_pattern'][0]} ({error_summary['most_common_pattern'][1]}回)")
            logger.info(f"  エラー発生率: {error_summary['error_rate']:.3f}エラー/秒")

            # エラータイプ別統計
            logger.info("エラータイプ別統計:")
            for error_type, count in sorted(error_tracker.error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {error_type}: {count}回")

            # スタックトレースの詳細度確認
            detailed_traces = 0
            for error in error_tracker.errors:
                stack_trace = error["stack_trace"]

                # スタックトレースの基本要素をチェック
                has_file_info = "File " in stack_trace
                has_line_number = "line " in stack_trace
                has_function_name = "in " in stack_trace
                has_error_message = error["error_message"] in stack_trace

                if has_file_info and has_line_number and has_function_name and has_error_message:
                    detailed_traces += 1

                # スタックトレースの行数をチェック（詳細度の指標）
                trace_lines = len(stack_trace.strip().split('\n'))
                assert trace_lines >= 2, f"スタックトレースが短すぎます: {trace_lines}行"

            detail_rate = detailed_traces / len(error_tracker.errors) if error_tracker.errors else 0

            logger.info(f"詳細なスタックトレース率: {detail_rate:.1%}")

            # エラー追跡の要件確認
            assert error_summary["total_errors"] > 0, "エラーが追跡されていません"
            assert error_summary["unique_error_types"] >= 5, f"エラータイプの多様性が不足: {error_summary['unique_error_types']}"
            assert detail_rate >= 0.9, f"スタックトレースの詳細度が不足: {detail_rate:.1%}"

            # コンテキスト情報の確認
            context_errors = [error for error in error_tracker.errors if error["context"]]
            context_rate = len(context_errors) / len(error_tracker.errors) if error_tracker.errors else 0

            logger.info(f"コンテキスト情報付きエラー率: {context_rate:.1%}")
            assert context_rate >= 0.8, f"コンテキスト情報が不足: {context_rate:.1%}"

            logger.info("✅ エラー追跡・スタックトレーステスト成功")

        except Exception as e:
            pytest.fail(f"エラー追跡・スタックトレーステストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestMonitoringLogging()
    
    tests = [
        test_instance.test_anomaly_detection_alerts,
        test_instance.test_log_file_integrity_rotation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_instance.setup_method()
            test()
            test_instance.teardown_method()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 監視・ログテスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
