"""
監視システムの包括的テスト

新規実装された監視・アラート・バックアップシステムをテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TestMonitoringSystems:
    """監視システムのテストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        # テスト用データの作成
        self.test_data = self.create_test_data()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_data(self, size: int = 1000) -> pd.DataFrame:
        """テスト用データを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='H')
        
        # 基本価格データ
        close_prices = 50000 + np.cumsum(np.random.randn(size) * 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': close_prices + np.random.randn(size) * 50,
            'High': close_prices + np.abs(np.random.randn(size)) * 100,
            'Low': close_prices - np.abs(np.random.randn(size)) * 100,
            'Close': close_prices,
            'Volume': np.random.exponential(1000, size),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_data_drift_detector(self):
        """データドリフト検出テスト"""
        logger.info("🔍 データドリフト検出テスト開始")
        
        try:
            from app.services.ml.monitoring.data_drift_detector import DataDriftDetector, DriftType
            
            detector = DataDriftDetector()
            
            # 参照データを設定
            reference_data = self.test_data.iloc[:500]
            detector.set_reference_data(reference_data)
            
            # 現在のデータ（ドリフトあり）
            current_data = self.test_data.iloc[500:].copy()
            # 意図的にドリフトを作成
            current_data['Close'] *= 1.2  # 20%の価格上昇
            current_data['Volume'] *= 0.8  # 20%のボリューム減少
            
            # ドリフト検出を実行
            drift_results = detector.detect_drift(current_data)
            
            # 結果の検証
            assert len(drift_results) > 0, "ドリフト検出結果が空です"
            
            # ドリフトが検出されていることを確認
            drift_detected = any(result.drift_type != DriftType.NO_DRIFT for result in drift_results)
            assert drift_detected, "ドリフトが検出されませんでした"
            
            # サマリー情報を取得
            summary = detector.get_drift_summary(hours=1)
            assert isinstance(summary, dict), "サマリー情報が無効です"
            assert 'total_detections' in summary, "サマリーに検出数が含まれていません"
            
            logger.info(f"✅ データドリフト検出成功: {len(drift_results)}個の結果")
            
        except Exception as e:
            pytest.fail(f"データドリフト検出エラー: {e}")
    
    def test_realtime_monitor(self):
        """リアルタイム監視テスト"""
        logger.info("🔍 リアルタイム監視テスト開始")
        
        try:
            from app.services.ml.monitoring.realtime_monitor import RealtimeMonitor
            
            monitor = RealtimeMonitor()
            
            # アラートコールバックを設定
            received_alerts = []
            def alert_callback(alert):
                received_alerts.append(alert)
            
            monitor.add_alert_callback(alert_callback)
            
            # 予測実行を記録
            monitor.record_prediction("test_model", 150.0, True)  # 正常
            monitor.record_prediction("test_model", 2000.0, False)  # 高レイテンシ・エラー
            monitor.record_prediction("test_model", 100.0, True)  # 正常
            
            # システムメトリクスをチェック（手動実行）
            monitor._check_system_metrics()
            
            # モデルメトリクスをチェック（手動実行）
            monitor._check_model_metrics()
            
            # 監視状態を取得
            status = monitor.get_monitoring_status()
            assert isinstance(status, dict), "監視状態が無効です"
            assert 'is_running' in status, "監視状態に実行状況が含まれていません"
            assert 'monitored_models' in status, "監視状態にモデル情報が含まれていません"
            assert 'test_model' in status['monitored_models'], "テストモデルが監視対象に含まれていません"
            
            logger.info(f"✅ リアルタイム監視成功: {len(received_alerts)}個のアラート")
            
        except Exception as e:
            pytest.fail(f"リアルタイム監視エラー: {e}")
    
    def test_alert_system(self):
        """アラートシステムテスト"""
        logger.info("🔍 アラートシステムテスト開始")
        
        try:
            from app.services.ml.monitoring.alert_system import AlertSystem
            from app.services.ml.monitoring.realtime_monitor import Alert, AlertLevel
            
            alert_system = AlertSystem()
            
            # テスト用アラートを作成
            test_alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                category="test",
                message="テストアラート",
                details={"test_key": "test_value"}
            )
            
            # アラート処理を実行（非同期）
            async def test_alert_handling():
                await alert_system.handle_alert(test_alert)
                
                # 保留中のエスカレーションを処理
                await alert_system.process_pending_escalations()
                
                # 統計情報を取得
                stats = alert_system.get_alert_statistics()
                assert isinstance(stats, dict), "アラート統計が無効です"
                assert 'enabled_channels' in stats, "統計に有効チャネル情報が含まれていません"
                
                return True
            
            # 非同期テストを実行
            result = asyncio.run(test_alert_handling())
            assert result, "アラート処理が失敗しました"
            
            logger.info("✅ アラートシステム成功")
            
        except Exception as e:
            pytest.fail(f"アラートシステムエラー: {e}")
    
    def test_backup_system(self):
        """バックアップシステムテスト"""
        logger.info("🔍 バックアップシステムテスト開始")
        
        try:
            from app.services.ml.backup.backup_system import BackupSystem, BackupType, BackupConfig
            
            # テスト用設定
            config = BackupConfig(
                model_directory=os.path.join(self.temp_dir, "models"),
                data_directory=os.path.join(self.temp_dir, "data"),
                config_directory=os.path.join(self.temp_dir, "config"),
                backup_root=os.path.join(self.temp_dir, "backups"),
                auto_backup_enabled=False,  # テストでは手動実行
                verify_backups=True
            )
            
            backup_system = BackupSystem(config)
            
            # テスト用ファイルを作成
            os.makedirs(config.model_directory, exist_ok=True)
            os.makedirs(config.data_directory, exist_ok=True)
            os.makedirs(config.config_directory, exist_ok=True)
            
            # ダミーファイルを作成
            with open(os.path.join(config.model_directory, "test_model.pkl"), 'w') as f:
                f.write("test model data")
            
            with open(os.path.join(config.data_directory, "test_data.csv"), 'w') as f:
                f.write("test,data\n1,2\n3,4\n")
            
            with open(os.path.join(config.config_directory, "test_config.json"), 'w') as f:
                f.write('{"test": "config"}')
            
            # フルバックアップを作成
            backup_record = backup_system.create_backup(BackupType.FULL)
            
            # バックアップ結果を検証
            assert backup_record is not None, "バックアップ記録が作成されませんでした"
            assert backup_record.files_count > 0, "バックアップにファイルが含まれていません"
            assert os.path.exists(backup_record.file_path), "バックアップファイルが存在しません"
            
            # バックアップ状態を取得
            status = backup_system.get_backup_status()
            assert isinstance(status, dict), "バックアップ状態が無効です"
            assert status['total_backups'] > 0, "バックアップ数が0です"
            
            # 復元テスト
            restore_path = os.path.join(self.temp_dir, "restored")
            restore_success = backup_system.restore_backup(backup_record.backup_id, restore_path)
            assert restore_success, "バックアップ復元が失敗しました"
            assert os.path.exists(restore_path), "復元ディレクトリが作成されませんでした"
            
            logger.info(f"✅ バックアップシステム成功: {backup_record.files_count}ファイル")
            
        except Exception as e:
            pytest.fail(f"バックアップシステムエラー: {e}")
    
    def test_integrated_monitoring_workflow(self):
        """統合監視ワークフローテスト"""
        logger.info("🔍 統合監視ワークフローテスト開始")
        
        try:
            from app.services.ml.monitoring.data_drift_detector import DataDriftDetector
            from app.services.ml.monitoring.realtime_monitor import RealtimeMonitor
            from app.services.ml.monitoring.alert_system import AlertSystem
            
            # 各システムを初期化
            drift_detector = DataDriftDetector()
            monitor = RealtimeMonitor()
            alert_system = AlertSystem()
            
            # 統合ワークフローをテスト
            
            # 1. データドリフト検出
            reference_data = self.test_data.iloc[:300]
            drift_detector.set_reference_data(reference_data)
            
            current_data = self.test_data.iloc[300:600].copy()
            current_data['Close'] *= 1.5  # 大きなドリフトを作成
            
            drift_results = drift_detector.detect_drift(current_data)
            
            # 2. 監視システムでの記録
            monitor.record_prediction("integrated_test_model", 500.0, True)
            monitor.record_prediction("integrated_test_model", 3000.0, False)  # 高レイテンシ
            
            # 3. アラート処理
            received_alerts = []
            def integrated_alert_callback(alert):
                received_alerts.append(alert)
            
            # AlertSystemには直接コールバック追加機能がないため、スキップ
            
            # 統合テストの検証
            assert len(drift_results) > 0, "統合テストでドリフトが検出されませんでした"
            
            # 監視状態の確認
            monitor_status = monitor.get_monitoring_status()
            assert 'integrated_test_model' in monitor_status['monitored_models'], "統合テストモデルが監視されていません"
            
            logger.info("✅ 統合監視ワークフロー成功")
            
        except Exception as e:
            pytest.fail(f"統合監視ワークフローエラー: {e}")
    
    def test_performance_and_scalability(self):
        """パフォーマンス・スケーラビリティテスト"""
        logger.info("🔍 パフォーマンス・スケーラビリティテスト開始")
        
        try:
            from app.services.ml.monitoring.data_drift_detector import DataDriftDetector
            import time
            
            detector = DataDriftDetector()
            
            # 大きなデータセットでのテスト
            large_reference_data = self.create_test_data(5000)
            large_current_data = self.create_test_data(2000)
            
            # パフォーマンス測定
            start_time = time.time()
            
            detector.set_reference_data(large_reference_data)
            drift_results = detector.detect_drift(large_current_data)
            
            processing_time = time.time() - start_time
            
            # パフォーマンス検証
            assert processing_time < 30.0, f"処理時間が長すぎます: {processing_time:.2f}秒"
            assert len(drift_results) > 0, "大規模データでドリフト検出が失敗しました"
            
            # メモリ使用量の確認（簡易）
            assert len(detector.drift_history) <= 10000, "ドリフト履歴が制限を超えています"
            
            logger.info(f"✅ パフォーマンステスト成功: {processing_time:.3f}秒")
            
        except Exception as e:
            pytest.fail(f"パフォーマンステストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestMonitoringSystems()
    test_instance.setup_method()
    
    # 各テストを実行
    tests = [
        test_instance.test_data_drift_detector,
        test_instance.test_realtime_monitor,
        test_instance.test_alert_system,
        test_instance.test_backup_system,
        test_instance.test_integrated_monitoring_workflow,
        test_instance.test_performance_and_scalability,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
        finally:
            # 各テスト後にクリーンアップ
            try:
                test_instance.teardown_method()
                test_instance.setup_method()
            except:
                pass
    
    # 最終クリーンアップ
    test_instance.teardown_method()
    
    print(f"\n📊 監視システムテスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
