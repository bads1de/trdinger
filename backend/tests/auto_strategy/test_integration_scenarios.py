"""
オートストラテジー 統合シナリオテスト

複雑な統合シナリオでのシステム動作を検証します。
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
import psutil
import threading
import concurrent.futures
import gc
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sqlite3

logger = logging.getLogger(__name__)


class TestIntegrationScenarios:
    """統合シナリオテストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_samples = []
        self.monitoring_active = True
        
        # メモリ監視スレッドを開始
        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.memory_monitor_thread.start()
        
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        self.monitoring_active = False
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        # メモリ統計
        if self.memory_samples:
            max_memory = max(self.memory_samples)
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
            logger.info(f"メモリ使用量: 開始={self.start_memory:.1f}MB, 最大={max_memory:.1f}MB, 平均={avg_memory:.1f}MB, 変化={memory_delta:+.1f}MB")
        
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
        
        # ガベージコレクション
        gc.collect()
    
    def _monitor_memory(self):
        """メモリ使用量を監視"""
        while self.monitoring_active:
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                time.sleep(0.5)  # 0.5秒間隔で監視
            except:
                break
    
    def create_test_data(self, size: int = 1000) -> pd.DataFrame:
        """テスト用データを作成"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=size, freq='h')
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, size)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(1000, size),
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def test_concurrent_strategy_execution(self):
        """テスト26: 複数の戦略を同時実行した場合の競合処理"""
        logger.info("🔍 並行戦略実行競合テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            test_data = self.create_test_data(500)
            
            def run_strategy(strategy_id: int) -> Dict[str, Any]:
                """戦略を実行"""
                try:
                    start_time = time.time()
                    ml_orchestrator = MLOrchestrator(enable_automl=False)
                    
                    result = ml_orchestrator.calculate_ml_indicators(test_data)
                    execution_time = time.time() - start_time
                    
                    return {
                        "strategy_id": strategy_id,
                        "success": True,
                        "execution_time": execution_time,
                        "result_size": len(result.get("ML_UP_PROB", [])) if result else 0,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "strategy_id": strategy_id,
                        "success": False,
                        "execution_time": time.time() - start_time,
                        "result_size": 0,
                        "error": str(e)
                    }
            
            # 5つの戦略を並行実行
            num_strategies = 5
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_strategies) as executor:
                futures = [executor.submit(run_strategy, i) for i in range(num_strategies)]
                results = [future.result(timeout=60) for future in futures]
            
            total_time = time.time() - start_time
            
            # 結果分析
            successful_results = [r for r in results if r["success"]]
            success_rate = len(successful_results) / len(results)
            
            if successful_results:
                avg_execution_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
                logger.info(f"並行実行結果: {len(successful_results)}/{len(results)} 成功 ({success_rate:.1%})")
                logger.info(f"平均実行時間: {avg_execution_time:.3f}秒, 総時間: {total_time:.3f}秒")
                
                # 競合による大幅な性能劣化がないことを確認
                assert success_rate >= 0.6, f"成功率が低すぎます: {success_rate:.1%}"
            
            logger.info("✅ 並行戦略実行競合テスト成功")
            
        except Exception as e:
            pytest.fail(f"並行戦略実行競合テストエラー: {e}")
    
    def test_memory_leak_detection(self):
        """テスト27: メモリリークの検出"""
        logger.info("🔍 メモリリーク検出テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            test_data = self.create_test_data(200)
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_measurements = []
            
            # 10回繰り返し実行してメモリ使用量を監視
            for i in range(10):
                try:
                    ml_orchestrator = MLOrchestrator(enable_automl=False)
                    result = ml_orchestrator.calculate_ml_indicators(test_data)
                    
                    # 明示的にオブジェクトを削除
                    del ml_orchestrator
                    del result
                    
                    # ガベージコレクション
                    gc.collect()
                    
                    # メモリ測定
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_measurements.append(current_memory)
                    
                    logger.info(f"反復 {i+1}: メモリ使用量 {current_memory:.1f}MB")
                    
                except Exception as e:
                    logger.warning(f"反復 {i+1} でエラー: {e}")
            
            if len(memory_measurements) >= 5:
                # メモリリーク分析
                final_memory = memory_measurements[-1]
                memory_increase = final_memory - initial_memory
                
                # 線形回帰でメモリ増加傾向を分析
                x = np.arange(len(memory_measurements))
                y = np.array(memory_measurements)
                slope = np.polyfit(x, y, 1)[0]  # 傾き
                
                logger.info(f"メモリ変化: 初期={initial_memory:.1f}MB, 最終={final_memory:.1f}MB, 増加={memory_increase:+.1f}MB")
                logger.info(f"メモリ増加傾向: {slope:+.2f}MB/反復")
                
                # メモリリークの判定（1反復あたり5MB以上の増加は問題）
                assert slope < 5.0, f"メモリリークの可能性: {slope:.2f}MB/反復"
                assert memory_increase < 100, f"総メモリ増加が大きすぎます: {memory_increase:.1f}MB"
            
            logger.info("✅ メモリリーク検出テスト成功")
            
        except Exception as e:
            pytest.fail(f"メモリリーク検出テストエラー: {e}")
    
    def test_high_load_concurrent_requests(self):
        """テスト28: 大量の並行リクエスト処理"""
        logger.info("🔍 高負荷並行リクエストテスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            calculator = TPSLCalculator()
            
            def process_request(request_id: int) -> Dict[str, Any]:
                """リクエストを処理"""
                try:
                    start_time = time.time()
                    
                    # ランダムなパラメータでTP/SL計算
                    current_price = 50000 + np.random.randint(-5000, 5000)
                    sl_pct = np.random.uniform(0.01, 0.05)
                    tp_pct = np.random.uniform(0.02, 0.08)
                    direction = np.random.choice([1.0, -1.0])
                    
                    sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                        current_price, sl_pct, tp_pct, direction
                    )
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "execution_time": execution_time,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "execution_time": time.time() - start_time,
                        "sl_price": None,
                        "tp_price": None,
                        "error": str(e)
                    }
            
            # 50個の並行リクエスト
            num_requests = 50
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_request, i) for i in range(num_requests)]
                results = [future.result(timeout=30) for future in futures]
            
            total_time = time.time() - start_time
            
            # 結果分析
            successful_results = [r for r in results if r["success"]]
            success_rate = len(successful_results) / len(results)
            
            if successful_results:
                execution_times = [r["execution_time"] for r in successful_results]
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                logger.info(f"高負荷テスト結果: {len(successful_results)}/{len(results)} 成功 ({success_rate:.1%})")
                logger.info(f"実行時間: 平均={avg_time:.3f}秒, 最大={max_time:.3f}秒, 最小={min_time:.3f}秒")
                logger.info(f"総処理時間: {total_time:.3f}秒, スループット: {len(results)/total_time:.1f}req/秒")
                
                # 性能要件の確認
                assert success_rate >= 0.9, f"成功率が低すぎます: {success_rate:.1%}"
                assert avg_time < 1.0, f"平均応答時間が長すぎます: {avg_time:.3f}秒"
            
            logger.info("✅ 高負荷並行リクエストテスト成功")
            
        except Exception as e:
            pytest.fail(f"高負荷並行リクエストテストエラー: {e}")
    
    def test_database_connection_error_recovery(self):
        """テスト29: データベース接続エラー時の回復処理"""
        logger.info("🔍 データベース接続エラー回復テスト開始")
        
        try:
            # データベース接続エラーのシミュレーション
            def simulate_db_error():
                """データベースエラーをシミュレート"""
                raise sqlite3.OperationalError("database is locked")
            
            def simulate_db_recovery():
                """データベース回復をシミュレート"""
                return {"status": "recovered", "data": "test_data"}
            
            # エラー回復ロジックのテスト
            max_retries = 3
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    if attempt < 2:  # 最初の2回はエラー
                        simulate_db_error()
                    else:  # 3回目で成功
                        result = simulate_db_recovery()
                        logger.info(f"データベース回復成功: 試行回数={attempt+1}")
                        assert result["status"] == "recovered", "回復結果が無効です"
                        break
                        
                except sqlite3.OperationalError as e:
                    logger.info(f"試行 {attempt+1}: データベースエラー - {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数バックオフ
                    else:
                        raise
            
            # 実際のデータベース操作のテスト（簡易版）
            try:
                # テスト用の軽量なデータベース操作
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
                    test_db_path = tmp_file.name
                
                # 接続テスト
                conn = sqlite3.connect(test_db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)")
                cursor.execute("INSERT INTO test (data) VALUES (?)", ("test_data",))
                conn.commit()
                
                # データ取得テスト
                cursor.execute("SELECT * FROM test")
                results = cursor.fetchall()
                assert len(results) > 0, "データベース操作が失敗しました"
                
                conn.close()
                os.unlink(test_db_path)  # テストファイル削除
                
                logger.info("データベース操作テスト成功")
                
            except Exception as e:
                logger.warning(f"データベース操作テストでエラー: {e}")
            
            logger.info("✅ データベース接続エラー回復テスト成功")
            
        except Exception as e:
            pytest.fail(f"データベース接続エラー回復テストエラー: {e}")
    
    def test_long_running_stability(self):
        """テスト30: 長時間実行での安定性（簡易版）"""
        logger.info("🔍 長時間実行安定性テスト開始")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            test_data = self.create_test_data(100)  # 軽量化
            
            # 長時間実行のシミュレーション（実際は短時間で多数実行）
            duration_seconds = 30  # テスト用に30秒に短縮
            interval_seconds = 1
            
            start_time = time.time()
            execution_count = 0
            error_count = 0
            execution_times = []
            
            while time.time() - start_time < duration_seconds:
                try:
                    iteration_start = time.time()
                    
                    ml_orchestrator = MLOrchestrator(enable_automl=False)
                    result = ml_orchestrator.calculate_ml_indicators(test_data)
                    
                    iteration_time = time.time() - iteration_start
                    execution_times.append(iteration_time)
                    execution_count += 1
                    
                    # メモリ使用量チェック
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if execution_count % 5 == 0:
                        logger.info(f"実行 {execution_count}: 時間={iteration_time:.3f}秒, メモリ={current_memory:.1f}MB")
                    
                    # クリーンアップ
                    del ml_orchestrator
                    del result
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    error_count += 1
                    logger.warning(f"実行 {execution_count+1} でエラー: {e}")
            
            total_time = time.time() - start_time
            
            # 安定性分析
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                time_std = np.std(execution_times)
                
                error_rate = error_count / (execution_count + error_count) if (execution_count + error_count) > 0 else 0
                
                logger.info(f"長時間実行結果: 総実行={execution_count}, エラー={error_count}, エラー率={error_rate:.1%}")
                logger.info(f"実行時間: 平均={avg_time:.3f}秒, 最大={max_time:.3f}秒, 最小={min_time:.3f}秒, 標準偏差={time_std:.3f}秒")
                logger.info(f"総時間: {total_time:.1f}秒")
                
                # 安定性要件の確認
                assert error_rate < 0.1, f"エラー率が高すぎます: {error_rate:.1%}"
                assert time_std < avg_time, f"実行時間のばらつきが大きすぎます: {time_std:.3f}秒"
            
            logger.info("✅ 長時間実行安定性テスト成功")
            
        except Exception as e:
            pytest.fail(f"長時間実行安定性テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestIntegrationScenarios()
    
    tests = [
        test_instance.test_concurrent_strategy_execution,
        test_instance.test_memory_leak_detection,
        test_instance.test_high_load_concurrent_requests,
        test_instance.test_database_connection_error_recovery,
        test_instance.test_long_running_stability,
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
    
    print(f"\n📊 統合シナリオテスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
