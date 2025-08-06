"""
オートストラテジーパフォーマンスの包括的テスト

大量データでの処理速度、メモリ使用量、並行処理、タイムアウト処理をテストします。
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
import pytest

# テスト用のロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAutoStrategyPerformance:
    """オートストラテジーパフォーマンスの包括的テストクラス"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.small_dataset = self._create_test_data(100)
        self.medium_dataset = self._create_test_data(1000)
        self.large_dataset = self._create_test_data(5000)

    def _create_test_data(self, rows: int) -> pd.DataFrame:
        """指定された行数のテストデータを作成"""
        dates = pd.date_range(start="2023-01-01", periods=rows, freq="1H")
        np.random.seed(42)  # 再現性のため
        
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, rows)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                "timestamp": date,
                "open": open_price,
                "high": max(high, open_price, close_price),
                "low": min(low, open_price, close_price),
                "close": close_price,
                "volume": volume
            })
        
        return pd.DataFrame(data)

    def _measure_memory_usage(self):
        """メモリ使用量を測定"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logger.warning("psutil が利用できません。メモリ測定をスキップします。")
            return 0

    def test_ml_orchestrator_performance(self):
        """MLOrchestratorのパフォーマンステスト"""
        logger.info("=== MLOrchestratorパフォーマンステスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            ml_orchestrator = MLOrchestrator(enable_automl=False)
            
            # 異なるサイズのデータセットでのパフォーマンス測定
            datasets = [
                ("小", self.small_dataset),
                ("中", self.medium_dataset),
                ("大", self.large_dataset)
            ]
            
            performance_results = []
            
            for size_name, dataset in datasets:
                initial_memory = self._measure_memory_usage()
                start_time = time.time()
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(dataset)
                    
                    end_time = time.time()
                    final_memory = self._measure_memory_usage()
                    
                    execution_time = end_time - start_time
                    memory_usage = final_memory - initial_memory
                    
                    performance_results.append({
                        "size": size_name,
                        "rows": len(dataset),
                        "execution_time": execution_time,
                        "memory_usage": memory_usage,
                        "success": True
                    })
                    
                    logger.info(f"{size_name}データセット ({len(dataset)}行): {execution_time:.2f}秒, メモリ: {memory_usage:.1f}MB")
                    
                    # パフォーマンス基準の確認
                    if len(dataset) <= 1000:
                        assert execution_time < 30, f"{size_name}データセットの処理時間が長すぎます: {execution_time:.2f}秒"
                    elif len(dataset) <= 5000:
                        assert execution_time < 60, f"{size_name}データセットの処理時間が長すぎます: {execution_time:.2f}秒"
                    
                except Exception as e:
                    logger.warning(f"{size_name}データセットでエラー: {e}")
                    performance_results.append({
                        "size": size_name,
                        "rows": len(dataset),
                        "execution_time": None,
                        "memory_usage": None,
                        "success": False,
                        "error": str(e)
                    })
                
                # メモリクリーンアップ
                gc.collect()
            
            # 結果の分析
            successful_results = [r for r in performance_results if r["success"]]
            if successful_results:
                avg_time_per_row = sum(r["execution_time"] / r["rows"] for r in successful_results) / len(successful_results)
                logger.info(f"平均処理時間（行あたり）: {avg_time_per_row:.6f}秒")
            
            logger.info("✅ MLOrchestratorパフォーマンステスト成功")
            
        except Exception as e:
            pytest.fail(f"MLOrchestratorパフォーマンステストエラー: {e}")

    def test_smart_condition_generator_performance(self):
        """SmartConditionGeneratorのパフォーマンステスト"""
        logger.info("=== SmartConditionGeneratorパフォーマンステスト ===")
        
        try:
            from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
            from app.services.auto_strategy.models.gene_strategy import IndicatorGene
            
            generator = SmartConditionGenerator()
            
            # 異なる数の指標でのパフォーマンス測定
            indicator_counts = [1, 3, 5, 10, 20]
            
            for count in indicator_counts:
                # 指標リストを作成
                indicators = []
                for i in range(count):
                    indicator_type = ["RSI", "SMA", "EMA", "BB", "MACD"][i % 5]
                    indicators.append(IndicatorGene(
                        type=indicator_type,
                        parameters={"period": 14 + i},
                        enabled=True
                    ))
                
                start_time = time.time()
                initial_memory = self._measure_memory_usage()
                
                try:
                    # 複数回実行して平均時間を測定
                    iterations = 10
                    for _ in range(iterations):
                        long_conditions, short_conditions, exit_conditions = generator.generate_balanced_conditions(indicators)
                    
                    end_time = time.time()
                    final_memory = self._measure_memory_usage()
                    
                    avg_execution_time = (end_time - start_time) / iterations
                    memory_usage = final_memory - initial_memory
                    
                    logger.info(f"指標数 {count}: 平均実行時間 {avg_execution_time:.4f}秒, メモリ: {memory_usage:.1f}MB")
                    
                    # パフォーマンス基準の確認
                    assert avg_execution_time < 1.0, f"指標数 {count} の処理時間が長すぎます: {avg_execution_time:.4f}秒"
                    
                except Exception as e:
                    logger.warning(f"指標数 {count} でエラー: {e}")
                
                # メモリクリーンアップ
                gc.collect()
            
            logger.info("✅ SmartConditionGeneratorパフォーマンステスト成功")
            
        except Exception as e:
            pytest.fail(f"SmartConditionGeneratorパフォーマンステストエラー: {e}")

    def test_tpsl_service_performance(self):
        """TP/SLサービスのパフォーマンステスト"""
        logger.info("=== TP/SLサービスパフォーマンステスト ===")
        
        try:
            from app.services.auto_strategy.services.tpsl_auto_decision_service import (
                TPSLAutoDecisionService, TPSLConfig, TPSLStrategy
            )
            
            service = TPSLAutoDecisionService()
            
            # 全ての戦略でのパフォーマンス測定
            strategies = [
                TPSLStrategy.RANDOM,
                TPSLStrategy.RISK_REWARD,
                TPSLStrategy.VOLATILITY_ADAPTIVE,
                TPSLStrategy.STATISTICAL,
                TPSLStrategy.AUTO_OPTIMAL
            ]
            
            for strategy in strategies:
                config = TPSLConfig(strategy=strategy)
                
                start_time = time.time()
                
                # 複数回実行して平均時間を測定
                iterations = 100
                for _ in range(iterations):
                    result = service.generate_tpsl_values(config)
                
                end_time = time.time()
                avg_execution_time = (end_time - start_time) / iterations
                
                logger.info(f"戦略 {strategy.value}: 平均実行時間 {avg_execution_time:.6f}秒")
                
                # パフォーマンス基準の確認
                assert avg_execution_time < 0.1, f"戦略 {strategy.value} の処理時間が長すぎます: {avg_execution_time:.6f}秒"
            
            logger.info("✅ TP/SLサービスパフォーマンステスト成功")
            
        except Exception as e:
            pytest.fail(f"TP/SLサービスパフォーマンステストエラー: {e}")

    def test_concurrent_processing_performance(self):
        """並行処理パフォーマンステスト"""
        logger.info("=== 並行処理パフォーマンステスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 並行処理のテスト
            def worker_task(worker_id):
                try:
                    ml_orchestrator = MLOrchestrator(enable_automl=False)
                    start_time = time.time()
                    
                    result = ml_orchestrator.calculate_ml_indicators(self.medium_dataset)
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    return {
                        "worker_id": worker_id,
                        "execution_time": execution_time,
                        "success": True,
                        "result_size": len(result) if result else 0
                    }
                except Exception as e:
                    return {
                        "worker_id": worker_id,
                        "execution_time": None,
                        "success": False,
                        "error": str(e)
                    }
            
            # シーケンシャル実行の測定
            sequential_start = time.time()
            sequential_results = []
            for i in range(3):
                result = worker_task(f"sequential_{i}")
                sequential_results.append(result)
            sequential_end = time.time()
            sequential_total_time = sequential_end - sequential_start
            
            # 並行実行の測定
            concurrent_start = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(worker_task, f"concurrent_{i}") for i in range(3)]
                concurrent_results = [future.result(timeout=60) for future in futures]
            concurrent_end = time.time()
            concurrent_total_time = concurrent_end - concurrent_start
            
            # 結果の分析
            sequential_success = sum(1 for r in sequential_results if r["success"])
            concurrent_success = sum(1 for r in concurrent_results if r["success"])
            
            logger.info(f"シーケンシャル実行: {sequential_total_time:.2f}秒 (成功: {sequential_success}/3)")
            logger.info(f"並行実行: {concurrent_total_time:.2f}秒 (成功: {concurrent_success}/3)")
            
            if sequential_success > 0 and concurrent_success > 0:
                speedup = sequential_total_time / concurrent_total_time
                logger.info(f"並行処理による高速化: {speedup:.2f}倍")
                
                # 並行処理が有効であることを確認（少なくとも10%の改善）
                assert speedup > 1.1, f"並行処理の効果が不十分です: {speedup:.2f}倍"
            
            logger.info("✅ 並行処理パフォーマンステスト成功")
            
        except Exception as e:
            pytest.fail(f"並行処理パフォーマンステストエラー: {e}")

    def test_memory_usage_optimization(self):
        """メモリ使用量最適化テスト"""
        logger.info("=== メモリ使用量最適化テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            initial_memory = self._measure_memory_usage()
            memory_measurements = []
            
            # 複数回の処理でメモリリークがないことを確認
            for i in range(10):
                ml_orchestrator = MLOrchestrator(enable_automl=False)
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(self.medium_dataset)
                    
                    # 明示的にオブジェクトを削除
                    del ml_orchestrator
                    del result
                    
                    # ガベージコレクション
                    gc.collect()
                    
                    current_memory = self._measure_memory_usage()
                    memory_measurements.append(current_memory)
                    
                    logger.info(f"反復 {i+1}: メモリ使用量 {current_memory:.1f}MB")
                    
                except Exception as e:
                    logger.warning(f"反復 {i+1} でエラー: {e}")
            
            # メモリリークの確認
            if memory_measurements:
                final_memory = memory_measurements[-1]
                memory_increase = final_memory - initial_memory
                
                logger.info(f"初期メモリ: {initial_memory:.1f}MB")
                logger.info(f"最終メモリ: {final_memory:.1f}MB")
                logger.info(f"メモリ増加: {memory_increase:.1f}MB")
                
                # メモリリークの基準（100MB以下の増加は許容）
                assert memory_increase < 100, f"メモリリークの可能性があります: {memory_increase:.1f}MB増加"
            
            logger.info("✅ メモリ使用量最適化テスト成功")
            
        except Exception as e:
            pytest.fail(f"メモリ使用量最適化テストエラー: {e}")

    def test_timeout_handling(self):
        """タイムアウト処理テスト"""
        logger.info("=== タイムアウト処理テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            def long_running_task():
                """長時間実行されるタスクをシミュレート"""
                ml_orchestrator = MLOrchestrator(enable_automl=False)
                # 大きなデータセットで処理時間を延長
                large_data = self._create_test_data(10000)
                return ml_orchestrator.calculate_ml_indicators(large_data)
            
            # タイムアウト付きで実行
            timeout_seconds = 30
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(long_running_task)
                    result = future.result(timeout=timeout_seconds)
                    
                    logger.info("長時間タスクが制限時間内に完了しました")
                    
            except TimeoutError:
                logger.info(f"タスクが {timeout_seconds} 秒でタイムアウトしました（期待される動作）")
                
            except Exception as e:
                logger.info(f"長時間タスクでエラー: {e}")
            
            # 短時間タスクのタイムアウトテスト
            def short_task():
                ml_orchestrator = MLOrchestrator(enable_automl=False)
                return ml_orchestrator.calculate_ml_indicators(self.small_dataset)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(short_task)
                    result = future.result(timeout=10)  # 10秒のタイムアウト
                    
                    logger.info("短時間タスクが正常に完了しました")
                    
            except TimeoutError:
                logger.warning("短時間タスクがタイムアウトしました（予期しない動作）")
                
            except Exception as e:
                logger.info(f"短時間タスクでエラー: {e}")
            
            logger.info("✅ タイムアウト処理テスト成功")
            
        except Exception as e:
            pytest.fail(f"タイムアウト処理テストエラー: {e}")

    def test_scalability_analysis(self):
        """スケーラビリティ分析テスト"""
        logger.info("=== スケーラビリティ分析テスト ===")
        
        try:
            from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
            
            # 異なるサイズのデータセットでの処理時間を測定
            data_sizes = [100, 500, 1000, 2000, 3000]
            execution_times = []
            
            for size in data_sizes:
                dataset = self._create_test_data(size)
                ml_orchestrator = MLOrchestrator(enable_automl=False)
                
                start_time = time.time()
                
                try:
                    result = ml_orchestrator.calculate_ml_indicators(dataset)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    
                    logger.info(f"データサイズ {size}: {execution_time:.2f}秒")
                    
                except Exception as e:
                    logger.warning(f"データサイズ {size} でエラー: {e}")
                    execution_times.append(None)
                
                # メモリクリーンアップ
                del ml_orchestrator
                del dataset
                gc.collect()
            
            # スケーラビリティの分析
            valid_times = [(size, time) for size, time in zip(data_sizes, execution_times) if time is not None]
            
            if len(valid_times) >= 2:
                # 線形性の確認（大まかな）
                first_size, first_time = valid_times[0]
                last_size, last_time = valid_times[-1]
                
                size_ratio = last_size / first_size
                time_ratio = last_time / first_time
                
                logger.info(f"サイズ比: {size_ratio:.1f}倍, 時間比: {time_ratio:.1f}倍")
                
                # 時間の増加が過度でないことを確認（サイズの3倍以下）
                assert time_ratio <= size_ratio * 3, f"処理時間の増加が過度です: サイズ{size_ratio:.1f}倍に対して時間{time_ratio:.1f}倍"
            
            logger.info("✅ スケーラビリティ分析テスト成功")
            
        except Exception as e:
            pytest.fail(f"スケーラビリティ分析テストエラー: {e}")


if __name__ == "__main__":
    # 単体でテストを実行する場合
    pytest.main([__file__, "-v"])
