"""
最適化機能のパフォーマンステスト

最適化処理の実行時間、メモリ使用量などのパフォーマンスを測定します。
"""

import pytest
import time
import psutil
import gc
from typing import Dict, Any
from unittest.mock import patch

from app.core.services.optimization.base_optimizer import ParameterSpace
from app.core.services.optimization.bayesian_optimizer import BayesianOptimizer
from app.core.services.optimization.grid_search_optimizer import GridSearchOptimizer
from app.core.services.optimization.random_search_optimizer import RandomSearchOptimizer
from app.core.services.optimization.optimizer_factory import OptimizerFactory


class TestOptimizationPerformance:
    """最適化機能のパフォーマンステスト"""

    def measure_performance(self, func, *args, **kwargs):
        """パフォーマンス測定ヘルパー"""
        # ガベージコレクションを実行
        gc.collect()
        
        # 開始時のメモリ使用量
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # 関数実行
        result = func(*args, **kwargs)
        
        # 終了時の測定
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return result, execution_time, memory_usage

    def test_bayesian_optimizer_performance(self):
        """ベイジアンオプティマイザーのパフォーマンステスト"""
        optimizer = BayesianOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            # 軽量な計算
            x = params["x"]
            y = params["y"]
            return -(x - 0.5) ** 2 - (y - 0.5) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0),
            "y": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        # パフォーマンス測定
        result, execution_time, memory_usage = self.measure_performance(
            optimizer.optimize,
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=20
        )
        
        # パフォーマンス要件の確認
        assert execution_time < 30.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 100.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        # 結果の妥当性確認
        assert result.total_evaluations == 20
        assert result.optimization_time > 0

    def test_grid_search_optimizer_performance(self):
        """グリッドサーチオプティマイザーのパフォーマンステスト"""
        optimizer = GridSearchOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            return -(x - 2) ** 2 - (y - 2) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="integer", low=0, high=4),
            "y": ParameterSpace(type="integer", low=0, high=4)
        }
        
        # パフォーマンス測定
        result, execution_time, memory_usage = self.measure_performance(
            optimizer.optimize,
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=50  # 最大組み合わせ数制限
        )
        
        # パフォーマンス要件の確認
        assert execution_time < 10.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 50.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        # 結果の妥当性確認
        assert result.total_evaluations <= 25  # 5x5グリッド

    def test_random_search_optimizer_performance(self):
        """ランダムサーチオプティマイザーのパフォーマンステスト"""
        optimizer = RandomSearchOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            z = params["z"]
            return -(x - 0.3) ** 2 - (y - 0.7) ** 2 - (z - 50) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0),
            "y": ParameterSpace(type="real", low=0.0, high=1.0),
            "z": ParameterSpace(type="integer", low=0, high=100)
        }
        
        # パフォーマンス測定
        result, execution_time, memory_usage = self.measure_performance(
            optimizer.optimize,
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=50
        )
        
        # パフォーマンス要件の確認
        assert execution_time < 5.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 30.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        # 結果の妥当性確認
        assert result.total_evaluations <= 50

    def test_optimizer_factory_performance(self):
        """OptimizerFactoryのパフォーマンステスト"""
        # 複数のオプティマイザー作成のパフォーマンス
        start_time = time.time()
        
        optimizers = []
        for method in ["bayesian", "grid", "random"] * 10:  # 30回作成
            optimizer = OptimizerFactory.create_optimizer(method)
            optimizers.append(optimizer)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # パフォーマンス要件の確認
        assert creation_time < 1.0, f"オプティマイザー作成時間が長すぎます: {creation_time:.2f}秒"
        assert len(optimizers) == 30

    def test_large_parameter_space_performance(self):
        """大きなパラメータ空間でのパフォーマンステスト"""
        optimizer = RandomSearchOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            # 多数のパラメータを使用
            total = 0
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    total += value
            return -abs(total - 50)  # 合計が50に近いほど高スコア
        
        # 10個のパラメータを持つ空間
        parameter_space = {}
        for i in range(10):
            parameter_space[f"param_{i}"] = ParameterSpace(type="real", low=0.0, high=10.0)
        
        # パフォーマンス測定
        result, execution_time, memory_usage = self.measure_performance(
            optimizer.optimize,
            objective_function=objective_function,
            parameter_space=parameter_space,
            n_calls=30
        )
        
        # パフォーマンス要件の確認
        assert execution_time < 15.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 80.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        
        # 結果の妥当性確認
        assert len(result.best_params) == 10

    def test_expensive_objective_function_performance(self):
        """計算コストの高い目的関数でのパフォーマンステスト"""
        optimizer = BayesianOptimizer()
        
        def expensive_objective_function(params: Dict[str, Any]) -> float:
            # 意図的に重い計算をシミュレート
            x = params["x"]
            result = 0
            for i in range(1000):  # 重い計算
                result += (x - 0.5) ** 2 * i
            return -result / 1000
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        # パフォーマンス測定（少ない試行回数で）
        result, execution_time, memory_usage = self.measure_performance(
            optimizer.optimize,
            objective_function=expensive_objective_function,
            parameter_space=parameter_space,
            n_calls=5  # 少ない試行回数
        )
        
        # パフォーマンス要件の確認（重い計算を考慮）
        assert execution_time < 60.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 150.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"

    def test_concurrent_optimization_performance(self):
        """並行最適化のパフォーマンステスト"""
        import threading
        import queue
        
        def run_optimization(result_queue):
            optimizer = RandomSearchOptimizer()
            
            def objective_function(params: Dict[str, Any]) -> float:
                x = params["x"]
                return -(x - 0.5) ** 2
            
            parameter_space = {
                "x": ParameterSpace(type="real", low=0.0, high=1.0)
            }
            
            start_time = time.time()
            result = optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=10
            )
            end_time = time.time()
            
            result_queue.put((result, end_time - start_time))
        
        # 3つの並行最適化を実行
        threads = []
        result_queue = queue.Queue()
        
        start_time = time.time()
        
        for _ in range(3):
            thread = threading.Thread(target=run_optimization, args=(result_queue,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 結果の収集
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # パフォーマンス要件の確認
        assert len(results) == 3
        assert total_time < 30.0, f"並行実行時間が長すぎます: {total_time:.2f}秒"
        
        # 各最適化が正常に完了していることを確認
        for result, execution_time in results:
            assert result.total_evaluations == 10
            assert execution_time > 0

    def test_memory_leak_detection(self):
        """メモリリークの検出テスト"""
        optimizer = RandomSearchOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            x = params["x"]
            return -(x - 0.5) ** 2
        
        parameter_space = {
            "x": ParameterSpace(type="real", low=0.0, high=1.0)
        }
        
        # 初期メモリ使用量
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 複数回最適化を実行
        for _ in range(10):
            result = optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=5
            )
            # 結果を明示的に削除
            del result
        
        # 最終メモリ使用量
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリリークの確認（50MB以下の増加は許容）
        assert memory_increase < 50.0, f"メモリリークの可能性: {memory_increase:.2f}MB増加"

    def test_optimization_scalability(self):
        """最適化のスケーラビリティテスト"""
        optimizer = GridSearchOptimizer()
        
        def objective_function(params: Dict[str, Any]) -> float:
            return sum(params.values())
        
        # 異なるサイズのパラメータ空間でのパフォーマンス測定
        performance_data = []
        
        for n_params in [1, 2, 3]:
            parameter_space = {}
            for i in range(n_params):
                parameter_space[f"param_{i}"] = ParameterSpace(type="integer", low=0, high=2)
            
            start_time = time.time()
            result = optimizer.optimize(
                objective_function=objective_function,
                parameter_space=parameter_space,
                n_calls=50  # 最大組み合わせ数制限
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            performance_data.append((n_params, execution_time, result.total_evaluations))
        
        # スケーラビリティの確認
        for n_params, execution_time, evaluations in performance_data:
            assert execution_time < 10.0, f"パラメータ数{n_params}で実行時間が長すぎます: {execution_time:.2f}秒"
            assert evaluations <= 3 ** n_params, f"評価回数が期待値を超えています: {evaluations}"
