"""
長期実行シナリオでのリソースリークテスト

このテストスイートは、長時間実行されるシナリオでのリソースリークを検出します。
特にGA最適化の長時間実行、バックテストの連続実行、イベントループのリークなどをテストします。
"""

import pytest
import gc
import time
import psutil
import threading
import weakref
from functools import wraps
from unittest.mock import patch, MagicMock
import asyncio


class TestLongRunningResourceLeak:
    """長期実行リソースリークバグ検出テスト"""

    def test_ga_engine_long_run_memory_leak(self):
        """GAエンジンの長時間実行でメモリリークが発生するテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
        from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        from app.services.auto_strategy.config import GAConfig

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 長時間実行設定（小規模デモ用に調整）
        config = GAConfig.from_dict({
            "population_size": 20,
            "generations": 10,  # 通常より少ない
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "max_indicators": 5,
        })

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.1}}

        factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(config)
        engine = GeneticAlgorithmEngine(mock_backtest, gene_generator, factory)

        try:
            # 長時間実行をシミュレート
            result = engine.run_evolution(config, {"symbol": "BTC/USDT"})

            # メモリチェック
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # 許容範囲を超えるメモリ増加はリークの兆候
            if memory_increase > 50:  # 50MB以上増加
                pytest.fail(f"メモリリーク検出: GA実行で {memory_increase:.1f}MB 増加")

            # ガベージコレクション後もリークを確認
            collected = gc.collect()
            post_gc_memory = process.memory_info().rss / 1024 / 1024
            gc_memory_increase = post_gc_memory - initial_memory

            if gc_memory_increase > 30:  # GC後も30MB以上増加
                pytest.fail(f"メモリリーク検出: GA実行でGC後も {gc_memory_increase:.1f}MB 増加")

        except Exception as e:
            pytest.fail(f"GA実行中の異常: {e}")

    def test_event_loop_resource_leak_in_long_running_watches(self):
        """長時間実行監視でのイベントループリークテスト"""

        class LongRunningWatcher:
            def __init__(self):
                self.watchers = []
                self._stopped = False

            def add_watcher(self, callback):
                watcher_ref = weakref.ref(callback, self._cleanup_watcher)
                self.watchers.append(watcher_ref)

            def _cleanup_watcher(self, weak_ref):
                try:
                    self.watchers.remove(weak_ref)
                except ValueError:
                    pass  # 既に削除されている

            async def run_long_watching_session(self, duration_hours=2):
                """長時間監視セッション（シミュレーション）"""
                start_time = time.time()
                check_interval = 1.0  # 1秒間隔

                while time.time() - start_time < duration_hours * 3600 and not self._stopped:
                    # 定期チェックをシミュレート
                    for i, watcher_ref in enumerate(self.watchers):
                        watcher = watcher_ref()
                        if watcher:
                            # メモリリークの原因: コールバック参照を保持
                            self._process_watcher_result(watcher())
                        else:
                            # 弱参照が死んだwatcherをクリーンアップ
                            self.watchers.pop(i)

                    await asyncio.sleep(check_interval)

                    # 長時間の兆候としてオブジェクト数をチェック
                    if len(gc.get_objects()) > 10000:  # オブジェクト数が異常に多い
                        break

            def _process_watcher_result(self, result):
                """監視結果処理（メモリリークの原因）"""
                # バグ: 結果を蓄積
                if not hasattr(self, '_results_cache'):
                    self._results_cache = []
                self._results_cache.append(result)

                # 無制限に蓄積（リーク）
                if len(self._results_cache) > 1000:
                    self._results_cache = self._results_cache[-500:]  # 一部削除だが不十分

            def stop(self):
                self._stopped = True

        watcher = LongRunningWatcher()

        # watcherコールバックを追加
        def sample_watcher():
            return {"timestamp": time.time(), "data": "sample_data"}

        watcher.add_watcher(sample_watcher)

        # 長時間実行前のメモリベースライン
        initial_objects = len(gc.get_objects())
        initial_memory = psutil.Process().memory_info().rss

        async def run_test():
            try:
                # シミュレートされた長時間実行（実際には短く）
                await watcher.run_long_watching_session(duration_hours=0.01)  # 数秒
            finally:
                watcher.stop()

        # 非同期実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

        # 長時間実行後のリークチェック
        final_objects = len(gc.get_objects())
        final_memory = psutil.Process().memory_info().rss

        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        object_increase = final_objects - initial_objects

        # 過剰な増加はリーク
        if object_increase > 1000 or memory_increase > 10:
            pytest.fail(f"リソースリーク検出: オブジェクト数 +{object_increase}, メモリ +{memory_increase:.1f}MB")

    def test_thread_pool_resource_leak_in_parallel_backtest(self):
        """並列バックテストでのスレッドプールリークテスト"""

        class ParallelBacktestEngine:
            def __init__(self, max_workers=4):
                self.max_workers = max_workers
                self.active_threads = set()
                self.completed_results = []

            def run_parallel_backtests(self, test_configs, duration_minutes=30):
                """並列バックテスト実行（リークリスク）"""
                from concurrent.futures import ThreadPoolExecutor
                import threading

                def backtest_worker(config):
                    """バックテストワーカー（長時間実行）"""
                    thread_id = threading.get_ident()
                    self.active_threads.add(thread_id)

                    try:
                        start_time = time.time()
                        while time.time() - start_time < duration_minutes * 60:
                            # ダミー処理（短く）
                            time.sleep(0.01)
                            result = {"config": config, "return": 0.1}

                            # バグ: 結果を無制限に蓄積
                            self.completed_results.append(result)

                            # リソースリークのシミュレーション
                            if len(self.completed_results) > 10000:
                                break

                    finally:
                        self.active_threads.discard(thread_id)

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(backtest_worker, config) for config in test_configs]
                    for future in futures:
                        try:
                            future.result(timeout=60)  # 1分タイムアウト
                        except Exception as e:
                            print(f"バックテスト失敗: {e}")

        engine = ParallelBacktestEngine()
        test_configs = [{"symbol": f"SYMBOL_{i}", "timeframe": "1h"} for i in range(10)]

        # 実行前のベースライン
        initial_thread_count = threading.active_count()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # 並列実行（短時間シミュレーション）
        engine.run_parallel_backtests(test_configs, duration_minutes=0.1)  # 数秒

        # 実行後のチェック
        final_thread_count = threading.active_count()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        memory_increase = final_memory - initial_memory
        thread_increase = final_thread_count - initial_thread_count

        # リーク検出
        if thread_increase > 2 or memory_increase > 20:
            pytest.fail(f"リソースリーク検出: スレッド数 +{thread_increase}, メモリ +{memory_increase:.1f}MB")

        # スレッドクリーンアップ確認
        if len(engine.active_threads) > 0:
            pytest.fail(f"スレッドリーク検出: {len(engine.active_threads)}個のスレッドがアクティブのまま")

    def test_context_manager_resource_leak_in_iterative_optimization(self):
        """反復最適化でのコンテキストマネージャーリークテスト"""

        class OptimizationContext:
            def __init__(self):
                self.iterations = []
                self._resource_pool = []  # リークの原因

            @wraps(lambda: None)
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # バグ: リソースプールを完全クリーンアップしない
                if len(self._resource_pool) > 100:  # 一部だけクリーンアップ
                    self._resource_pool = self._resource_pool[:50]
                # ガベージコレクションを呼ばない

            def add_iteration_resource(self, resource):
                """反復毎のリソース追加"""
                self._resource_pool.append(resource)
                self.iterations.append(resource)

            def get_resource_count(self):
                return len(self._resource_pool)

        def iterative_optimization(iterations=100):
            """反復最適化（リソースリークの可能性）"""
            with OptimizationContext() as context:
                for i in range(iterations):
                    # 大きなデータを追加してリークをシミュレート
                    large_data = {"data": [0] * 1000, "metadata": "iteration_" + str(i)}
                    context.add_iteration_resource(large_data)

                    # 短い処理時間
                    time.sleep(0.001)

                return context

        # 実行前のベースライン
        initial_objects = len(gc.get_objects())

        # 反復最適化実行
        context = iterative_optimization(iterations=50)

        # 実行後のチェック
        final_objects = len(gc.get_objects())
        resource_count = context.get_resource_count()

        # 強制GC
        collected = gc.collect()
        post_gc_objects = len(gc.get_objects())

        # リーク検出
        net_object_increase = post_gc_objects - initial_objects
        retained_resource_count = context.get_resource_count()

        if net_object_increase > 1000 or retained_resource_count > 20:
            pytest.fail(f"リソースリーク検出: オブジェクト増加 {net_object_increase}, 保持リソース {retained_resource_count}")

    @pytest.mark.asyncio
    async def test_async_event_loop_memory_leak_in_continuous_monitoring(self):
        """連続監視での非同期イベントループメモリリークテスト"""

        async def create_monitoring_task():
            """監視タスク（リークの原因となる）"""
            monitoring_data = []

            try:
                for i in range(100):  # 連続監視
                    # データ蓄積（リーク）
                    monitoring_data.append({
                        "timestamp": time.time(),
                        "metrics": {"cpu": psutil.cpu_percent(), "memory": psutil.virtual_memory().percent},
                        "index": i
                    })

                    # タスク完了を待つ
                    await asyncio.sleep(0.01)

                    # メモリ使用量チェック
                    if psutil.virtual_memory().percent > 80:
                        break

            except Exception as e:
                await asyncio.sleep(0.1)  # エラー後のクリーンアップ待機

            return monitoring_data, len(monitoring_data)

        # 複数タスクの並列実行
        tasks = [create_monitoring_task() for _ in range(5)]

        # 実行前のメモリチェック
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            # タスク実行
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 結果処理
            total_data_points = sum(len(data) for data, count in results if isinstance(data, list) else 0)

        except Exception as e:
            pytest.fail(f"非同期監視タスクでエラー: {e}")

        # 実行後のメモリチェック
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # ガベージコレクション実行
        collected = gc.collect()
        post_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
        gc_memory_increase = post_gc_memory - initial_memory

        # リーク判定
        if gc_memory_increase > 20 or total_data_points > 300:
            pytest.fail(f"非同期メモリリーク検出: GC後メモリ増加 {gc_memory_increase:.1f}MB, データポイント {total_data_points}")

    def test_weak_reference_cleanup_failure_in_long_session(self):
        """長時間セッションでの弱参照クリーンアップ失敗テスト"""

        class SessionManager:
            def __init__(self):
                self.active_sessions = []
                self._weak_refs = []

            def create_session(self, session_id):
                """セッション作成（弱参照管理）"""
                session_obj = {"session_id": session_id, "data": []}

                # 弱参照作成
                weak_ref = weakref.ref(session_obj, self._cleanup_session)
                self._weak_refs.append(weak_ref)

                self.active_sessions.append(session_obj)
                return session_obj

            def _cleanup_session(self, weak_ref):
                """セッションクリーンアップ（バグのある実装）"""
                try:
                    self._weak_refs.remove(weak_ref)
                except ValueError:
                    pass  # リストに見つからない場合

            def simulate_long_session_usage(self, hours=1):
                """長時間セッション使用をシミュレート"""
                start_time = time.time()

                while time.time() - start_time < hours * 3600:
                    # 新規セッション作成と破棄をシミュレート
                    session = self.create_session(f"session_{len(self.active_sessions)}")

                    # 短い使用
                    session["data"].append("usage_data")
                    time.sleep(0.01)

                    # セッション破棄（弱参照クリーンアップが発生すべき）
                    del session

                    # メモリチェック
                    if len(gc.get_objects()) > 10000:
                        break

                    # 短くするために100回のループで終了
                    if len(self.active_sessions) > 100:
                        break

        manager = SessionManager()

        # 実行前のベースライン
        initial_weak_refs = len(manager._weak_refs)

        # 長時間実行
        manager.simulate_long_session_usage(hours=0.001)  # 数秒

        # ガベージコレクション実行
        gc.collect()

        # 実行後のチェック
        final_weak_refs = len(manager._weak_refs)
        retained_objects = len(manager.active_sessions)

        # 弱参照がクリーンアップされていない場合の判定
        if final_weak_refs > initial_weak_refs + 10 or retained_objects > 50:
            pytest.fail(f"弱参照リーク検出: 弱参照増加 +{final_weak_refs - initial_weak_refs}, 保持オブジェクト {retained_objects}")


class TestResourceMonitoringToolkit:
    """リソース監視ツールキットテスト"""

    def test_memory_profiler_detected_leaks_in_ga_population(self):
        """GA個体群でのメモリプロファイラリーク検出テスト"""

        @profile
        def memory_intensive_population_generation(population_size=100):
            """メモリ集約的な個体群生成"""
            population = []

            for i in range(population_size):
                # 大きな個体生成
                individual = {
                    "genes": [0.1] * 1000,  # 遺伝子データ
                    "fitness": 0.0,
                    "metadata": {"generation": i, "data": "large_string" * 100}
                }
                population.append(individual)

            return population

        try:
            from memory_profiler import profile
        except ImportError:
            pytest.skip("memory_profiler not available")

        # メモリ使用量測定前のベースライン
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # メモリプロファイリング実行
        population = memory_intensive_population_generation()

        # 実行後チェック
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        population_size = len(population)

        # リーク判定
        if memory_increase > 100:  # 100MB以上増加
            pytest.fail(f"メモリリーク検出: 個体群生成で {memory_increase:.1f}MB 増加")

        if population_size != 100:
            pytest.fail("個体群サイズ異常")

    def test_cpu_resource_contention_in_parallel_optimization(self):
        """並列最適化でのCPUリソース競合テスト"""

        def cpu_intensive_optimization(task_id, duration_sec=1):
            """CPU集約的な最適化タスク"""
            start_time = time.time()
            calculations = 0

            while time.time() - start_time < duration_sec:
                # CPU負荷計算
                result = sum(i * i for i in range(10000))
                calculations += 1

            return {"task_id": task_id, "calculations": calculations}

        initial_cpu_percent = psutil.cpu_percent(interval=1)

        # 並列実行
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_optimization, i, 0.5) for i in range(8)]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"CPU最適化タスク失敗: {e}")

        final_cpu_percent = psutil.cpu_percent(interval=1)

        # CPU負荷が異常に高い場合の判定
        if final_cpu_percent > 90:  # 90%以上
            pytest.fail(f"CPU競合検出: 最終CPU使用率 {final_cpu_percent}%")

        if len(results) != 8:
            pytest.fail("並列最適化タスク完了数異常")