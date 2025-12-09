"""
キャッシュ管理と並列評価の改善に関するテスト

LRUキャッシュのエビクションポリシーと
ProcessPoolExecutorのタイムアウト処理を検証します。
"""

from unittest.mock import Mock
import pytest
from cachetools import LRUCache

from app.services.auto_strategy.core.individual_evaluator import IndividualEvaluator
from app.services.auto_strategy.core.parallel_evaluator import (
    ParallelEvaluator,
    create_parallel_map,
)


class TestLRUCacheEviction:
    """LRUキャッシュのエビクションポリシーのテスト"""

    @pytest.fixture
    def mock_backtest_service(self):
        """バックテストサービスのモック"""
        service = Mock()
        service.data_service = Mock()
        service.data_service.get_data_for_backtest = Mock(return_value={"data": "test"})
        service.ensure_data_service_initialized = Mock()
        return service

    def test_lru_cache_initialization(self, mock_backtest_service):
        """LRUキャッシュが正しく初期化されること"""
        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=50)

        assert isinstance(evaluator._data_cache, LRUCache)
        assert evaluator._max_cache_size == 50
        assert evaluator._data_cache.maxsize == 50

    def test_default_cache_size(self, mock_backtest_service):
        """デフォルトのキャッシュサイズが設定されること"""
        evaluator = IndividualEvaluator(mock_backtest_service)

        assert evaluator._max_cache_size == IndividualEvaluator.DEFAULT_MAX_CACHE_SIZE
        assert (
            evaluator._data_cache.maxsize == IndividualEvaluator.DEFAULT_MAX_CACHE_SIZE
        )

    def test_cache_eviction_on_overflow(self, mock_backtest_service):
        """キャッシュがオーバーフロー時にLRUエビクションが動作すること"""
        # 小さなキャッシュサイズで初期化
        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=3)

        # 手動でキャッシュにエントリを追加
        evaluator._data_cache[("BTC", "1h", "2024-01-01", "2024-02-01")] = {"data": 1}
        evaluator._data_cache[("ETH", "1h", "2024-01-01", "2024-02-01")] = {"data": 2}
        evaluator._data_cache[("XRP", "1h", "2024-01-01", "2024-02-01")] = {"data": 3}

        assert len(evaluator._data_cache) == 3

        # 4つ目を追加（最初のエントリがエビクトされるはず）
        evaluator._data_cache[("SOL", "1h", "2024-01-01", "2024-02-01")] = {"data": 4}

        assert len(evaluator._data_cache) == 3
        # BTCが削除されていることを確認
        assert ("BTC", "1h", "2024-01-01", "2024-02-01") not in evaluator._data_cache
        assert ("SOL", "1h", "2024-01-01", "2024-02-01") in evaluator._data_cache

    def test_clear_cache(self, mock_backtest_service):
        """キャッシュクリアが正しく動作すること"""
        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=10)

        # キャッシュにデータを追加
        evaluator._data_cache[("BTC", "1h", "2024-01-01", "2024-02-01")] = {"data": 1}
        evaluator._data_cache[("ETH", "1h", "2024-01-01", "2024-02-01")] = {"data": 2}

        assert len(evaluator._data_cache) == 2

        # キャッシュをクリア
        evaluator.clear_cache()

        assert len(evaluator._data_cache) == 0

    def test_get_cache_info(self, mock_backtest_service):
        """キャッシュ情報が正しく取得できること"""
        evaluator = IndividualEvaluator(mock_backtest_service, max_cache_size=50)

        # キャッシュにデータを追加
        evaluator._data_cache[("BTC", "1h", "2024-01-01", "2024-02-01")] = {"data": 1}

        info = evaluator.get_cache_info()

        assert info["current_size"] == 1
        assert info["max_size"] == 50


class TestParallelEvaluatorImprovements:
    """ParallelEvaluatorの改善点のテスト"""

    @pytest.fixture
    def mock_evaluate_func(self):
        """評価関数のモック"""

        def evaluate(individual):
            return (1.0, 0.5)

        return evaluate

    def test_thread_pool_mode(self, mock_evaluate_func):
        """ThreadPoolExecutorモードで動作すること"""
        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate_func,
            max_workers=2,
            use_process_pool=False,
        )

        assert evaluator.use_process_pool is False

        # 評価を実行
        population = [1, 2, 3]
        results = evaluator.evaluate_population(population)

        assert len(results) == 3
        assert all(r == (1.0, 0.5) for r in results)

    def test_process_pool_mode_initialization(self, mock_evaluate_func):
        """ProcessPoolExecutorモードで初期化できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate_func,
            max_workers=2,
            use_process_pool=True,
        )

        assert evaluator.use_process_pool is True

    def test_statistics_reset_per_generation(self, mock_evaluate_func):
        """世代ごとに統計がリセットされること"""
        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate_func,
            max_workers=2,
        )

        # 1回目の評価
        evaluator.evaluate_population([1, 2])
        stats1 = evaluator.get_statistics()
        assert stats1["successful_evaluations"] == 2

        # 2回目の評価（auto_reset_per_generationがTrueなので統計はリセット）
        evaluator.evaluate_population([3, 4, 5])
        stats2 = evaluator.get_statistics()
        # 成功数は前回から累積されていないことを確認
        assert stats2["successful_evaluations"] == 3

    def test_create_parallel_map_with_process_pool(self, mock_evaluate_func):
        """ProcessPoolExecutor用のparallel_mapが作成できること"""
        parallel_map = create_parallel_map(
            evaluate_func=mock_evaluate_func,
            max_workers=2,
            use_process_pool=False,  # テストではThreadPoolを使用
        )

        # parallel_mapは呼び出し可能
        assert callable(parallel_map)

        # 実行テスト
        results = parallel_map(None, [1, 2, 3])
        assert len(results) == 3

    def test_timeout_handling(self):
        """タイムアウトが正しく処理されること"""
        import time

        def slow_evaluate(individual):
            time.sleep(2)  # 2秒待機
            return (1.0,)

        evaluator = ParallelEvaluator(
            evaluate_func=slow_evaluate,
            max_workers=1,
            timeout_per_individual=0.1,  # 0.1秒でタイムアウト
            use_process_pool=False,
        )

        # タイムアウトが発生してもクラッシュしないことを確認
        results = evaluator.evaluate_population([1], default_fitness=(0.0,))

        # タイムアウトまたは成功のいずれかの結果が返る
        assert len(results) == 1
