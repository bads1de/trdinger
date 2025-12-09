"""
ParallelEvaluatorのユニットテスト
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.services.auto_strategy.core.parallel_evaluator import (
    ParallelEvaluator,
    create_parallel_map,
)


class TestParallelEvaluator:
    """ParallelEvaluatorのテストクラス"""

    def test_init_default_workers(self):
        """デフォルトのワーカー数で初期化できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
        )
        assert evaluator.max_workers >= 1
        assert evaluator.timeout_per_individual == 300.0

    def test_init_custom_workers(self):
        """カスタムワーカー数で初期化できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
            max_workers=4,
            timeout_per_individual=60.0,
        )
        assert evaluator.max_workers == 4
        assert evaluator.timeout_per_individual == 60.0

    def test_evaluate_population_empty(self):
        """空のpopulationを評価できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
        )
        result = evaluator.evaluate_population([])
        assert result == []

    def test_evaluate_population_single(self):
        """単一の個体を評価できること"""
        mock_individual = MagicMock()
        mock_individual.id = "test_id"

        def mock_evaluate(ind):
            return (0.5,)

        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate,
            max_workers=1,
        )
        result = evaluator.evaluate_population([mock_individual])
        assert len(result) == 1
        assert result[0] == (0.5,)

    def test_evaluate_population_multiple(self):
        """複数の個体を並列評価できること"""
        individuals = [MagicMock() for _ in range(5)]
        for i, ind in enumerate(individuals):
            ind.id = f"ind_{i}"

        def mock_evaluate(ind):
            return (float(individuals.index(ind)),)

        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate,
            max_workers=2,
        )
        result = evaluator.evaluate_population(individuals)
        assert len(result) == 5
        # 順序が保持されていること
        for i, fitness in enumerate(result):
            assert fitness == (float(i),)

    def test_evaluate_population_with_error(self):
        """評価中のエラーが適切に処理されること"""
        individuals = [MagicMock() for _ in range(3)]
        for i, ind in enumerate(individuals):
            ind.id = f"ind_{i}"

        def mock_evaluate(ind):
            if ind.id == "ind_1":
                raise ValueError("Test error")
            return (1.0,)

        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate,
            max_workers=2,
        )
        result = evaluator.evaluate_population(individuals, default_fitness=(0.0,))
        assert len(result) == 3
        assert result[0] == (1.0,)
        assert result[1] == (0.0,)  # エラー時はデフォルト値
        assert result[2] == (1.0,)

    def test_evaluate_parallel_faster_than_sequential(self):
        """並列評価がシーケンシャル評価より速いこと"""
        individuals = [MagicMock() for _ in range(4)]

        def slow_evaluate(ind):
            time.sleep(0.1)  # 100ms の遅延
            return (1.0,)

        # 並列評価
        evaluator = ParallelEvaluator(
            evaluate_func=slow_evaluate,
            max_workers=4,
        )
        start = time.time()
        evaluator.evaluate_population(individuals)
        parallel_time = time.time() - start

        # シーケンシャル評価
        start = time.time()
        for ind in individuals:
            slow_evaluate(ind)
        sequential_time = time.time() - start

        # 並列評価は少なくとも2倍速いはず
        assert parallel_time < sequential_time * 0.75

    def test_get_statistics(self):
        """統計情報が正しく取得できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
            max_workers=1,
        )

        # 初期状態
        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 0
        assert stats["successful_evaluations"] == 0

        # 評価後
        individuals = [MagicMock() for _ in range(3)]
        evaluator.evaluate_population(individuals)

        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 3
        assert stats["successful_evaluations"] == 3
        assert stats["success_rate"] == 1.0

    def test_reset_statistics(self):
        """統計情報がリセットできること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
            max_workers=1,
        )

        individuals = [MagicMock() for _ in range(3)]
        evaluator.evaluate_population(individuals)

        evaluator.reset_statistics()
        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 0

    def test_evaluate_invalid_individuals(self):
        """適応度が無効な個体のみを評価できること"""

        def mock_evaluate(ind):
            return (1.0,)

        evaluator = ParallelEvaluator(
            evaluate_func=mock_evaluate,
            max_workers=1,
        )

        # フィットネスが無効な個体を作成
        individuals = []
        for i in range(3):
            ind = MagicMock()
            ind.fitness = MagicMock()
            ind.fitness.valid = i != 1  # ind_1 のみ無効
            individuals.append(ind)

        result = evaluator.evaluate_invalid_individuals(individuals)
        assert len(result) == 1  # 無効な個体は1つだけ
        assert result[0][0] == (1.0,)  # フィットネス値
        assert result[0][1] == individuals[1]  # 無効だった個体


class TestCreateParallelMap:
    """create_parallel_map関数のテスト"""

    def test_create_and_use_parallel_map(self):
        """並列map関数を作成して使用できること"""

        def mock_evaluate(ind):
            return (float(ind),)

        parallel_map = create_parallel_map(
            evaluate_func=mock_evaluate,
            max_workers=2,
        )

        individuals = [1, 2, 3, 4, 5]
        result = parallel_map(None, individuals)  # funcは無視される

        assert len(result) == 5
        for i, fitness in enumerate(result):
            assert fitness == (float(i + 1),)
