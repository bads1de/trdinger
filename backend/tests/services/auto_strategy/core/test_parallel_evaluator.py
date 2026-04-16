"""
ParallelEvaluatorのユニットテスト
"""

import time
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
    ParallelEvaluationResult,
    ParallelEvaluator,
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

    def test_evaluate_population_caches_behavior_summary_from_rich_result(self):
        """拡張返り値から behavior summary を回収できること"""
        individual = MagicMock()
        individual.id = "ind_behavior"

        evaluator = ParallelEvaluator(
            evaluate_func=lambda _: ParallelEvaluationResult(
                fitness=(1.25,),
                behavior_summary={"pass_rate": 0.4, "mean_total_return": 0.12},
            ),
            max_workers=1,
        )

        result = evaluator.evaluate_population([individual])

        assert result == [(1.25,)]
        assert evaluator.get_cached_behavior_profile(individual) == {
            "pass_rate": 0.4,
            "mean_total_return": 0.12,
        }

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

    def test_evaluate_population_times_out_per_individual(self):
        """個体ごとの timeout を超えたタスクは default_fitness になること"""
        individuals = [MagicMock() for _ in range(5)]
        delays = [0.12, 0.01, 0.01, 0.01, 0.01]

        for i, (ind, delay) in enumerate(zip(individuals, delays)):
            ind.id = f"ind_{i}"
            ind.delay = delay
            ind.fitness_value = (float(i),)

        def timed_evaluate(ind):
            time.sleep(ind.delay)
            return ind.fitness_value

        evaluator = ParallelEvaluator(
            evaluate_func=timed_evaluate,
            max_workers=len(individuals),
            timeout_per_individual=0.05,
        )

        result = evaluator.evaluate_population(individuals, default_fitness=(-1.0,))

        assert result[0] == (-1.0,)
        assert result[1:] == [(1.0,), (2.0,), (3.0,), (4.0,)]

        stats = evaluator.get_statistics()
        assert stats["timeout_evaluations"] == 1
        assert stats["successful_evaluations"] == 4

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

    def test_reset_generation_stats_clears_behavior_summary_cache(self):
        """世代リセット時に behavior summary cache も破棄すること"""
        individual = MagicMock()
        individual.id = "ind_behavior_reset"

        evaluator = ParallelEvaluator(
            evaluate_func=lambda _: ParallelEvaluationResult(
                fitness=(1.0,),
                behavior_summary={"pass_rate": 0.5},
            ),
            max_workers=1,
        )

        evaluator.evaluate_population([individual])
        assert evaluator.get_cached_behavior_profile(individual) == {"pass_rate": 0.5}

        evaluator._reset_generation_stats()

        assert evaluator.get_cached_behavior_profile(individual) is None

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

    def test_evaluate_with_process_pool_initializer(self):
        """プロセスプールでカスタム初期化子が実行されること"""
        # Note: This test mocks ProcessPoolExecutor to verify initialization logic
        # because actual multiprocessing is hard to test reliably in unit tests environment.

        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance

        # モックのsubmitはFutureを返す必要がある
        mock_future = MagicMock()
        mock_executor_instance.submit.return_value = mock_future
        mock_future.result.return_value = (1.0,)

        with patch(
            "app.services.auto_strategy.core.evaluation.parallel_evaluator.ProcessPoolExecutor",
            return_value=mock_executor_instance,
        ) as mock_executor_cls:
            with patch(
                "app.services.auto_strategy.core.evaluation.parallel_evaluator.wait",
                return_value=({mock_future}, set()),
            ):
                initializer_mock = MagicMock()
                init_args = ("arg1", 123)

                evaluator = ParallelEvaluator(
                    evaluate_func=lambda x: (1.0,),
                    use_process_pool=True,
                    worker_initializer=initializer_mock,
                    worker_initargs=init_args,
                )

                individuals = [MagicMock()]
                evaluator.evaluate_population(individuals)

                # ProcessPoolExecutorが正しい引数で初期化されたか確認
                mock_executor_cls.assert_called_once()
                call_kwargs = mock_executor_cls.call_args[1]
                assert "initializer" in call_kwargs
                assert "initargs" in call_kwargs

                # 内部ラッパー関数を取得
                wrapper_initializer = call_kwargs["initializer"]
                wrapper_initargs = call_kwargs["initargs"]

                # wrapper_initargs = (self.worker_initializer, self.worker_initargs)
                assert wrapper_initargs[0] == initializer_mock
                assert wrapper_initargs[1] == init_args

                # ラッパー関数を実行して動作確認
                wrapper_initializer(initializer_mock, init_args)

                # ユーザー指定のinitializerが呼ばれたか
                initializer_mock.assert_called_once_with("arg1", 123)

    def test_persistent_executor_lifecycle(self):
        """永続的なExecutorのライフサイクル管理ができること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
            max_workers=2,
        )
        assert evaluator._executor is None

        evaluator.start()
        assert evaluator._executor is not None

        # 2回呼んでも問題ない（同じインスタンスが維持される）
        old_executor = evaluator._executor
        evaluator.start()
        assert evaluator._executor is old_executor

        evaluator.shutdown()
        assert evaluator._executor is None

    def test_context_manager(self):
        """コンテキストマネージャとして使用できること"""
        evaluator = ParallelEvaluator(
            evaluate_func=lambda x: (1.0,),
            max_workers=2,
        )
        assert evaluator._executor is None

        with evaluator:
            assert evaluator._executor is not None
            # 評価実行
            mock_ind = MagicMock()
            mock_ind.id = "test"
            result = evaluator.evaluate_population([mock_ind])
            assert len(result) == 1

        assert evaluator._executor is None
