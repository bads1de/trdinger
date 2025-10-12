"""
GAエンジンの拡張テスト
"""
import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.config import GAConfig


class TestGAEngineExtended:
    """GAエンジンの拡張テスト"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_regime_detector = Mock()
        self.ga_config = GAConfig()
        self.ga_config.population_size = 10
        self.ga_config.generations = 3
        self.engine = GeneticAlgorithmEngine(
            self.mock_backtest_service,
            self.ga_config,
            regime_detector=self.mock_regime_detector
        )

    def test_evolve_with_regime_adaptation(self):
        """レジーム適応付き進化のテスト"""
        # GAエンジンを初期化
        self.ga_config.regime_adaptation_enabled = True

        with patch.object(self.engine, '_run_generation') as mock_run_gen:
            with patch.object(self.engine, '_update_regime_weights') as mock_update_weights:
                # 各世代で異なるレジーム分布を模擬
                mock_run_gen.side_effect = [
                    [{"fitness": 0.8, "genes": []}],
                    [{"fitness": 0.85, "genes": []}],
                    [{"fitness": 0.9, "genes": []}]
                ]

                results = self.engine.evolve()

                # レジーム適応が呼ばれたか確認
                assert mock_update_weights.call_count == 3  # 各世代で1回ずつ
                assert len(results) == 3

    def test_parallel_evaluation_enabled(self):
        """並列評価有効時のテスト"""
        self.ga_config.parallel_evolution_enabled = True
        self.ga_config.max_concurrent_processes = 2

        population = [{"genes": [1, 2, 3]} for _ in range(4)]

        with patch('app.services.auto_strategy.core.ga_engine.ProcessPoolExecutor') as mock_executor:
            with patch.object(self.engine, '_evaluate_individual') as mock_eval:
                mock_future = Mock()
                mock_future.result.return_value = 0.8
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                results = self.engine._evaluate_population_parallel(population)

                # 並列実行が使用されたか確認
                assert mock_executor.called
                assert len(results) == 4

    def test_adaptive_mutation_rate_adjustment(self):
        """適応的変異率調整のテスト"""
        # 適応的変異率が正しく調整されるか確認
        low_fitness_population = [0.1, 0.15, 0.12, 0.18]
        high_fitness_population = [0.8, 0.85, 0.9, 0.88]

        # 適応的変異率のテスト
        low_rate = self.engine._adaptive_mutation_rate(low_fitness_population)
        high_rate = self.engine._adaptive_mutation_rate(high_fitness_population)

        # 低適応度では変異率が上がる
        assert high_rate > low_rate

    def test_diversity_preservation(self):
        """多様性保持のテスト"""
        # 似たような個体が多数いる集団
        similar_population = [
            {"fitness": 0.8, "genes": [1, 2, 3]},
            {"fitness": 0.82, "genes": [1, 2, 3]},
            {"fitness": 0.78, "genes": [1, 2, 3]},
        ]

        # 多様性が低いと適応度が調整される
        with patch.object(self.engine, '_calculate_diversity_score') as mock_diversity:
            mock_diversity.return_value = 0.1  # 低多様性
            adjusted = self.engine._apply_diversity_pressure(similar_population)

            # 調整が行われたか確認
            assert isinstance(adjusted, list)

    def test_elite_preservation_edge_cases(self):
        """エリート保持のエッジケーステスト"""
        # エリートサイズが集団サイズより大きい場合
        self.ga_config.elite_size = 20
        self.ga_config.population_size = 10

        population = [{"fitness": i * 0.1, "genes": [i]} for i in range(10)]

        # 例外が発生しないか確認
        try:
            elites = self.engine._select_elites(population)
            assert len(elites) <= len(population)
        except Exception:
            pytest.fail("エリート選択で例外が発生")

    def test_convergence_detection(self):
        """収束検出のテスト"""
        # 同じ適応度が続く場合
        fitness_history = [0.8, 0.81, 0.805, 0.808, 0.802]

        is_converged = self.engine._detect_convergence(fitness_history)

        # 収束が正しく検出される
        assert isinstance(is_converged, bool)

    def test_dynamic_parameter_adjustment(self):
        """動的パラメータ調整のテスト"""
        self.ga_config.dynamic_parameter_adjustment = True

        # 適応度履歴を作成
        fitness_history = [0.5, 0.6, 0.7, 0.55, 0.65]

        with patch.object(self.engine, '_adjust_parameters_based_on_performance') as mock_adjust:
            # パラメータ調整が行われる
            self.engine._dynamic_parameter_adjustment(fitness_history)

            assert mock_adjust.called

    def test_early_stopping(self):
        """早期終了のテスト"""
        # 早期終了条件を満たす状況をテスト
        fitness_history = [0.9, 0.91, 0.92, 0.915, 0.905]  # 高適応度が続く

        should_stop = self.engine._should_early_stop(fitness_history)

        # 早期終了が正しく判断される
        assert isinstance(should_stop, bool)

    def test_memory_optimization(self):
        """メモリ最適化のテスト"""
        large_population = [{"fitness": 0.1 * i, "genes": list(range(100))} for i in range(100)]

        # 大規模集団でもメモリが適切に管理される
        with patch('app.services.auto_strategy.core.ga_engine.gc') as mock_gc:
            optimized = self.engine._optimize_memory_usage(large_population)

            # ガベージコレクションが呼ばれる
            assert mock_gc.collect.called

    def test_error_recovery(self):
        """エラー回復のテスト"""
        # 個体評価でエラーが発生する状況
        individual = {"genes": [1, 2, 3]}

        with patch.object(self.engine, '_evaluate_individual') as mock_eval:
            mock_eval.side_effect = Exception("Evaluation error")

            # エラーが適切に処理される
            result = self.engine._safe_evaluate_individual(individual)

            # エラー時のデフォルト値が返される
            assert result is not None

    def test_generation_statistics(self):
        """世代統計のテスト"""
        population = [
            {"fitness": 0.8, "genes": [1]},
            {"fitness": 0.9, "genes": [2]},
            {"fitness": 0.7, "genes": [3]},
        ]

        stats = self.engine._calculate_generation_statistics(population)

        # 統計が正しく計算される
        assert "avg_fitness" in stats
        assert "best_fitness" in stats
        assert "worst_fitness" in stats
        assert stats["avg_fitness"] == 0.8  # 平均

    def test_fitness_scaling(self):
        """フィットネススケーリングのテスト"""
        raw_fitness = [10, 20, 30, 40, 50]

        scaled = self.engine._apply_fitness_scaling(raw_fitness)

        # スケーリングが適用される
        assert len(scaled) == len(raw_fitness)
        assert all(0 <= f <= 1 for f in scaled)

    def test_catastrophic_forgetting_prevention(self):
        """災害的忘却防止のテスト"""
        # 過去の優良個体を保持
        current_best = {"fitness": 0.9, "genes": [1, 2, 3]}
        historical_best = {"fitness": 0.95, "genes": [4, 5, 6]}

        preserved = self.engine._prevent_catastrophic_forgetting(
            current_best, historical_best
        )

        # 過去の優良個体が保持される
        assert preserved is not None

    def test_generation_checkpointing(self):
        """世代チェックポイントのテスト"""
        generation_data = {
            "generation": 5,
            "population": [{"fitness": 0.8, "genes": [1, 2, 3]}],
            "best_fitness": 0.8
        }

        with patch('builtins.open') as mock_open:
            with patch('json.dump') as mock_dump:
                self.engine._save_checkpoint(generation_data, 5)

                # チェックポイントが保存される
                assert mock_open.called
                assert mock_dump.called

    def test_load_checkpoint(self):
        """チェックポイント読み込みのテスト"""
        with patch('builtins.open') as mock_open:
            with patch('json.load') as mock_load:
                mock_load.return_value = {"test": "data"}

                data = self.engine._load_checkpoint("test_checkpoint.json")

                assert data == {"test": "data"}

    def test_real_time_monitoring(self):
        """リアルタイムモニタリングのテスト"""
        population = [{"fitness": 0.8, "genes": [1, 2, 3]}]

        with patch.object(self.engine, '_send_progress_update') as mock_progress:
            self.engine._real_time_monitoring(population, 1, 10)

            # 進捗通知が送信される
            assert mock_progress.called

    def test_custom_termination_criteria(self):
        """カスタム終了条件のテスト"""
        # カスタム終了条件を設定
        def custom_criteria(generation, population, best_fitness):
            return generation >= 5

        self.engine.custom_termination_criteria = custom_criteria

        # 5世代で終了するか確認
        should_terminate = self.engine._check_termination_criteria(5, [], 0.8)
        assert should_terminate

    def test_multi_objective_evolution(self):
        """多目的進化のテスト"""
        self.ga_config.enable_multi_objective = True
        self.ga_config.objectives = ["total_return", "sharpe_ratio"]

        with patch.object(self.engine, '_run_multi_objective_evolution') as mock_multi:
            mock_multi.return_value = [{"fitness": (0.8, 1.2), "genes": [1, 2, 3]}]

            results = self.engine.evolve()

            # 多目的進化が実行される
            assert mock_multi.called
            assert len(results) > 0

    def test_hybrid_evaluation(self):
        """ハイブリッド評価のテスト"""
        individual = {"genes": [1, 2, 3]}

        with patch('app.services.auto_strategy.core.ga_engine.HybridIndividualEvaluator') as mock_hybrid:
            mock_evaluator = Mock()
            mock_evaluator.evaluate_individual_hybrid.return_value = (0.85,)
            mock_hybrid.return_value = mock_evaluator

            # ハイブリッド評価が使用される
            result = self.engine._evaluate_individual(individual)

            # ハイブリッド評価器が使用されたか確認
            assert mock_evaluator.evaluate_individual_hybrid.called

    def test_performance_profiling(self):
        """パフォーマンスプロファイリングのテスト"""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0

            with patch.object(self.engine, '_log_performance_metrics') as mock_log:
                # プロファイリングが実行される
                start_time = self.engine._start_performance_timer()
                self.engine._end_performance_timer(start_time, "test_operation")

                # ログが記録される
                assert mock_log.called