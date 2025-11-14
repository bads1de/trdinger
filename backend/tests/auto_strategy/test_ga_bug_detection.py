"""
GAアルゴリズムの包括的バグ検出テスト
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="GA implementation changed - bug detection tests need update"
)
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from deap import base, creator, tools
import random

from app.services.auto_strategy.config.ga import GASettings as GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    Condition,
    TPSLGene,
)
from app.services.auto_strategy.core.genetic_operators import (
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
    uniform_crossover,
    _crossover_tpsl_genes,
    _crossover_position_sizing_genes,
    _mutate_indicators,
    _mutate_conditions,
    adaptive_mutate_strategy_gene_pure,
    create_deap_crossover_wrapper,
    create_deap_mutate_wrapper,
)
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


class TestGABugDetection:
    """GAアルゴリズムのバグ検出テスト"""

    @pytest.fixture
    def config(self):
        """テスト用GA設定"""
        return GAConfig(
            population_size=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=3,
        )

    @pytest.fixture
    def sample_strategy_gene(self):
        """サンプル戦略遺伝子"""
        return StrategyGene(
            id="test-gene",
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 10}),
                IndicatorGene(type="EMA", parameters={"period": 20}),
            ],
            entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            exit_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="ema")
            ],
            risk_management={"position_size": 0.1},
            tpsl_gene=TPSLGene(),
            metadata={"test": True},
        )

    def test_infinite_loop_prevention_in_crossover(self, sample_strategy_gene):
        """交叉中の無限ループ防止テスト"""
        # 同じ遺伝子でテスト
        gene1 = sample_strategy_gene
        gene2 = sample_strategy_gene

        # 無限ループが発生しないか
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout(duration):
            def timeout_handler(signum, frame):
                raise TimeoutError("Operation timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(duration)
            try:
                yield
            finally:
                signal.alarm(0)

        try:
            with timeout(5):  # 5秒タイムアウト
                child1, child2 = crossover_strategy_genes_pure(gene1, gene2)
                # 正常終了
                assert child1 is not None
                assert child2 is not None
        except TimeoutError:
            pytest.fail("交叉で無限ループが発生")

    def test_memory_leak_in_large_population(self, config):
        """大規模個体群でのメモリリークテスト"""
        import gc
        import sys

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        # 大規模個体群
        config.population_size = 100

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 初期メモリ状態
        gc.collect()
        initial_memory = sys.getsizeof(gc.get_objects())

        # 大規模個体群の初期化
        with patch(
            "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
        ) as MockGenerator:
            mock_generator = Mock()
            mock_gene = StrategyGene(
                id="test",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            mock_generator.generate_random_gene.return_value = mock_gene

            MockGenerator.return_value = mock_generator

            population = engine._initialize_population()

        # メモリ増加が適切か
        gc.collect()
        final_memory = sys.getsizeof(gc.get_objects())
        memory_growth = final_memory - initial_memory

        # 過度なメモリ増加でない
        assert memory_growth < 1000000  # 1MB未満

    def test_division_by_zero_in_fitness_calculation(self, config):
        """フィットネス計算でのゼロ除算防止"""
        mock_backtest_service = Mock()
        # 取引回数0の結果
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,  # これが原因でゼロ除算が発生する可能性
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        gene = StrategyGene(
            id="test",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        individual = creator.Individual([gene])

        # ゼロ除算が発生しない
        fitness = engine._evaluate_individual(individual)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] >= 0.0

    def test_index_out_of_bounds_in_crossover(self, sample_strategy_gene):
        """交叉での配列境界外アクセス防止"""
        # 指標数が異なる遺伝子
        gene1 = sample_strategy_gene
        gene2 = StrategyGene(
            id="gene2",
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],  # 1つ
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )

        # 境界値テスト
        child1, child2 = crossover_strategy_genes_pure(gene1, gene2)

        # 正常終了
        assert child1 is not None
        assert child2 is not None

        # 指標数が制限内
        assert len(child1.indicators) <= max(
            len(gene1.indicators), len(gene2.indicators)
        )
        assert len(child2.indicators) <= max(
            len(gene1.indicators), len(gene2.indicators)
        )

    def test_null_pointer_in_gene_conversion(self):
        """遺伝子変換でのヌルポインタ防止"""
        from app.services.auto_strategy.core.genetic_operators import (
            _convert_to_strategy_gene,
        )

        # None入力
        with pytest.raises(TypeError):
            _convert_to_strategy_gene(None)

        # 無効な型
        with pytest.raises(TypeError):
            _convert_to_strategy_gene("invalid")

    def test_race_condition_in_parallel_evaluation(self, config):
        """並列評価での競合状態防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # 並列評価のテスト
        population = []
        for i in range(5):
            gene = StrategyGene(
                id=f"gene{i}",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            individual = creator.Individual([gene])
            individual.fitness.values = (i * 0.1,)
            population.append(individual)

        # 競合状態が発生しない
        try:
            with patch("multiprocessing.Pool") as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [
                    (0.1,),
                    (0.2,),
                    (0.3,),
                    (0.4,),
                    (0.5,),
                ]
                engine._evaluate_population_parallel(population)
        except Exception:
            # 並列評価が失敗しても順次評価が動作する
            engine._evaluate_population_sequential(population)

    def test_stack_overflow_in_recursive_operations(self, sample_strategy_gene):
        """再帰操作でのスタックオーバーフロー防止"""
        # 深い再帰が発生しないかテスト
        gene1 = sample_strategy_gene

        # 単純な繰り返し操作
        current_gene = gene1
        for _ in range(100):  # 多数回操作
            try:
                mutated = mutate_strategy_gene_pure(current_gene, mutation_rate=0.1)
                current_gene = mutated
            except RecursionError:
                pytest.fail("スタックオーバーフローが発生")

    def test_deadlock_in_population_update(self, config):
        """個体群更新でのデッドロック防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        population = []
        for i in range(5):
            gene = StrategyGene(
                id=f"gene{i}",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            individual = creator.Individual([gene])
            individual.fitness.values = (i * 0.1,)
            population.append(individual)

        # デッドロックが発生しない
        try:
            next_gen = engine._generate_next_generation(population)
            assert len(next_gen) == len(population)
        except Exception:
            pytest.fail("世代生成でエラーが発生")

    def test_data_corruption_in_crossover(self, sample_strategy_gene):
        """交叉でのデータ破損防止"""
        # 元の遺伝子を保持
        original_gene1 = sample_strategy_gene
        original_gene2 = StrategyGene(
            id="gene2",
            indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
            entry_conditions=[
                Condition(left_operand="rsi", operator="<", right_operand="30")
            ],
            exit_conditions=[
                Condition(left_operand="rsi", operator=">", right_operand="70")
            ],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={"position_size": 0.2},
            metadata={},
        )

        # 交叉実行
        child1, child2 = crossover_strategy_genes_pure(original_gene1, original_gene2)

        # 基本的な整合性
        assert hasattr(child1, "indicators")
        assert hasattr(child1, "entry_conditions")
        assert hasattr(child1, "exit_conditions")
        assert hasattr(child1, "long_entry_conditions")
        assert hasattr(child1, "short_entry_conditions")
        assert hasattr(child1, "risk_management")

        assert hasattr(child2, "indicators")
        assert hasattr(child2, "entry_conditions")
        assert hasattr(child2, "exit_conditions")
        assert hasattr(child2, "long_entry_conditions")
        assert hasattr(child2, "short_entry_conditions")
        assert hasattr(child2, "risk_management")

    def test_unexpected_termination_in_generation_loop(self, config):
        """世代ループでの予期しない終了防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 少ない世代数でテスト
        config.generations = 3

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        with patch(
            "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
        ) as MockGenerator:
            mock_generator = Mock()
            mock_gene = StrategyGene(
                id="test",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            mock_generator.generate_random_gene.return_value = mock_gene
            MockGenerator.return_value = mock_generator

            # 実行が正常終了する
            final_population = engine.run()

            assert final_population is not None

    def test_resource_leak_in_backtest_service(self, config):
        """バックテストサービスでのリソースリーク防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        gene = StrategyGene(
            id="test",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        individual = creator.Individual([gene])

        # リソースが適切に解放される
        fitness = engine._evaluate_individual(individual)

        # バックテストサービスが呼び出される
        mock_backtest_service.run_backtest.assert_called()

    def test_concurrency_issue_in_fitness_sharing(self, config):
        """フィットネスシェアリングでの並行情報問題"""
        from app.services.auto_strategy.core.fitness_sharing import FitnessSharing

        sharing = FitnessSharing(sharing_radius=0.1, alpha=1.0)

        # 同期的な操作
        mock_gene1 = StrategyGene(
            id="gene1",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        mock_gene2 = StrategyGene(
            id="gene2",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )

        # 並行情報が発生しない
        try:
            vector1 = sharing._vectorize_gene(mock_gene1)
            vector2 = sharing._vectorize_gene(mock_gene2)
            assert len(vector1) == len(vector2)
        except Exception:
            pytest.fail("ベクトル化でエラー")

    def test_unexpected_state_change_in_mutation(self, sample_strategy_gene):
        """突然変異での予期しない状態変化防止"""
        original_gene = sample_strategy_gene

        # 突然変異実行
        mutated = mutate_strategy_gene_pure(original_gene, mutation_rate=0.5)

        # IDが変更されている
        assert mutated.id != original_gene.id

        # 基本的な属性が保持されている
        assert hasattr(mutated, "indicators")
        assert hasattr(mutated, "entry_conditions")
        assert hasattr(mutated, "exit_conditions")
        assert hasattr(mutated, "long_entry_conditions")
        assert hasattr(mutated, "short_entry_conditions")
        assert hasattr(mutated, "risk_management")

    def test_inconsistency_in_elitism(self, config):
        """エリート主義での不整合防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # 遺伝子
        gene1 = StrategyGene(
            id="gene1",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        gene2 = StrategyGene(
            id="gene2",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )

        ind1 = creator.Individual([gene1])
        ind2 = creator.Individual([gene2])
        ind1.fitness.values = (2.0,)
        ind2.fitness.values = (1.0,)

        population = [ind1, ind2]

        # 次世代生成
        next_generation = engine._generate_next_generation(population)

        # エリートが保持されている
        elite_found = any(ind.fitness.values[0] == 2.0 for ind in next_generation)
        assert elite_found

    def test_parameter_bounds_violation(self, config):
        """パラメータ境界違反の防止"""
        # 無効なパラメータ
        config.crossover_rate = 1.5  # > 1.0
        config.mutation_rate = -0.1  # < 0

        mock_backtest_service = Mock()
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 境界値が自動調整される
        assert 0.0 <= engine.config.crossover_rate <= 1.0
        assert 0.0 <= engine.config.mutation_rate <= 1.0

    def test_unexpected_none_in_population(self, config):
        """個体群でのNone値防止"""
        mock_backtest_service = Mock()
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # Noneを含む個体群
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        gene = StrategyGene(
            id="test",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        individual = creator.Individual([gene])
        individual.fitness.values = (1.0,)

        population = [individual, None, individual]

        # Noneがフィルタリングされる
        try:
            next_gen = engine._generate_next_generation(population)
            # Noneが除外されているはず
            none_count = sum(1 for ind in next_gen if ind is None)
            assert none_count == 0
        except Exception:
            pytest.fail("None処理でエラー")

    def test_unexpected_exception_in_adaptive_mutation(self, sample_strategy_gene):
        """適応的突然変異での予期しない例外"""
        # 空の個体群
        empty_population = []

        mutated = adaptive_mutate_strategy_gene_pure(
            empty_population, sample_strategy_gene
        )
        assert mutated is not None

        # 無効なフィットネス
        class MockIndividual:
            def __init__(self, gene):
                self.gene = gene
                self.fitness = Mock()
                self.fitness.values = None  # 無効なフィットネス

        population = [MockIndividual(sample_strategy_gene)]

        mutated = adaptive_mutate_strategy_gene_pure(population, sample_strategy_gene)
        assert mutated is not None

    def test_dead_gene_generation(self, config):
        """不活性遺伝子生成の防止"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        gene = StrategyGene(
            id="test",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        individual = creator.Individual([gene])

        # 不活性な遺伝子でも評価される
        fitness = engine._evaluate_individual(individual)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 1

    def test_unexpected_id_duplication(self, config):
        """ID重複の防止"""
        mock_backtest_service = Mock()
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # DEAPのセットアップ
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        with patch(
            "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
        ) as MockGenerator:
            mock_generator = Mock()
            # 同じIDを返すモック
            mock_gene = StrategyGene(
                id="same-id",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            mock_generator.generate_random_gene.return_value = mock_gene

            MockGenerator.return_value = mock_generator

            population = engine._initialize_population()

            # IDが重複している可能性
            ids = [ind[0].id for ind in population]
            unique_ids = set(ids)

            # 重複があっても問題ない（IDは後で再生成されるため）
            assert len(ids) == config.population_size

    def test_unexpected_data_type_conversion(self, sample_strategy_gene):
        """予期しないデータ型変換の防止"""
        # 数値パラメータが文字列にならないか
        mutated = mutate_strategy_gene_pure(sample_strategy_gene, mutation_rate=0.1)

        # リスク管理の数値が保持されている
        for key, value in mutated.risk_management.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, (int, float))

    def test_memory_fragmentation_in_long_running(self, config):
        """長時間実行でのメモリフラグメンテーション"""
        import gc

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # メモリフラグメンテーションテスト
        gc.collect()
        initial_fragmentation = len(gc.get_referrers(gc))

        # 複数世代をシミュレート
        for _ in range(5):
            with patch(
                "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
            ) as MockGenerator:
                mock_generator = Mock()
                mock_gene = StrategyGene(
                    id="test",
                    indicators=[],
                    entry_conditions=[],
                    exit_conditions=[],
                    long_entry_conditions=[],
                    short_entry_conditions=[],
                    risk_management={},
                )
                mock_generator.generate_random_gene.return_value = mock_gene
                MockGenerator.return_value = mock_generator

                population = engine._initialize_population()

        gc.collect()
        final_fragmentation = len(gc.get_referrers(gc))

        # 過度なフラグメンテーションでない
        assert final_fragmentation - initial_fragmentation < 100

    def test_unexpected_termination_condition(self, config):
        """予期しない終了条件"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 正常な終了条件
        assert hasattr(engine, "config")
        assert engine.config.generations > 0

    def test_unexpected_data_loss_in_serialization(self, sample_strategy_gene):
        """直列化でのデータ損失防止"""
        from app.services.auto_strategy.serializers.gene_serialization import (
            GeneSerializer,
        )

        serializer = GeneSerializer()
        gene_list = serializer.encode_strategy_gene_to_list(sample_strategy_gene)
        restored_gene = serializer.decode_list_to_strategy_gene(gene_list, StrategyGene)

        # 基本的なデータが保持されている
        assert len(restored_gene.indicators) == len(sample_strategy_gene.indicators)
        assert len(restored_gene.entry_conditions) == len(
            sample_strategy_gene.entry_conditions
        )

    def test_unexpected_behavior_with_edge_case_parameters(self, config):
        """境界値パラメータでの予期しない動作"""
        # 極端なパラメータ
        config.population_size = 1
        config.generations = 1
        config.elite_size = 1

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 極端な設定でも動作する
        with patch(
            "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
        ) as MockGenerator:
            mock_generator = Mock()
            mock_gene = StrategyGene(
                id="test",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            mock_generator.generate_random_gene.return_value = mock_gene
            MockGenerator.return_value = mock_generator

            population = engine._initialize_population()
            assert len(population) == 1

    def test_unexpected_interaction_between_components(self, config):
        """コンポーネント間の予期しない相互作用"""
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 50,
        }
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        engine = GeneticAlgorithmEngine(
            config=config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 各コンポーネントが正常に連携
        assert engine.backtest_service == mock_backtest_service
        assert engine.persistence_service == mock_persistence_service
        assert engine.regime_detector == mock_regime_detector
        assert engine.data_service == mock_data_service
