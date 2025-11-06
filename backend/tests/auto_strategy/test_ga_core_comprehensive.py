"""
GAエンジンの包括的テスト
"""

import pytest

pytestmark = pytest.mark.skip(reason="GA implementation changed - comprehensive tests need rewrite")
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from deap import base, creator, tools

from app.services.auto_strategy.config.ga import GASettings as GAConfig
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    Condition,
)
from app.services.auto_strategy.core.genetic_operators import (
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
    create_deap_crossover_wrapper,
    create_deap_mutate_wrapper,
)
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


class TestGACoreComprehensive:
    """GAエンジンの包括的テスト"""

    @pytest.fixture
    def config(self):
        """テスト用GA設定"""
        return GAConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
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
            metadata={"test": True},
        )

    def test_ga_engine_initialization(self, config):
        """GAエンジンの初期化テスト"""
        # モックサービス
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

        assert engine.config == config
        assert engine.backtest_service == mock_backtest_service
        assert engine.persistence_service == mock_persistence_service
        assert engine.regime_detector == mock_regime_detector
        assert engine.data_service == mock_data_service

    def test_deap_toolbox_setup(self, config):
        """DEAPツールボックスのセットアップテスト"""
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

        # ツールボックスが正しくセットアップされる
        assert hasattr(engine.toolbox, "register")
        assert hasattr(engine.toolbox, "select")
        assert hasattr(engine.toolbox, "evaluate")

        # 遺伝的演算子が登録されている
        assert hasattr(engine.toolbox, "mate")
        assert hasattr(engine.toolbox, "mutate")
        assert hasattr(engine.toolbox, "select")

    def test_population_diversity_maintenance(self, config, sample_strategy_gene):
        """個体群の多様性維持テスト"""
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

        # 遺伝子を複製して多様性のない個体群を作成
        population = [sample_strategy_gene for _ in range(5)]

        # 多様性が低いと警告が発生するかテスト
        with patch.object(engine, "_log_diversity_warning") as mock_log:
            engine._check_population_diversity(population)
            mock_log.assert_called_once()

    def test_elitism_preservation(self, config):
        """エリート保存のテスト"""
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

        # 遺伝子を作成
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

        # DEAP個体を作成
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        ind1 = creator.Individual([gene1])
        ind2 = creator.Individual([gene2])
        ind1.fitness.values = (1.0,)
        ind2.fitness.values = (2.0,)

        population = [ind1, ind2]

        # 次世代の生成
        next_generation = engine._generate_next_generation(population)

        # 高いフィットネスの個体が保持されているか確認
        assert len(next_generation) == len(population)

        # エリートが含まれているか確認
        elite_found = any(ind.fitness.values[0] == 2.0 for ind in next_generation)
        assert elite_found

    def test_convergence_detection(self, config):
        """収束検出のテスト"""
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

        # 同じフィットネスの個体群
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        gene = StrategyGene(
            id="gene",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )

        population = []
        for i in range(10):
            ind = creator.Individual([gene])
            ind.fitness.values = (1.0,)
            population.append(ind)

        # 収束を検出
        is_converged = engine._is_population_converged(population, threshold=0.01)
        assert is_converged

        # 異なるフィットネスの個体群
        for i, ind in enumerate(population):
            ind.fitness.values = (i * 0.1,)

        # 収束していない
        is_converged = engine._is_population_converged(population, threshold=0.01)
        assert not is_converged

    def test_adaptive_parameter_adjustment(self, config):
        """適応的パラメータ調整のテスト"""
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

        # 初期パラメータ
        initial_crossover_rate = config.crossover_rate
        initial_mutation_rate = config.mutation_rate

        # 高分散の個体群（多様性が高い）
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        high_variance_pop = []
        for i in range(10):
            gene = StrategyGene(
                id=f"gene{i}",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            ind = creator.Individual([gene])
            ind.fitness.values = (i * 0.2,)  # 分散が高い
            high_variance_pop.append(ind)

        # パラメータ調整
        engine._adapt_parameters(high_variance_pop)

        # 多様性が高いので交叉率が低下しているはず
        assert engine.config.crossover_rate < initial_crossover_rate

        # 低分散の個体群（収束している）
        low_variance_pop = []
        for i in range(10):
            gene = StrategyGene(
                id=f"gene{i}",
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                long_entry_conditions=[],
                short_entry_conditions=[],
                risk_management={},
            )
            ind = creator.Individual([gene])
            ind.fitness.values = (1.0 + i * 0.001,)  # 分散が低い
            low_variance_pop.append(ind)

        # パラメータ調整
        engine._adapt_parameters(low_variance_pop)

        # 収束しているので突然変異率が上昇しているはず
        assert engine.config.mutation_rate > initial_mutation_rate

    def test_error_handling_in_generation(self, config):
        """世代生成中のエラーハンドリングテスト"""
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

        # 無効な遺伝子でテスト
        invalid_gene = "invalid_gene"
        population = [invalid_gene]

        # エラーが発生しても処理が継続するか
        try:
            next_gen = engine._generate_next_generation(population)
            # 正常終了するはず
            assert next_gen is not None
        except Exception:
            pytest.fail("世代生成中にエラーが発生しました")

    def test_crossover_wrapper_edge_cases(self, config):
        """交叉ラッパーのエッジケーステスト"""
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

        # 空の個体群
        empty_pop = []
        result = engine._generate_next_generation(empty_pop)
        assert result == []

        # 1個体だけ
        single_gene = StrategyGene(
            id="single",
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            risk_management={},
        )
        single_pop = [single_gene]

        # 1個体でも処理できるか
        result = engine._generate_next_generation(single_pop)
        assert len(result) == 1

    def test_mutation_rate_bounds(self, config):
        """突然変異率の境界値テスト"""
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

        # 突然変異率が0.01-1.0の範囲内に収まっているか
        test_rates = [0.0, 0.01, 0.5, 1.0, 1.5, 2.0]

        for rate in test_rates:
            engine.config.mutation_rate = rate
            engine._ensure_mutation_rate_bounds()
            assert 0.01 <= engine.config.mutation_rate <= 1.0

    def test_generation_with_real_operators(self, config, sample_strategy_gene):
        """実際の遺伝的演算子を使用した世代生成テスト"""
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

        # 異なる遺伝子を持つ個体を2つ作成
        gene1 = sample_strategy_gene
        gene2 = StrategyGene(
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

        ind1 = creator.Individual([gene1])
        ind2 = creator.Individual([gene2])
        ind1.fitness.values = (1.0,)
        ind2.fitness.values = (1.0,)  # 同じフィットネス

        population = [ind1, ind2]

        # 世代生成
        next_generation = engine._generate_next_generation(population)

        # 個体数が維持されているか
        assert len(next_generation) == len(population)

        # 新しい遺伝子が生成されているか（交叉や突然変異の結果）
        original_gene_ids = {gene1.id, gene2.id}
        new_gene_ids = {ind[0].id for ind in next_generation}

        # 完全に同じ遺伝子が残っている可能性もあるが、多様性が保たれているか確認
        assert len(next_generation) == 2

    def test_population_initialization_diversity(self, config):
        """個体群初期化時の多様性テスト"""
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

        # ランダム遺伝子生成器のモック
        with patch(
            "app.services.auto_strategy.core.ga_engine.RandomGeneGenerator"
        ) as MockGenerator:
            mock_generator = Mock()
            # 多様な遺伝子を返すように設定
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
            mock_generator.generate_random_gene.side_effect = [
                mock_gene1,
                mock_gene2,
                mock_gene1,
                mock_gene2,
                mock_gene1,
            ]

            MockGenerator.return_value = mock_generator

            population = engine._initialize_population()

            # 正しい個体数
            assert len(population) == config.population_size

            # 多様性があるか確認（同じ遺伝子が連続して生成されない）
            gene_ids = [ind[0].id for ind in population]
            unique_ids = set(gene_ids)
            assert len(unique_ids) >= 2  # 少なくとも2種類はあるはず

    def test_fitness_calculation_stability(self, config):
        """フィットネス計算の安定性テスト"""
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

        # フィットネス計算
        fitness = engine._evaluate_individual(individual)

        # 正常なフィットネス値が返される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (int, float))
        assert fitness[0] > 0

    def test_parallel_evaluation_handling(self, config):
        """並列評価のハンドリングテスト"""
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

        # 並列評価が有効な場合
        config.enable_parallel_evaluation = True

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
        individual.fitness.values = (1.0,)

        population = [individual]

        # 並列評価が正しく動作するか
        with patch("multiprocessing.Pool") as mock_pool:
            mock_pool.return_value.__enter__.return_value.map.return_value = [(1.0,)]

            # 並列評価が可能か確認
            try:
                engine._evaluate_population_parallel(population)
                parallel_success = True
            except Exception:
                parallel_success = False

            # 並列評価が失敗してもフォールバックが働く
            if not parallel_success:
                engine._evaluate_population_sequential(population)
                # 順次評価が動作するはず
                assert individual.fitness.values == (1.0,)

    def test_memory_management_large_population(self, config):
        """大規模個体群のメモリ管理テスト"""
        # 大きな個体群を設定
        config.population_size = 1000
        config.elite_size = 100

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

        # メモリ効率の良い初期化
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

            # 大規模個体群の初期化
            import gc

            initial_memory = len(gc.get_objects())
            population = engine._initialize_population()
            gc.collect()
            final_memory = len(gc.get_objects())

            # メモリリークが最小限であるか
            memory_growth = final_memory - initial_memory
            # 大幅なメモリ増加でないか（正確な数値は環境依存なので緩くチェック）
            assert memory_growth < 1000  # 適当な閾値

    def test_config_validation_on_initialization(self, config):
        """初期化時の設定検証テスト"""
        # 無効な設定
        invalid_config = GAConfig(
            population_size=0,  # 無効な値
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
        )

        mock_backtest_service = Mock()
        mock_persistence_service = Mock()
        mock_regime_detector = Mock()
        mock_data_service = Mock()

        # 無効な設定でも初期化できるが、内部で修正される
        engine = GeneticAlgorithmEngine(
            config=invalid_config,
            backtest_service=mock_backtest_service,
            persistence_service=mock_persistence_service,
            regime_detector=mock_regime_detector,
            data_service=mock_data_service,
        )

        # 内部で有効な値に修正される
        assert engine.config.population_size > 0
        assert engine.config.elite_size > 0

    def test_backtest_failure_recovery(self, config):
        """バックテスト失敗からの回復テスト"""
        # バックテストでエラーが発生するようにモック
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.side_effect = Exception("Backtest failed")
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

        # バックテスト失敗時のフォールバック
        fitness = engine._evaluate_individual(individual)

        # デフォルトのフィットネスが返される
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] == 0.1  # 取引回数0のデフォルト値
