"""
coreパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.core as core_package


class TestAutoStrategyCoreInitExports:
    """core/__init__.pyのエクスポートテスト"""

    def test_deap_setup_lazy_load(self):
        """DEAPSetupが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.deap_setup import DEAPSetup

        setup = getattr(core_package, "DEAPSetup")

        assert setup is DEAPSetup

    def test_evolution_runner_lazy_load(self):
        """EvolutionRunnerが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.evolution_runner import (
            EvolutionRunner,
        )

        runner = getattr(core_package, "EvolutionRunner")

        assert runner is EvolutionRunner

    def test_genetic_algorithm_engine_lazy_load(self):
        """GeneticAlgorithmEngineが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.ga_engine import (
            GeneticAlgorithmEngine,
        )

        engine = getattr(core_package, "GeneticAlgorithmEngine")

        assert engine is GeneticAlgorithmEngine

    def test_genetic_algorithm_engine_factory_lazy_load(self):
        """GeneticAlgorithmEngineFactoryが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.ga_engine_factory import (
            GeneticAlgorithmEngineFactory,
        )

        factory = getattr(core_package, "GeneticAlgorithmEngineFactory")

        assert factory is GeneticAlgorithmEngineFactory

    def test_crossover_strategy_genes_lazy_load(self):
        """crossover_strategy_genesが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.ga_utils import (
            crossover_strategy_genes,
        )

        func = getattr(core_package, "crossover_strategy_genes")

        assert func is crossover_strategy_genes

    def test_mutate_strategy_gene_lazy_load(self):
        """mutate_strategy_geneが遅延ロードされる"""
        from app.services.auto_strategy.core.engine.ga_utils import mutate_strategy_gene

        func = getattr(core_package, "mutate_strategy_gene")

        assert func is mutate_strategy_gene

    def test_condition_evaluator_lazy_load(self):
        """ConditionEvaluatorが遅延ロードされる"""
        from app.services.auto_strategy.core.evaluation.condition_evaluator import (
            ConditionEvaluator,
        )

        evaluator = getattr(core_package, "ConditionEvaluator")

        assert evaluator is ConditionEvaluator

    def test_evaluation_strategy_lazy_load(self):
        """EvaluationStrategyが遅延ロードされる"""
        from app.services.auto_strategy.core.evaluation.evaluation_strategies import (
            EvaluationStrategy,
        )

        strategy = getattr(core_package, "EvaluationStrategy")

        assert strategy is EvaluationStrategy

    def test_evaluator_wrapper_lazy_load(self):
        """EvaluatorWrapperが遅延ロードされる"""
        from app.services.auto_strategy.core.evaluation.evaluator_wrapper import (
            EvaluatorWrapper,
        )

        wrapper = getattr(core_package, "EvaluatorWrapper")

        assert wrapper is EvaluatorWrapper

    def test_individual_evaluator_lazy_load(self):
        """IndividualEvaluatorが遅延ロードされる"""
        from app.services.auto_strategy.core.evaluation.individual_evaluator import (
            IndividualEvaluator,
        )

        evaluator = getattr(core_package, "IndividualEvaluator")

        assert evaluator is IndividualEvaluator

    def test_parallel_evaluator_lazy_load(self):
        """ParallelEvaluatorが遅延ロードされる"""
        from app.services.auto_strategy.core.evaluation.parallel_evaluator import (
            ParallelEvaluator,
        )

        evaluator = getattr(core_package, "ParallelEvaluator")

        assert evaluator is ParallelEvaluator

    def test_fitness_calculator_lazy_load(self):
        """FitnessCalculatorが遅延ロードされる"""
        from app.services.auto_strategy.core.fitness.fitness_calculator import (
            FitnessCalculator,
        )

        calculator = getattr(core_package, "FitnessCalculator")

        assert calculator is FitnessCalculator

    def test_fitness_sharing_lazy_load(self):
        """FitnessSharingが遅延ロードされる"""
        from app.services.auto_strategy.core.fitness.fitness_sharing import (
            FitnessSharing,
        )

        sharing = getattr(core_package, "FitnessSharing")

        assert sharing is FitnessSharing

    def test_hybrid_feature_adapter_lazy_load(self):
        """HybridFeatureAdapterが遅延ロードされる"""
        from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
            HybridFeatureAdapter,
        )

        adapter = getattr(core_package, "HybridFeatureAdapter")

        assert adapter is HybridFeatureAdapter

    def test_wavelet_feature_transformer_lazy_load(self):
        """WaveletFeatureTransformerが遅延ロードされる"""
        from app.services.auto_strategy.core.hybrid.hybrid_feature_adapter import (
            WaveletFeatureTransformer,
        )

        transformer = getattr(core_package, "WaveletFeatureTransformer")

        assert transformer is WaveletFeatureTransformer

    def test_hybrid_individual_evaluator_lazy_load(self):
        """HybridIndividualEvaluatorが遅延ロードされる"""
        from app.services.auto_strategy.core.hybrid.hybrid_individual_evaluator import (
            HybridIndividualEvaluator,
        )

        evaluator = getattr(core_package, "HybridIndividualEvaluator")

        assert evaluator is HybridIndividualEvaluator

    def test_hybrid_predictor_lazy_load(self):
        """HybridPredictorが遅延ロードされる"""
        from app.services.auto_strategy.core.hybrid.hybrid_predictor import (
            HybridPredictor,
        )

        predictor = getattr(core_package, "HybridPredictor")

        assert predictor is HybridPredictor

    def test_operand_group_lazy_load(self):
        """OperandGroupが遅延ロードされる"""
        from app.services.auto_strategy.core.strategy.operand_grouping import (
            OperandGroup,
        )

        group = getattr(core_package, "OperandGroup")

        assert group is OperandGroup

    def test_operand_grouping_system_lazy_load(self):
        """OperandGroupingSystemが遅延ロードされる"""
        from app.services.auto_strategy.core.strategy.operand_grouping import (
            OperandGroupingSystem,
        )

        system = getattr(core_package, "OperandGroupingSystem")

        assert system is OperandGroupingSystem

    def test_operand_grouping_system_instance_lazy_load(self):
        """operand_grouping_systemインスタンスが遅延ロードされる"""
        from app.services.auto_strategy.core.strategy.operand_grouping import (
            operand_grouping_system,
        )

        system = getattr(core_package, "operand_grouping_system")

        assert system is operand_grouping_system

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = core_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            # Engine
            "DEAPSetup",
            "EvolutionRunner",
            "GeneticAlgorithmEngine",
            "GeneticAlgorithmEngineFactory",
            "crossover_strategy_genes",
            "mutate_strategy_gene",
            # Evaluation
            "ConditionEvaluator",
            "EvaluationStrategy",
            "EvaluatorWrapper",
            "IndividualEvaluator",
            "ParallelEvaluator",
            # Fitness
            "FitnessCalculator",
            "FitnessSharing",
            # Hybrid
            "HybridFeatureAdapter",
            "HybridIndividualEvaluator",
            "HybridPredictor",
            "WaveletFeatureTransformer",
            # Strategy
            "OperandGroup",
            "OperandGroupingSystem",
            "operand_grouping_system",
        ]

        for item in expected_items:
            assert item in core_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(core_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert core_package.__doc__ is not None
        assert len(core_package.__doc__) > 0
