"""
遺伝的演算子モジュール

戦略遺伝子の交叉・突然変異などの遺伝的演算子を提供します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine.deap_setup import DEAPSetup
    from .engine.evolution_runner import EvolutionRunner
    from .engine.ga_engine import GeneticAlgorithmEngine
    from .engine.ga_engine_factory import GeneticAlgorithmEngineFactory
    from .engine.ga_utils import crossover_strategy_genes, mutate_strategy_gene
    from .evaluation.condition_evaluator import ConditionEvaluator
    from .evaluation.evaluation_strategies import EvaluationStrategy
    from .evaluation.evaluator_wrapper import EvaluatorWrapper
    from .evaluation.individual_evaluator import IndividualEvaluator
    from .evaluation.parallel_evaluator import ParallelEvaluator
    from .fitness.fitness_calculator import FitnessCalculator
    from .fitness.fitness_sharing import FitnessSharing
    from .hybrid.hybrid_feature_adapter import (
        HybridFeatureAdapter,
        WaveletFeatureTransformer,
    )
    from .hybrid.hybrid_individual_evaluator import HybridIndividualEvaluator
    from .hybrid.hybrid_predictor import HybridPredictor
    from .strategy.operand_grouping import (
        OperandGroup,
        OperandGroupingSystem,
        operand_grouping_system,
    )

_ATTRIBUTE_EXPORTS = {
    "DEAPSetup": ".engine.deap_setup",
    "EvolutionRunner": ".engine.evolution_runner",
    "GeneticAlgorithmEngine": ".engine.ga_engine",
    "GeneticAlgorithmEngineFactory": ".engine.ga_engine_factory",
    "crossover_strategy_genes": ".engine.ga_utils",
    "mutate_strategy_gene": ".engine.ga_utils",
    "ConditionEvaluator": ".evaluation.condition_evaluator",
    "EvaluationStrategy": ".evaluation.evaluation_strategies",
    "EvaluatorWrapper": ".evaluation.evaluator_wrapper",
    "IndividualEvaluator": ".evaluation.individual_evaluator",
    "ParallelEvaluator": ".evaluation.parallel_evaluator",
    "FitnessCalculator": ".fitness.fitness_calculator",
    "FitnessSharing": ".fitness.fitness_sharing",
    "HybridFeatureAdapter": ".hybrid.hybrid_feature_adapter",
    "WaveletFeatureTransformer": ".hybrid.hybrid_feature_adapter",
    "HybridIndividualEvaluator": ".hybrid.hybrid_individual_evaluator",
    "HybridPredictor": ".hybrid.hybrid_predictor",
    "OperandGroup": ".strategy.operand_grouping",
    "OperandGroupingSystem": ".strategy.operand_grouping",
    "operand_grouping_system": ".strategy.operand_grouping",
}

__all__ = [
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

from .._lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
