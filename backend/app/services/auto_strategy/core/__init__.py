"""
遺伝的演算子モジュール

戦略遺伝子の交叉・突然変異などの遺伝的演算子を提供します。
"""

import importlib as _importlib
import sys as _sys

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
from .hybrid.hybrid_feature_adapter import HybridFeatureAdapter, WaveletFeatureTransformer
from .hybrid.hybrid_individual_evaluator import HybridIndividualEvaluator
from .hybrid.hybrid_predictor import HybridPredictor
from .strategy.operand_grouping import OperandGroup, OperandGroupingSystem, operand_grouping_system


def __getattr__(name: str):
    """遅延インポートで循環インポートを回避"""
    if name in ("DEAPSetup",):
        from .engine.deap_setup import DEAPSetup

        return DEAPSetup
    if name in ("EvolutionRunner",):
        from .engine.evolution_runner import EvolutionRunner

        return EvolutionRunner
    if name in ("GeneticAlgorithmEngine",):
        from .engine.ga_engine import GeneticAlgorithmEngine

        return GeneticAlgorithmEngine
    if name in ("GeneticAlgorithmEngineFactory",):
        from .engine.ga_engine_factory import GeneticAlgorithmEngineFactory

        return GeneticAlgorithmEngineFactory
    if name in ("crossover_strategy_genes", "mutate_strategy_gene"):
        from .engine.ga_utils import crossover_strategy_genes, mutate_strategy_gene

        return {"crossover_strategy_genes": crossover_strategy_genes, "mutate_strategy_gene": mutate_strategy_gene}[name]
    if name in ("ConditionEvaluator",):
        from .evaluation.condition_evaluator import ConditionEvaluator

        return ConditionEvaluator
    if name in ("EvaluationStrategy",):
        from .evaluation.evaluation_strategies import EvaluationStrategy

        return EvaluationStrategy
    if name in ("EvaluatorWrapper",):
        from .evaluation.evaluator_wrapper import EvaluatorWrapper

        return EvaluatorWrapper
    if name in ("IndividualEvaluator",):
        from .evaluation.individual_evaluator import IndividualEvaluator

        return IndividualEvaluator
    if name in ("ParallelEvaluator",):
        from .evaluation.parallel_evaluator import ParallelEvaluator

        return ParallelEvaluator
    if name in ("FitnessCalculator",):
        from .fitness.fitness_calculator import FitnessCalculator

        return FitnessCalculator
    if name in ("FitnessSharing",):
        from .fitness.fitness_sharing import FitnessSharing

        return FitnessSharing
    if name in ("HybridFeatureAdapter", "WaveletFeatureTransformer"):
        from .hybrid.hybrid_feature_adapter import HybridFeatureAdapter, WaveletFeatureTransformer

        return {"HybridFeatureAdapter": HybridFeatureAdapter, "WaveletFeatureTransformer": WaveletFeatureTransformer}[name]
    if name in ("HybridIndividualEvaluator",):
        from .hybrid.hybrid_individual_evaluator import HybridIndividualEvaluator

        return HybridIndividualEvaluator
    if name in ("HybridPredictor",):
        from .hybrid.hybrid_predictor import HybridPredictor

        return HybridPredictor
    if name in ("OperandGroup", "OperandGroupingSystem", "operand_grouping_system"):
        from .strategy.operand_grouping import OperandGroup, OperandGroupingSystem, operand_grouping_system

        return {
            "OperandGroup": OperandGroup,
            "OperandGroupingSystem": OperandGroupingSystem,
            "operand_grouping_system": operand_grouping_system,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

# Backward compatibility for legacy shim module paths.
for _alias, _target in {
    "condition_evaluator": ".evaluation.condition_evaluator",
    "parallel_evaluator": ".evaluation.parallel_evaluator",
    "individual_evaluator": ".evaluation.individual_evaluator",
    "evaluation_metrics": ".evaluation.evaluation_metrics",
    "evaluator_wrapper": ".evaluation.evaluator_wrapper",
    "evaluation_worker": ".evaluation.evaluation_worker",
    "hybrid_feature_adapter": ".hybrid.hybrid_feature_adapter",
    "hybrid_individual_evaluator": ".hybrid.hybrid_individual_evaluator",
    "hybrid_predictor": ".hybrid.hybrid_predictor",
    "operand_grouping": ".strategy.operand_grouping",
    "ga_engine": ".engine.ga_engine",
    "ga_engine_factory": ".engine.ga_engine_factory",
    "fitness_sharing": ".fitness.fitness_sharing",
}.items():
    _module = _importlib.import_module(_target, __name__)
    _sys.modules[f"{__name__}.{_alias}"] = _module
    setattr(_sys.modules[__name__], _alias, _module)
