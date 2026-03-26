"""
遺伝的演算子モジュール

戦略遺伝子の交叉・突然変異などの遺伝的演算子を提供します。
"""

import importlib as _importlib
import sys as _sys

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


def _load_export(name: str):
    module = _importlib.import_module(_ATTRIBUTE_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    """遅延インポートで循環インポートを回避"""
    if name in _ATTRIBUTE_EXPORTS:
        return _load_export(name)
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
