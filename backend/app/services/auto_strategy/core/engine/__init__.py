"""
GAエンジンコアモジュール

遺伝的アルゴリズムエンジン、DEAP設定、進化実行を提供します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .deap_setup import DEAPSetup
    from .evolution_runner import EvolutionRunner, EvolutionStoppedError
    from .ga_engine import GeneticAlgorithmEngine
    from .ga_engine_factory import GeneticAlgorithmEngineFactory
    from .ga_utils import (
        _gene_kwargs,
        create_deap_mutate_wrapper,
        crossover_strategy_genes,
        mutate_strategy_gene,
    )

_ATTRIBUTE_EXPORTS = {
    "DEAPSetup": ".deap_setup",
    "EvolutionRunner": ".evolution_runner",
    "EvolutionStoppedError": ".evolution_runner",
    "GeneticAlgorithmEngine": ".ga_engine",
    "GeneticAlgorithmEngineFactory": ".ga_engine_factory",
    "_gene_kwargs": ".ga_utils",
    "create_deap_mutate_wrapper": ".ga_utils",
    "crossover_strategy_genes": ".ga_utils",
    "mutate_strategy_gene": ".ga_utils",
}

__all__ = [
    "DEAPSetup",
    "EvolutionRunner",
    "EvolutionStoppedError",
    "GeneticAlgorithmEngine",
    "GeneticAlgorithmEngineFactory",
    "_gene_kwargs",
    "create_deap_mutate_wrapper",
    "crossover_strategy_genes",
    "mutate_strategy_gene",
]

from ..._lazy_import import setup_lazy_import  # noqa: E402
setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
