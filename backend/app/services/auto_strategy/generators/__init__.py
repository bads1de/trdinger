"""
Auto Strategy Generators モジュール

戦略生成に関連するジェネレータークラスを提供します。
factories/ の機能を統合しています。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .condition_generator import ConditionGenerator
    from .random_gene_generator import RandomGeneGenerator
    from .seed_strategy_factory import (
        SeedStrategyFactory,
        inject_seeds_into_population,
    )

_ATTRIBUTE_EXPORTS = {
    "RandomGeneGenerator": ".random_gene_generator",
    "ConditionGenerator": ".condition_generator",
    "SeedStrategyFactory": ".seed_strategy_factory",
    "inject_seeds_into_population": ".seed_strategy_factory",
}

__all__ = [
    "RandomGeneGenerator",
    "ConditionGenerator",
    "SeedStrategyFactory",
    "inject_seeds_into_population",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
