"""
StrategyGene の遺伝的演算ロジック（後方互換用）。

このモジュールは operators/ パッケージにリダイレクトします。
新しいコードでは直接 `from .operators import ...` を使用してください。
"""

from __future__ import annotations

from .operators import (
    adaptive_mutate_strategy_gene,
    crossover_entry_genes,
    crossover_position_sizing_genes,
    crossover_strategy_genes,
    crossover_tpsl_genes,
    mutate_conditions,
    mutate_indicators,
    mutate_strategy_gene,
    single_point_crossover,
    uniform_crossover,
)
from .operators.mutation import (  # noqa: F401 (テスト用)
    _create_sub_gene,  # pyright: ignore[reportUnusedImport]
    _get_creation_probability_multiplier,  # pyright: ignore[reportUnusedImport]
    _iter_mutable_sub_gene_specs,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    "mutate_indicators",
    "mutate_conditions",
    "mutate_strategy_gene",
    "adaptive_mutate_strategy_gene",
    "crossover_tpsl_genes",
    "crossover_position_sizing_genes",
    "crossover_entry_genes",
    "crossover_strategy_genes",
    "uniform_crossover",
    "single_point_crossover",
]
