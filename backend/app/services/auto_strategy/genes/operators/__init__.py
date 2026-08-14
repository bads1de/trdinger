"""
StrategyGene の遺伝的演算ロジック。

交叉（crossover）と突然変異（mutation）の演算を提供します。

GA 演算の責務レイヤー
----------------------
遺伝的演算は以下の層に分かれており、それぞれの責務が異なります。

1. ``genes/genetic_utils.py`` (``GeneticUtils``)
   遺伝子種別に依存しない汎用ヘルパー。
   パラメータ抽出・smart_copy・ジェネリック交叉/突然変異・条件コピーを担う。

2. ``genes/operators/`` (本モジュール)
   ``StrategyGene`` 全体の演算実装。
   ``crossover_strategy_genes``（uniform/single-point 交叉）と
   ``mutate_strategy_gene`` / ``adaptive_mutate_strategy_gene`` を担い、
   サブ遺伝子（TPSL/ポジションサイジング/エントリー/イグジット）単位の
   演算は各 Gene クラスの ``crossover`` / ``mutate`` へ委譲する。

3. 各 Gene クラス（``strategy.py`` / ``entry.py`` / ``exit.py`` /
   ``tpsl.py`` / ``position_sizing.py``）
   サブ遺伝子単位の mutate/crossover と ``crossover`` クラスメソッドを提供。

4. ``core/engine/ga_utils.py``
   DEAP toolbox へのアダプタ層。
   ``deap_crossover_strategy_genes`` と ``create_deap_mutate_wrapper`` が
   DEAP の ``mate(ind1, ind2)`` / ``mutate(ind)`` 呼び出しを
   上記レイヤーのシグネチャへ適合させる（実装を持たない薄いラッパー）。
"""

from .crossover import (
    crossover_entry_genes,
    crossover_exit_genes,
    crossover_position_sizing_genes,
    crossover_strategy_genes,
    crossover_tpsl_genes,
    single_point_crossover,
    uniform_crossover,
)
from .mutation import (
    adaptive_mutate_strategy_gene,
    mutate_conditions,
    mutate_indicators,
    mutate_strategy_gene,
)

__all__ = [
    # mutation
    "mutate_indicators",
    "mutate_conditions",
    "mutate_strategy_gene",
    "adaptive_mutate_strategy_gene",
    # crossover
    "crossover_tpsl_genes",
    "crossover_position_sizing_genes",
    "crossover_entry_genes",
    "crossover_exit_genes",
    "crossover_strategy_genes",
    "uniform_crossover",
    "single_point_crossover",
]
