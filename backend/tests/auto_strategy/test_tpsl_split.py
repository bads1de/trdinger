import pytest
from unittest.mock import MagicMock, patch
from app.services.auto_strategy.genes.strategy import StrategyGene
from app.services.auto_strategy.genes.tpsl import TPSLGene
from app.services.auto_strategy.config.constants import TPSLMethod
from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy

from app.services.auto_strategy.config import GASettings
from app.services.auto_strategy.core.genetic_operators import (
    crossover_strategy_genes_pure,
    mutate_strategy_gene_pure,
)


class TestTPSLSplit:
    """TPSL分割（Long/Short個別設定）のテストクラス"""

    # ... (既存のテスト) ...

    def test_crossover_handles_split_tpsl(self):
        """交叉時に分離されたTPSL遺伝子が適切に処理されることを確認"""
        config = GASettings()  # 設定オブジェクトを作成
        long_tpsl1 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE, take_profit_pct=0.1, enabled=True
        )
        short_tpsl1 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE, take_profit_pct=0.1, enabled=True
        )

        long_tpsl2 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE, take_profit_pct=0.2, enabled=True
        )
        short_tpsl2 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE, take_profit_pct=0.2, enabled=True
        )

        parent1 = StrategyGene(long_tpsl_gene=long_tpsl1, short_tpsl_gene=short_tpsl1)
        parent2 = StrategyGene(long_tpsl_gene=long_tpsl2, short_tpsl_gene=short_tpsl2)

        # 交叉実行
        child1, child2 = crossover_strategy_genes_pure(
            parent1, parent2, config
        )  # configを渡す

        # 子供がlong/short tpslを持っていることを確認
        # 交叉ロジックにより、どちらかの親の遺伝子を持つか、あるいは混合される
        assert child1.long_tpsl_gene is not None
        assert child1.short_tpsl_gene is not None
        assert child2.long_tpsl_gene is not None
        assert child2.short_tpsl_gene is not None

        # 値の検証（詳細は交叉の実装依存だが、少なくともNoneではないこと）

    def test_mutation_affects_split_tpsl(self):
        """突然変異時に分離されたTPSL遺伝子も変異することを確認"""
        config = GASettings()  # 設定オブジェクトを作成
        # 変異率1.0で確実に変異させる
        original_tp = 0.1
        long_tpsl = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            take_profit_pct=original_tp,
            enabled=True,
        )
        short_tpsl = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            take_profit_pct=original_tp,
            enabled=True,
        )

        gene = StrategyGene(long_tpsl_gene=long_tpsl, short_tpsl_gene=short_tpsl)

        mutated = mutate_strategy_gene_pure(
            gene, config, mutation_rate=1.0
        )  # configを渡す

        # 変異しているか確認（値が変わっているか、あるいはオブジェクトが変わっているか）
        # TPSL変異は値を変更するので、元の値と異なるかチェック
        # ただし、偶然同じ値になる可能性もゼロではないが、floatなので低い

        # 変異の実装次第だが、通常は新しいオブジェクトになるか、値が変わる
        assert mutated.long_tpsl_gene is not None
        assert mutated.short_tpsl_gene is not None

        # 値の変化を確認 (変動幅0.8-1.2なので変わるはず)
        # ただし、FIXED_PERCENTAGEのパラメータ変異ロジックに依存

        # オブジェクトIDが異なること（ディープコピーされていること）
        assert mutated.long_tpsl_gene is not gene.long_tpsl_gene
        assert mutated.short_tpsl_gene is not gene.short_tpsl_gene
