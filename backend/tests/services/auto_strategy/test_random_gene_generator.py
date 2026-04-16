from unittest.mock import patch

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.genes import Condition, ExitGene, IndicatorGene


class TestRandomGeneGenerator:
    def setup_method(self):
        self.config = GAConfig()
        self.generator = RandomGeneGenerator(self.config)

    def test_initialization(self):
        assert self.generator.config == self.config

    def test_generate_random_gene(self):
        gene = self.generator.generate_random_gene()
        assert gene is not None
        assert hasattr(gene, "indicators")
        assert hasattr(gene, "long_entry_conditions")
        assert hasattr(gene, "short_entry_conditions")
        assert hasattr(gene, "tpsl_gene")
        assert hasattr(gene, "position_sizing_gene")
        assert gene.id != ""
        assert "generated_by" in gene.metadata
        assert gene.metadata["generated_by"] == "RandomGeneGenerator"

    def test_generate_random_gene_populates_split_tpsl(self):
        """ランダム生成時にlong/short別のTPSL設定が生成されることを確認"""
        gene = self.generator.generate_random_gene()

        # 共通設定が生成されているか（既存ロジック）
        assert gene.tpsl_gene is not None
        assert gene.tpsl_gene.enabled is True

        # 新しい分離設定が生成されているか
        assert gene.long_tpsl_gene is not None
        assert gene.long_tpsl_gene.enabled is True

        assert gene.short_tpsl_gene is not None
        assert gene.short_tpsl_gene.enabled is True

        # IDが異なる（別オブジェクトである）ことを確認
        assert gene.long_tpsl_gene is not gene.short_tpsl_gene

    def test_generate_random_gene_populates_exit_logic(self):
        """ランダム生成時に exit_gene と方向別 exit 条件が生成されることを確認"""
        gene = self.generator.generate_random_gene()

        assert isinstance(gene.exit_gene, ExitGene)
        assert gene.exit_gene.enabled is True

        assert gene.long_exit_conditions
        assert gene.short_exit_conditions

        assert gene.long_exit_conditions is not gene.short_entry_conditions
        assert gene.short_exit_conditions is not gene.long_entry_conditions
        assert gene.long_exit_conditions[0] is not gene.short_entry_conditions[0]
        assert gene.short_exit_conditions[0] is not gene.long_entry_conditions[0]

    def test_generate_random_gene_uses_exit_specific_condition_generator(self):
        """exit 条件は entry 条件のコピーではなく専用生成器の出力を使う"""
        indicators = [
            IndicatorGene(
                id="ema123456789",
                type="EMA",
                parameters={"period": 20},
                enabled=True,
            )
        ]
        entry_long = [Condition(left_operand="close", operator=">", right_operand=1.0)]
        entry_short = [
            Condition(left_operand="close", operator="<", right_operand=-1.0)
        ]
        exit_long = [
            Condition(left_operand="close", operator="<", right_operand="exit_long")
        ]
        exit_short = [
            Condition(left_operand="close", operator=">", right_operand="exit_short")
        ]

        with (
            patch.object(
                self.generator, "_get_cached_indicators", return_value=indicators
            ),
            patch.object(
                self.generator.smart_condition_generator,
                "generate_balanced_conditions",
                return_value=(entry_long, entry_short, []),
            ),
            patch.object(
                self.generator.smart_condition_generator,
                "generate_exit_conditions",
                return_value=(exit_long, exit_short, []),
                create=True,
            ),
        ):
            gene = self.generator.generate_random_gene()

        assert gene.long_exit_conditions[0].right_operand == "exit_long"
        assert gene.short_exit_conditions[0].right_operand == "exit_short"
