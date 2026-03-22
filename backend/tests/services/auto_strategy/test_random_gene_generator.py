from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


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
