import pytest
from backend.app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from backend.app.services.auto_strategy.config.ga import GASettings


class TestRandomGeneGenerator:
    def setup_method(self):
        self.config = GASettings()
        self.generator = RandomGeneGenerator(self.config, enable_smart_generation=True)

    def test_initialization(self):
        assert self.generator.config == self.config

    def test_generate_random_gene(self):
        gene = self.generator.generate_random_gene()
        assert gene is not None
        assert hasattr(gene, 'indicators')
        assert hasattr(gene, 'long_entry_conditions')
        assert hasattr(gene, 'short_entry_conditions')
        assert hasattr(gene, 'tpsl_gene')
        assert hasattr(gene, 'position_sizing_gene')