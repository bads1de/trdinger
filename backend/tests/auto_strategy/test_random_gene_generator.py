from backend.app.services.auto_strategy.config.ga import GASettings
from backend.app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


class TestRandomGeneGenerator:
    def setup_method(self):
        self.config = GASettings()
        self.generator = RandomGeneGenerator(self.config, enable_smart_generation=True)

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

    def test_ensure_or_with_fallback_basic(self):
        """_ensure_or_with_fallbackの基本テスト"""
        from backend.app.services.auto_strategy.models.strategy_models import Condition

        # シンプルな条件のみのリスト
        simple_conditions = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]

        indicators = [type("MockIndicator", (), {"type": "SMA", "enabled": True})()]

        result = self.generator._ensure_or_with_fallback(
            simple_conditions, "long", indicators
        )

        # 結果がリストであること
        assert isinstance(result, list)
        # 単一条件の場合は追加条件が含まれる可能性
        assert len(result) >= 1
