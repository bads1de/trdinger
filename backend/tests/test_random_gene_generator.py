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

    def test_generate_complex_fallback_conditions_long(self):
        """複雑なフォールバック条件生成（ロング）をテスト"""
        indicators = [
            type('MockIndicator', (), {'type': 'SMA', 'enabled': True})(),
            type('MockIndicator', (), {'type': 'EMA', 'enabled': True})()
        ]
        conditions = self.generator._generate_complex_fallback_conditions("long", indicators)

        assert len(conditions) >= 2  # 最低2つの条件を生成
        for condition in conditions:
            assert hasattr(condition, 'left_operand')
            assert hasattr(condition, 'operator')
            assert hasattr(condition, 'right_operand')
            # シンプルなclose > trend_name ではないことを確認
            assert not (condition.left_operand == "close" and
                       condition.operator == ">" and
                       isinstance(condition.right_operand, str) and
                       condition.right_operand in ["SMA", "EMA"])

    def test_is_simple_price_comparison_true(self):
        """シンプルな価格比較を正しく判定することをテスト"""
        from backend.app.services.auto_strategy.models.strategy_models import Condition

        simple_condition = Condition(left_operand="close", operator=">", right_operand="open")
        result = self.generator._is_simple_price_comparison(simple_condition)
        assert result is True

    def test_is_simple_price_comparison_false_for_indicator(self):
        """指標との比較がシンプルでないことをテスト"""
        from backend.app.services.auto_strategy.models.strategy_models import Condition

        indicator_condition = Condition(left_operand="SMA", operator=">", right_operand="close")
        result = self.generator._is_simple_price_comparison(indicator_condition)
        assert result is False

    def test_ensure_or_with_fallback_avoids_simple_conditions(self):
        """_ensure_or_with_fallbackがシンプルな条件を避けることをテスト"""
        from backend.app.services.auto_strategy.models.strategy_models import Condition

        # シンプルな条件のみのリスト
        simple_conditions = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]

        indicators = [
            type('MockIndicator', (), {'type': 'SMA', 'enabled': True})()
        ]

        result = self.generator._ensure_or_with_fallback(simple_conditions, "long", indicators)

        # 結果はConditionGroupを含むリスト
        assert len(result) == 1
        assert hasattr(result[0], 'conditions')
        # ConditionGroupの中に複数の条件があるはず
        conditions_in_group = result[0].conditions
        assert len(conditions_in_group) > 1
        # シンプルな条件以外に複雑な条件が追加されているはず
        complex_conditions = [c for c in conditions_in_group if not self.generator._is_simple_price_comparison(c)]
        assert len(complex_conditions) > 0