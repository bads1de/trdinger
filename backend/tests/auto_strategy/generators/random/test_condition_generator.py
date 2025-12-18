"""
ConditionGenerator (Random features) のテスト
"""

import pytest
from unittest.mock import Mock, patch
from app.services.auto_strategy.generators.condition_generator import (
    ConditionGenerator,
)
from app.services.auto_strategy.genes import Condition


class TestConditionGeneratorInit:
    """初期化のテスト"""

    def test_init_stores_config(self):
        """設定を保存する"""
        config = Mock()
        generator = ConditionGenerator(enable_smart_generation=False, ga_config=config)
        # ConditionGeneratorでは config は ga_config_obj として保存される
        assert generator.ga_config_obj == config
        assert generator.operand_generator is not None
        assert generator.available_operators is not None


class TestGenerateRandomConditions:
    """generate_random_conditionsのテスト"""

    @pytest.fixture
    def generator(self):
        config = Mock()
        # 条件数の範囲
        config.min_conditions = 1
        config.max_conditions = 3
        return ConditionGenerator(enable_smart_generation=False, ga_config=config)

    def test_generates_conditions_within_range(self, generator):
        """指定された範囲内の数の条件を生成"""
        indicators = [Mock()]
        condition_type = "entry"

        # random.randintをモックして常に2を返すようにする
        with patch("random.randint", return_value=2):
            with patch.object(generator, "_generate_single_condition") as mock_single:
                mock_single.return_value = Condition("close", ">", "SMA")

                conditions = generator.generate_random_conditions(
                    indicators, condition_type
                )

                assert len(conditions) == 2
                assert mock_single.call_count == 2

    def test_fallback_when_no_conditions_generated(self, generator):
        """条件が生成されなかった場合のフォールバック"""
        indicators = [Mock()]
        condition_type = "entry"

        # random.randintは1を返すが、_generate_single_conditionがNoneを返す
        with patch("random.randint", return_value=1):
            with patch.object(
                generator, "_generate_single_condition", return_value=None
            ):
                conditions = generator.generate_random_conditions(
                    indicators, condition_type
                )

                # フォールバック条件が1つ生成されるはず
                assert len(conditions) == 1
                assert conditions[0].left_operand == "close"
                assert conditions[0].right_operand == "SMA"

    def test_handles_inverted_min_max_config(self, generator):
        """min > maxの設定でも正しく動作する"""
        generator.ga_config_obj.min_conditions = 5
        generator.ga_config_obj.max_conditions = 2

        indicators = [Mock()]
        condition_type = "entry"

        with patch("random.randint") as mock_randint:
            mock_randint.return_value = 3
            with patch.object(
                generator,
                "_generate_single_condition",
                return_value=Condition("A", ">", "B"),
            ):
                generator.generate_random_conditions(indicators, condition_type)

                # randint(2, 5) が呼ばれることを確認（swapされているはず）
                mock_randint.assert_called_with(2, 5)


class TestGenerateSingleCondition:
    """_generate_single_conditionのテスト"""

    @pytest.fixture
    def generator(self):
        config = Mock()
        return ConditionGenerator(enable_smart_generation=False, ga_config=config)

    def test_generates_valid_condition(self, generator):
        """有効な条件を生成"""
        indicators = [Mock()]
        condition_type = "entry"

        # オペランド生成のモック
        generator.operand_generator.choose_operand = Mock(return_value="RSI")
        generator.operand_generator.choose_right_operand = Mock(return_value=70)

        # 演算子選択のモック
        with patch("random.choice", return_value=">"):
            condition = generator._generate_single_condition(indicators, condition_type)

            assert condition.left_operand == "RSI"
            assert condition.operator == ">"
            assert condition.right_operand == 70


class TestGenerateFallbackCondition:
    """_generate_fallback_conditionのテスト"""

    def test_entry_fallback(self):
        """エントリー用フォールバック"""
        generator = ConditionGenerator(enable_smart_generation=False, ga_config=Mock())
        condition = generator._generate_fallback_condition("entry")
        assert condition.operator == ">"

    def test_exit_fallback(self):
        """エグジット用フォールバック"""
        generator = ConditionGenerator(enable_smart_generation=False, ga_config=Mock())
        condition = generator._generate_fallback_condition("exit")
        assert condition.operator == "<"
