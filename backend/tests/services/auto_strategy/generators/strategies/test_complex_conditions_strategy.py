"""
ComplexConditionsStrategyのユニットテスト。
"""

import pytest
from unittest.mock import Mock
from typing import List

from app.services.auto_strategy.generators.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.genes import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.config.constants import IndicatorType


@pytest.fixture
def mock_condition_generator():
    """モック ConditionGenerator フィクスチャ"""
    generator = Mock()
    generator.context = {"timeframe": "1h", "symbol": "BTC/USDT"}
    generator._dynamic_classify = Mock(
        return_value={
            IndicatorType.TREND: [],
            IndicatorType.MOMENTUM: [],
            IndicatorType.VOLATILITY: [],
        }
    )
    generator._generic_long_conditions = Mock(
        return_value=[Condition(left_operand="RSI", operator="<", right_operand=30)]
    )
    generator._generic_short_conditions = Mock(
        return_value=[Condition(left_operand="RSI", operator=">", right_operand=70)]
    )
    generator._get_indicator_type = Mock(return_value=IndicatorType.MOMENTUM)
    return generator


@pytest.fixture
def sample_indicator():
    """サンプル指標フィクスチャ"""
    return IndicatorGene(
        id="test-id-12345678",
        type="RSI",
        parameters={"period": 14},
        enabled=True,
    )


@pytest.fixture
def strategy(mock_condition_generator):
    """テスト対象の戦略インスタンス"""
    return ComplexConditionsStrategy(mock_condition_generator)


class TestGetIndicatorName:
    """_get_indicator_name メソッドのテスト"""

    def test_returns_name_with_id_suffix(self, strategy, sample_indicator):
        """指標名にIDサフィックスが付与されることを確認"""
        name = strategy._get_indicator_name(sample_indicator)
        assert name == "RSI_test-id-"

    def test_returns_name_without_id_when_none(self, strategy):
        """IDがNoneの場合、指標タイプのみが返されることを確認"""
        indicator = IndicatorGene(
            id=None,
            type="SMA",
            parameters={"period": 20},
            enabled=True,
        )
        name = strategy._get_indicator_name(indicator)
        assert name == "SMA"

    def test_returns_name_with_empty_id(self, strategy):
        """IDが空文字の場合、指標タイプのみが返されることを確認"""
        indicator = IndicatorGene(
            id="",
            type="EMA",
            parameters={"period": 20},
            enabled=True,
        )
        name = strategy._get_indicator_name(indicator)
        assert name == "EMA"


class TestStructureConditions:
    """_structure_conditions メソッドのテスト"""

    def test_returns_empty_list_for_empty_input(self, strategy):
        """空リストの場合は空リストを返すことを確認"""
        result = strategy._structure_conditions([])
        assert result == []

    def test_returns_single_condition_unchanged(self, strategy):
        """単一条件の場合はそのまま返すことを確認"""
        condition = Condition(left_operand="RSI", operator="<", right_operand=30)
        result = strategy._structure_conditions([condition])
        assert result == [condition]

    def test_multiple_conditions_can_be_grouped(self, strategy):
        """複数条件が確率的にグループ化されることを確認"""
        conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="MACD", operator=">", right_operand=0),
            Condition(left_operand="SMA_20", operator="<", right_operand="Close"),
        ]

        # 複数回実行して、グループ化が発生することを確認
        grouped_count = 0
        for _ in range(100):
            result = strategy._structure_conditions(conditions.copy())
            for item in result:
                if isinstance(item, ConditionGroup):
                    grouped_count += 1
                    break

        # 30%の確率でグループ化されるので、100回中少なくとも数回はグループ化されるはず
        assert grouped_count > 0, "グループ化が一度も発生しなかった"

    def test_grouping_uses_or_operator(self, strategy):
        """グループ化時にOR演算子が使用されることを確認"""
        conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="MACD", operator=">", right_operand=0),
        ]

        # グループ化が発生するまで実行
        for _ in range(100):
            result = strategy._structure_conditions(conditions.copy())
            for item in result:
                if isinstance(item, ConditionGroup):
                    assert item.operator == "OR"
                    assert len(item.conditions) == 2
                    return

        pytest.fail("グループ化が発生しなかった")
