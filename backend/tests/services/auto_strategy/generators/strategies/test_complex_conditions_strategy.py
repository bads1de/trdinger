"""
ComplexConditionsStrategyのユニットテスト。
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from app.services.auto_strategy.generators.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from app.services.auto_strategy.genes import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.config.constants import IndicatorType


@pytest.fixture
def mock_condition_generator():
    """モック ConditionGenerator フィクスチャ"""
    generator = MagicMock()
    generator.context = {"timeframe": "1h", "symbol": "BTC/USDT"}
    generator._get_indicator_name.side_effect = lambda i: f"{i.type}_{i.id[:8]}" if i.id else i.type
    generator._structure_conditions.side_effect = lambda x: x
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
        strategy.gen._structure_conditions.return_value = []
        result = strategy._structure_conditions([])
        assert result == []

    def test_returns_single_condition_unchanged(self, strategy):
        """単一条件の場合はそのまま返すことを確認"""
        condition = Condition(left_operand="RSI", operator="<", right_operand=30)
        strategy.gen._structure_conditions.return_value = [condition]
        result = strategy._structure_conditions([condition])
        assert result == [condition]

    def test_multiple_conditions_can_be_grouped(self, strategy):
        """複数条件がグループ化（委譲）されることを確認"""
        conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="MACD", operator=">", right_operand=0),
        ]
        
        expected_group = [ConditionGroup(operator="OR", conditions=conditions)]
        # side_effect をクリアして return_value が使われるようにする
        strategy.gen._structure_conditions.side_effect = None
        strategy.gen._structure_conditions.return_value = expected_group
        
        result = strategy._structure_conditions(conditions)
        assert result == expected_group

    def test_grouping_uses_or_operator(self, strategy):
        """グループ化（委譲）時に正しい結果が得られることを確認"""
        conditions = [
            Condition(left_operand="RSI", operator="<", right_operand=30),
            Condition(left_operand="MACD", operator=">", right_operand=0),
        ]
        
        expected_group = [ConditionGroup(operator="OR", conditions=conditions)]
        strategy.gen._structure_conditions.side_effect = None
        strategy.gen._structure_conditions.return_value = expected_group
        
        result = strategy._structure_conditions(conditions)
        assert result[0].operator == "OR"

