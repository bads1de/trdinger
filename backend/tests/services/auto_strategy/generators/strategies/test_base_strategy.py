"""
ConditionStrategyベースクラスのユニットテスト。

TDDで開発:
1. ベースクラスが提供すべき共通ヘルパーメソッドのテスト
2. サブクラスで重複していたロジックがベースクラスに移動後も正しく動作することを確認
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from app.services.auto_strategy.generators.strategies.base_strategy import (
    ConditionStrategy,
)
from app.services.auto_strategy.genes import IndicatorGene, Condition, ConditionGroup
from app.services.auto_strategy.config.constants import IndicatorType


class ConcreteStrategy(ConditionStrategy):
    """テスト用の具象クラス"""

    def generate_conditions(self, indicators: List[IndicatorGene]):
        return [], [], []


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
    return ConcreteStrategy(mock_condition_generator)


class TestConditionStrategyBase:
    """ConditionStrategy 基本機能のテスト"""

    def test_init_stores_condition_generator(self, mock_condition_generator):
        """初期化時にcondition_generatorが保存されることを確認"""
        strategy = ConcreteStrategy(mock_condition_generator)
        assert strategy.condition_generator is mock_condition_generator

    def test_generate_conditions_is_abstract(self):
        """generate_conditionsが抽象メソッドであることを確認"""
        assert hasattr(ConditionStrategy.generate_conditions, "__isabstractmethod__")


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


class TestGetIndicatorConfig:
    """_get_indicator_config メソッドのテスト"""

    def test_returns_config_from_registry(self, strategy):
        """レジストリから指標設定を取得できることを確認"""
        with patch(
            "app.services.auto_strategy.generators.strategies.base_strategy.indicator_registry"
        ) as mock_registry:
            mock_config = Mock()
            mock_registry.get_indicator_config.return_value = mock_config

            config = strategy._get_indicator_config("RSI")

            mock_registry.get_indicator_config.assert_called_once_with("RSI")
            assert config is mock_config

    def test_returns_none_for_unknown_indicator(self, strategy):
        """不明な指標の場合はNoneを返すことを確認"""
        with patch(
            "app.services.auto_strategy.generators.strategies.base_strategy.indicator_registry"
        ) as mock_registry:
            mock_registry.get_indicator_config.return_value = None

            config = strategy._get_indicator_config("UNKNOWN_INDICATOR")

            assert config is None


class TestGetIndicatorType:
    """_get_indicator_type メソッドのテスト"""

    def test_delegates_to_condition_generator(self, strategy, sample_indicator):
        """condition_generatorに委譲することを確認"""
        result = strategy._get_indicator_type(sample_indicator)

        strategy.condition_generator._get_indicator_type.assert_called_once_with(
            sample_indicator
        )
        assert result == IndicatorType.MOMENTUM


class TestClassifyIndicatorsByType:
    """_classify_indicators_by_type メソッドのテスト"""

    def test_delegates_to_condition_generator(self, strategy, sample_indicator):
        """condition_generatorに委譲することを確認"""
        indicators = [sample_indicator]
        result = strategy._classify_indicators_by_type(indicators)

        strategy.condition_generator._dynamic_classify.assert_called_once_with(
            indicators
        )
        assert IndicatorType.TREND in result
        assert IndicatorType.MOMENTUM in result
        assert IndicatorType.VOLATILITY in result


class TestContextProperty:
    """context プロパティのテスト"""

    def test_returns_condition_generator_context(self, strategy):
        """condition_generator.contextを返すことを確認"""
        context = strategy.context
        assert context == {"timeframe": "1h", "symbol": "BTC/USDT"}

    def test_returns_empty_dict_when_no_context(self, mock_condition_generator):
        """contextがない場合は空辞書を返すことを確認"""
        mock_condition_generator.context = None
        strategy = ConcreteStrategy(mock_condition_generator)
        context = strategy.context
        assert context == {}


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


class TestCreateGenericConditions:
    """汎用条件生成ヘルパーのテスト"""

    def test_create_generic_long_conditions(self, strategy, sample_indicator):
        """ロング条件生成が委譲されることを確認"""
        result = strategy._create_generic_long_conditions(sample_indicator)

        strategy.condition_generator._generic_long_conditions.assert_called_once_with(
            sample_indicator
        )
        assert len(result) == 1
        assert result[0].left_operand == "RSI"
        assert result[0].operator == "<"
        assert result[0].right_operand == 30

    def test_create_generic_short_conditions(self, strategy, sample_indicator):
        """ショート条件生成が委譲されることを確認"""
        result = strategy._create_generic_short_conditions(sample_indicator)

        strategy.condition_generator._generic_short_conditions.assert_called_once_with(
            sample_indicator
        )
        assert len(result) == 1
        assert result[0].left_operand == "RSI"
        assert result[0].operator == ">"
        assert result[0].right_operand == 70
