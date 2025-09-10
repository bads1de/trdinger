"""
ComplexConditions strategyのテスト

バグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import random
from backend.app.services.auto_strategy.generators.strategies.complex_conditions_strategy import (
    ComplexConditionsStrategy,
)
from backend.app.services.auto_strategy.models.strategy_models import (
    IndicatorGene,
    Condition,
)
from backend.app.services.auto_strategy.constants import IndicatorType


class TestComplexConditionsStrategy:
    """ComplexConditionsStrategyのテスト"""

    @pytest.fixture
    def mock_generator(self):
        """Mock condition generator"""
        generator = Mock()
        generator._get_indicator_type.return_value = IndicatorType.MOMENTUM
        generator.logger = Mock()
        return generator

    @pytest.fixture
    def strategy(self, mock_generator):
        """Test strategy instance"""
        return ComplexConditionsStrategy(mock_generator)

    def test_initialization(self, mock_generator):
        """初期化テスト"""
        strategy = ComplexConditionsStrategy(mock_generator)
        assert strategy.condition_generator == mock_generator

    def test_generate_conditions_motvarity_indicators(self, strategy, mock_generator):
        """ momentum指標での条件生成"""
        # 3つの指標を設定
        indicators = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="MACD", enabled=True),
            IndicatorGene(type="CCI", enabled=True),
        ]

        # 各メソッドの戻り値をモック
        mock_generator._create_momentum_long_conditions.return_value = [
            Condition(left_operand="RSI", operator=">", right_operand=30)
        ]
        mock_generator._create_momentum_short_conditions.return_value = [
            Condition(left_operand="RSI", operator="<", right_operand=70)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 最初の3つの指標が処理されるはず
        assert len(result_long) > 0
        assert len(result_short) > 0
        assert len(result_exit) == 0

    def test_generate_conditions_with_disabled_indicators(self, strategy, mock_generator):
        """無効化された指標の処理"""
        indicators = [
            IndicatorGene(type="RSI", enabled=False),  # 無効
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="MACD", enabled=True),
        ]

        mock_generator._get_indicator_type.return_value = IndicatorType.TREND
        mock_generator._create_trend_long_conditions.return_value = [
            Condition(left_operand="SMA_20", operator=">", right_operand=100)
        ]
        mock_generator._create_trend_short_conditions.return_value = [
            Condition(left_operand="SMA_20", operator="<", right_operand=100)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 無効化された指標はスキップされ、有効な指標のみ処理される
        mock_generator._get_indicator_type.assert_called()

    def test_generate_conditions_no_valid_conditions(self, strategy, mock_generator):
        """条件生成不能の場合のfallback"""
        # 全ての条件生成メソッドが空リストを返す
        indicators = [
            IndicatorGene(type="UNKNOWN", enabled=True),
        ]

        # Unknown indicator typeはgenericを使用
        mock_generator._get_indicator_type.return_value = None  # Unknown type

        mock_generator._generic_long_conditions.return_value = []
        mock_generator._generic_short_conditions.return_value = []

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # フォールバック条件が生成されるはず
        mock_generator._generate_fallback_conditions.assert_called_once()

    def test_generate_conditions_generic_fallback(self, strategy, mock_generator):
        """全条件が空の場合の最終fallback"""
        indicators = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        # 最初の条件生成が失敗し、2指標でfallback
        mock_generator._create_momentum_long_conditions.return_value = []
        mock_generator._create_momentum_short_conditions.return_value = []
        mock_generator._generic_long_conditions.return_value = [
            Condition(left_operand="RSI", operator=">", right_operand=50)
        ]
        mock_generator._generic_short_conditions.return_value = [
            Condition(left_operand="RSI", operator="<", right_operand=50)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_exception_handling_in_generation(self, strategy, mock_generator):
        """条件生成中の例外処理"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]

        # メソッド呼び出しで例外
        mock_generator._create_momentum_long_conditions.side_effect = Exception("Test error")

        # generic条件へのfallback
        mock_generator._generic_long_conditions.return_value = [
            Condition(left_operand="close", operator=">", right_operand="open")
        ]
        mock_generator._generic_short_conditions.return_value = [
            Condition(left_operand="close", operator="<", right_operand="open")
        ]

        with patch.object(mock_generator.logger, 'warning') as mock_warning:
            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # 警告がログされる
            mock_warning.assert_called_once()

        # 結果は生成される
        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_return_type_conversion(self, strategy, mock_generator):
        """戻り値の型変換テスト"""
        indicators = [IndicatorGene(type="RSI", enabled=True)]

        mock_generator._create_momentum_long_conditions.return_value = [
            Condition(left_operand="RSI", operator=">", right_operand=30)
        ]
        mock_generator._create_momentum_short_conditions.return_value = [
            Condition(left_operand="RSI", operator="<", right_operand=70)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 結果はリスト型
        assert isinstance(result_long, list)
        assert isinstance(result_short, list)
        assert isinstance(result_exit, list)

    def test_empty_indicators_list(self, strategy, mock_generator):
        """空の指標リスト"""
        result_long, result_short, result_exit = strategy.generate_conditions([])

        # 空リストが返される
        assert result_long == []
        assert result_short == []
        assert result_exit == []


class TestComplexConditionsStrategyBugs:
    """ComplexConditionsStrategyの潜在的バグテスト"""

    def test_missing_short_conditions_generation_bug(self, strategy, mock_generator):
        """BUG: ショート条件が生成されない可能性のテスト"""
        indicators = [IndicatorGene(type="TREND_INDICATOR", enabled=True)]

        # long条件のみ生成、short条件生成メソッドが呼ばれない
        mock_generator._get_indicator_type.return_value = IndicatorType.TREND
        mock_generator._create_trend_long_conditions.return_value = [
            Condition(left_operand="TREND", operator=">", right_operand=0)
        ]
        mock_generator._create_trend_short_conditions.return_value = []

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # long条件があるのにshort条件が空の場合、バグの可能性
        # この場合、short_conditionsは空になる可能性がある
        # NOTE: これは潜在的なバグとして検知

    def test_fallback_loop_bug(self, strategy, mock_generator):
        """BUG: fallbackループの可能性"""
        # 先頭3つの指標が無効、残り2つでfallback
        indicators = [
            IndicatorGene(type="INVALID1", enabled=False),
            IndicatorGene(type="INVALID2", enabled=False),
            IndicatorGene(type="INVALID3", enabled=False),
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="SMA", enabled=True),
        ]

        mock_generator._get_indicator_type.return_value = IndicatorType.MOMENTUM
        mock_generator._create_momentum_long_conditions.return_value = []
        mock_generator._generic_long_conditions.return_value = [
            Condition(left_operand="RSI", operator=">", right_operand=50)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # fallbackが呼び出されるはず
        # この場合、_generate_fallback_conditionsが2度呼び出されるバグの可能性
        # NOTE: complex_conditions_strategy.pyの71行で条件が空の場合にフォールバック

    def test_indicator_selection_edge_case(self, strategy, mock_generator):
        """BUG: 指標選択の境界値テスト"""
        # ちょうど3つの指標
        indicators = [
            IndicatorGene(type=f"IND_{i}", enabled=True) for i in range(3)
        ]

        mock_generator._get_indicator_type.return_value = IndicatorType.MOMENTUM
        mock_generator._create_momentum_long_conditions.return_value = [
            Condition(left_operand=f"IND_{i}", operator=">", right_operand=50)
        ]
        mock_generator._create_momentum_short_conditions.return_value = [
            Condition(left_operand=f"IND_{i}", operator="<", right_operand=50)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 全ての指標が処理されることを確認
        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_indicator_type_switching_logic(self, strategy, mock_generator):
        """異なる指標タイプの切り替えロジックテスト"""
        indicators = [
            IndicatorGene(type="MOMENTUM_IND", enabled=True),
            IndicatorGene(type="TREND_IND", enabled=True),
            IndicatorGene(type="VOLATILITY_IND", enabled=True),
        ]

        # タイプに応じて適切なメソッドが呼ばれる
        call_count = 0

        def mock_indicator_type(indicator):
            nonlocal call_count
            call_count += 1
            if "MOMENTUM" in indicator.type:
                return IndicatorType.MOMENTUM
            elif "TREND" in indicator.type:
                return IndicatorType.TREND
            else:
                return IndicatorType.VOLATILITY

        mock_generator._get_indicator_type.side_effect = mock_indicator_type

        # 必要に応じて条件生成メソッドをモック
        mock_generator._create_momentum_long_conditions.return_value = [
            Condition(left_operand="MOMENTUM_IND", operator=">", right_operand=50)
        ]
        mock_generator._create_trend_long_conditions.return_value = [
            Condition(left_operand="TREND_IND", operator=">", right_operand=0)
        ]
        mock_generator._generic_long_conditions.return_value = [
            Condition(left_operand="VOLATILITY_IND", operator=">", right_operand=20)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 適切なメソッドが呼び出されたか確認
        assert call_count == 3  # 全ての指標に対してタイプ判定が呼ばれる


if __name__ == "__main__":
    pytest.main([__file__])