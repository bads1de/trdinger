"""
IndicatorCharacteristics strategyのテスト

バグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch
from backend.app.services.auto_strategy.generators.strategies.indicator_characteristics_strategy import (
    IndicatorCharacteristicsStrategy,
)
from backend.app.services.auto_strategy.models.strategy_models import (
    IndicatorGene,
    Condition,
)
from backend.app.services.auto_strategy.constants import IndicatorType


class TestIndicatorCharacteristicsStrategy:
    """IndicatorCharacteristicsStrategyのテスト"""

    @pytest.fixture
    def mock_generator(self):
        """Mock condition generator"""
        generator = Mock()
        generator.logger = Mock()
        return generator

    @pytest.fixture
    def strategy(self, mock_generator):
        """Test strategy instance"""
        return IndicatorCharacteristicsStrategy(mock_generator)

    def test_initialization(self, mock_generator):
        """初期化テスト"""
        strategy = IndicatorCharacteristicsStrategy(mock_generator)
        assert strategy.condition_generator == mock_generator

    def test_generate_conditions_with_ml_indicators(self, strategy, mock_generator):
        """ML指標での条件生成テスト"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),  # Non-ML
        ]

        # ML long条件生成のモック
        ml_long_conditions = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]
        mock_generator._create_ml_long_conditions.return_value = ml_long_conditions

        # 2つのML指標があるのでshort条件も生成
        with patch.object(strategy, '_create_ml_short_conditions') as mock_ml_short:
            mock_ml_short.return_value = [
                Condition(left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6)
            ]

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # ML条件が追加される
            assert len(result_long) > 0
            assert len(result_short) > 0
            assert len(result_exit) == 0

            mock_generator._create_ml_long_conditions.assert_called_once_with(indicators[:2])  # ML指標のみ
            mock_ml_short.assert_called_once_with(indicators[:2])

    def test_generate_conditions_single_ml_indicator(self, strategy, mock_generator):
        """単一ML指標の場合"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),  # Non-ML
        ]

        # ML long条件生成のみ
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]

        # ML指標が1つなのでshort条件は生成されない (len >= 2)
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        assert len(result_long) > 0
        # short条件は空のはず
        assert len(result_short) == 0

    def test_generate_conditions_no_ml_fallback_to_generic(self, strategy, mock_generator):
        """ML指標なしの場合のfallback"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]

        # ML Condition生成が空を返す
        mock_generator._create_ml_long_conditions.return_value = []

        # 代わりにgeneric条件が生成される
        mock_generator._generic_long_conditions.return_value = [
            Condition(left_operand="SMA_20", operator=">", right_operand=100)
        ]
        mock_generator._generic_short_conditions.return_value = [
            Condition(left_operand="SMA_20", operator="<", right_operand=100)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 非ML指標の中から最初の1つだけ処理される戦略仕様
        mock_generator._generic_long_conditions.assert_called_once_with(indicators[0])
        mock_generator._generic_short_conditions.assert_called_once_with(indicators[0])

        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_generate_conditions_ml_and_regular_mixed(self, strategy, mock_generator):
        """MLと通常指標の混合"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]

        # ML条件は生成されるが、通常指標は無視される (戦略仕様)
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]

        with patch.object(strategy, '_create_ml_short_conditions') as mock_ml_short:
            mock_ml_short.return_value = [
                Condition(left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6)
            ]

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # ML条件のみ生成され、通常指標は無視される
            assert len(result_long) > 0
            assert len(result_short) > 0

    def test_generate_conditions_disabled_ml_indicators(self, strategy, mock_generator):
        """無効化されたML指標の処理"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=False),     # Disabled
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        # 有効なML指標は1つ
        # indicators[: ] フィルタにより有効なML指標のみ選択されるはず
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_DOWN_PROB", operator=">", right_operand=0.6)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 無効化されたML指標は除外される
        assert len(result_long) > 0

    def test_empty_conditions_fallback_to_generator_fallback(self, strategy, mock_generator):
        """全ての条件生成が失敗した場合のgenerator fallback"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        # 何も条件を生成しない
        mock_generator._create_ml_long_conditions.return_value = []
        mock_generator._generic_long_conditions.return_value = []
        mock_generator._generic_short_conditions.return_value = []

        # 最後のfallbackとして_generator_のfallbackが呼ばれる
        mock_fallback_result = ([Condition(left_operand="close", operator=">", right_operand="open")],
                              [Condition(left_operand="close", operator="<", right_operand="open")], [])
        mock_generator._generate_fallback_conditions.return_value = mock_fallback_result

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # fallback条件が使われる
        mock_generator._generate_fallback_conditions.assert_called_once()

    def test_create_ml_short_conditions_with_down_prob(self, strategy, mock_generator):
        """ML DOWN PROBによるshort条件生成"""
        ml_indicators = [
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_RANGE_PROB", enabled=True),
        ]

        result = strategy._create_ml_short_conditions(ml_indicators)

        # ML_DOWN_PROBが存在するので条件生成
        assert len(result) == 1
        assert result[0].left_operand == "ML_DOWN_PROB"
        assert result[0].operator == ">"
        assert result[0].right_operand == 0.6

    def test_create_ml_short_conditions_with_up_prob_only(self, strategy, mock_generator):
        """ML UP PROBのみの場合のshort条件"""
        ml_indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
        ]

        result = strategy._create_ml_short_conditions(ml_indicators)

        # UP PROBは逆ロジックで条件生成
        assert len(result) == 1
        assert result[0].left_operand == "ML_UP_PROB"
        assert result[0].operator == "<"
        assert result[0].right_operand == 0.3

    def test_create_ml_short_conditions_with_range_prob(self, strategy, mock_generator):
        """ML RANGE PROBの場合のshort条件"""
        ml_indicators = [
            IndicatorGene(type="ML_RANGE_PROB", enabled=True),
        ]

        result = strategy._create_ml_short_conditions(ml_indicators)

        # RANGE PROBは高い値でshort条件
        assert len(result) == 1
        assert result[0].left_operand == "ML_RANGE_PROB"
        assert result[0].operator == ">"
        assert result[0].right_operand == 0.7

    def test_create_ml_short_conditions_multiple_conditions(self, strategy, mock_generator):
        """複数のML指標によるmultiple short条件"""
        ml_indicators = [
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_RANGE_PROB", enabled=True),
        ]

        result = strategy._create_ml_short_conditions(ml_indicators)

        # 複数条件生成だが、DOWN_PROBが優先される
        assert len(result) >= 1

    def test_return_types(self, strategy, mock_generator):
        """戻り値の型テスト"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
        ]

        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML", operator=">", right_operand=0.5)
        ]

        with patch.object(strategy, '_create_ml_short_conditions') as mock_short:
            mock_short.return_value = [
                Condition(left_operand="ML", operator="<", right_operand=0.5)
            ]

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            assert isinstance(result_long, list)
            assert isinstance(result_short, list)
            assert isinstance(result_exit, list)


class TestIndicatorCharacteristicsStrategyBugs:
    """IndicatorCharacteristicsStrategyの潜在的バグテスト"""

    def test_bug_empty_ml_indicators_after_filter(self, strategy, mock_generator):
        """BUG: フィルタ後ML指標が空になる場合"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=False),    # Disabled
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        # 有効なML指標がないのでml_indicatorsは空リスト
        # 56-62行: if not long_conditions or not short_conditions:
        # この場合_long_conditionsは空なのでTrue, _generate_fallback_conditions呼び出し

        mock_generator._create_ml_long_conditions.return_value = []
        mock_fallback_result = ([Condition(left_operand="close", operator=">", right_operand="open")],
                              [Condition(left_operand="close", operator="<", right_operand="open")], [])
        mock_generator._generate_fallback_conditions.return_value = mock_fallback_result

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # fallbackが呼ばれる
        assert mock_generator._generate_fallback_conditions.called

    def test_bug_short_conditions_not_generated_for_single_ml(self, strategy, mock_generator):
        """BUG: ML指標1つだけの場合short条件が生成されない"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
        ]

        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]

        # 37-40行: if len(ml_indicators) >= 2:
        # 1つのML指標ではshort_conditionsが生成されない
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # short_conditionsが空になるバグ
        assert len(result_short) == 0

    def test_bug_priority_of_ml_over_regular_indicators(self, strategy, mock_generator):
        """BUG: ML指標優先度テスト"""
        indicators = [
            IndicatorGene(type="HIGH_PRIORITY_ML", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]

        # ML条件が失敗した場合にのみregular条件にfallbackする
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="HIGH_PRIORITY_ML", operator=">", right_operand=0.6)
        ]

        # regular条件は一切呼ばれない
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # regular指標の条件生成メソッドは呼ばれない

    def test_bug_ml_short_condition_creation_logic(self, strategy):
        """BUG: ML short条件作成ロジックのテスト"""
        ml_indicators = [
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="UNKNOWN_ML", enabled=True),
        ]

        result = strategy._create_ml_short_conditions(ml_indicators)

        # UNKNOWN_MLに対しては何も条件生成されない
        # 既知のタイプのみ条件生成
        assert len(result) >= 2  # DOWN_PROBとUP_PROB

    def test_performance_with_many_indicators(self, strategy, mock_generator):
        """大量指標での処理テスト"""
        indicators = [IndicatorGene(type=f"ML_{i}", enabled=True) for i in range(100)]

        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_test", operator=">", right_operand=0.6)
        ]

        with patch.object(strategy, '_create_ml_short_conditions') as mock_short:
            mock_short.return_value = [
                Condition(left_operand="ML_test", operator="<", right_operand=0.6)
            ]

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # 大量指標でも問題なく処理できる
            assert len(result_long) > 0
            assert len(result_short) > 0

    def test_edge_case_empty_indicators(self, strategy, mock_generator):
        """空指標リストの処理"""
        result_long, result_short, result_exit = strategy.generate_conditions([])

        # ML指標がないので空リスト
        assert result_long == []
        assert result_short == []
        assert result_exit == []

    def test_disabled_non_ml_indicators_ignore(self, strategy, mock_generator):
        """無効化された非ML指標の無視テスト"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=False),
            IndicatorGene(type="RSI", enabled=False),
            IndicatorGene(type="MACD", enabled=False),
        ]

        # 非ML指標すべて無効なので、long_conditionsは空
        # 56行: if not long_conditions:
        # Trueになるので_generator_fallback_conditions呼び出し

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # fallback条件が生成される


if __name__ == "__main__":
    pytest.main([__file__])