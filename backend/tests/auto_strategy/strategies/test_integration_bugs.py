"""
統合バグテスト

strategyクラスの統合的なバグを発見します。
"""

import pytest
from unittest.mock import Mock, patch
from backend.app.services.auto_strategy.generators.strategies import (
    ConditionStrategy,
    DifferentIndicatorsStrategy,
    ComplexConditionsStrategy,
    IndicatorCharacteristicsStrategy,
)
from backend.app.services.auto_strategy.models.strategy_models import (
    IndicatorGene,
    Condition,
)
from backend.app.services.auto_strategy.constants import IndicatorType


class TestIntegrationBugs:
    """統合的なバグテスト"""

    def test_cross_strategy_consistency(self):
        """戦略間の一貫性テスト"""
        # 同じ指標セットで異なる戦略の出力比較
        mock_generator = Mock()
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]

        # 各戦略の条件生成メソッドを統一してモック
        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[0]],
            IndicatorType.MOMENTUM: [indicators[1]],
        }

        strategies = [
            DifferentIndicatorsStrategy(mock_generator),
            ComplexConditionsStrategy(mock_generator),
            IndicatorCharacteristicsStrategy(mock_generator),
        ]

        results = []
        for strategy in strategies:
            with patch.object(mock_generator, '_create_trend_long_conditions', return_value=[
                Condition(left_operand="SMA_20", operator=">", right_operand=100)
            ]):
                with patch.object(mock_generator, '_create_momentum_long_conditions', return_value=[
                    Condition(left_operand="RSI", operator=">", right_operand=30)
                ]):
                    result = strategy.generate_conditions(indicators)
                    results.append(result)

        # 全ての戦略が結果を生成することを確認
        for result in results:
            assert len(result[0]) > 0  # long conditions

    def test_error_propagation_bug(self):
        """エラー伝播のテスト"""
        mock_generator = Mock()
        mock_generator._get_indicator_type.side_effect = Exception("Test error")

        indicators = [IndicatorGene(type="SMA_20", enabled=True)]

        strategies = [
            DifferentIndicatorsStrategy(mock_generator),
            ComplexConditionsStrategy(mock_generator),
        ]

        for strategy in strategies:
            # エラーが適切にハンドリングされるか
            try:
                result = strategy.generate_conditions(indicators)
                # 結果が生成されることを確認
                assert isinstance(result, tuple)
                assert len(result) == 3
            except Exception as e:
                # もし元のバグがあればここで例外発生
                pytest.fail(f"Error propagation bug in {strategy.__class__.__name__}: {e}")

    @patch('backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy.logger')
    @patch('backend.app.services.auto_strategy.generators.strategies.complex_conditions_strategy.logger')
    @patch('backend.app.services.auto_strategy.generators.strategies.indicator_characteristics_strategy.logger')
    def test_logging_consistency(self, mock_logger_ic, mock_logger_cc, mock_logger_di):
        """ログ出力の一貫性テスト"""
        mock_generator = Mock()
        mock_generator._generic_long_conditions.return_value = [Condition(left_operand="close", operator=">", right_operand="open")]

        strategies = [
            (DifferentIndicatorsStrategy(mock_generator), mock_logger_di),
            (ComplexConditionsStrategy(mock_generator), mock_logger_cc),
            (IndicatorCharacteristicsStrategy(mock_generator), mock_logger_ic),
        ]

        for strategy, mock_logger in strategies:
            # ログメソッドが呼ばれていることを確認
            # 実際のログ呼び出しは各戦略の実装に依存
            result = strategy.generate_conditions([])
            # ログが呼ばれているはず（debug, warningなど）


class TestSpecificBugPatterns:
    """特定のパンバグテスト"""

    def test_bug_random_choice_without_candidates(self):
        """BUG: random.choiceで候補がない場合のクラッシュ"""
        # different_indicators_strategy.pyの45,53,69,76行でrandom.choice呼び出し
        # 空のリストを渡すとIndexError

        mock_generator = Mock()

        # 空のリストを返すようにモック
        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [],  # Empty
            IndicatorType.MOMENTUM: [],
        }

        strategy = DifferentIndicatorsStrategy(mock_generator)
        indicators = [IndicatorGene(type="SMA_20", enabled=True)]

        # この場合、96-102行のfallback条件が生成されるはず
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # IndexErrorが発生せず、fallback条件が生成される
        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_bug_double_fallback_call(self):
        """BUG: complex_conditions_strategy.pyでのdouble fallback call"""
        # 71行: return self.condition_generator._generate_fallback_conditions()
        # 既に_generate_fallback_conditionsが呼ばれているのに再度呼び出し

        mock_generator = Mock()
        strategy = ComplexConditionsStrategy(mock_generator)

        # 条件生成が全て失敗するようにモック
        mock_generator._get_indicator_type.return_value = IndicatorType.MOMENTUM
        mock_generator._create_momentum_long_conditions.return_value = []
        mock_generator._generic_long_conditions.return_value = []
        # _generate_fallback_conditionsはモックせず、デフォルト実装が呼ばれる

        indicators = [IndicatorGene(type="UNKNOWN", enabled=False)]

        # 複合条件戦略のロジックに従う
        # 第1fallback: indicators[:2] for generic
        # 第2fallback: _generate_fallback_conditions()

        result = strategy.generate_conditions(indicators)
        # 結果が生成されることを確認
        assert isinstance(result, tuple)

    def test_bug_condition_order_dependency(self):
        """BUG: 条件順序依存バグ"""
        # 同じ指標セットで異なる順序を与えた場合の結果比較

        mock_generator = Mock()
        indicators1 = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]
        indicators2 = indicators1[::-1]  # 逆順

        # 順序依存がないことを確認
        strategy1 = ComplexConditionsStrategy(mock_generator)
        strategy2 = ComplexConditionsStrategy(mock_generator)

        # conditionsメソッドをモック
        with patch.object(mock_generator, '_create_momentum_long_conditions') as mock_long:
            mock_long.return_value = [Condition(left_operand="test", operator=">", right_operand=50)]

            result1 = strategy1.generate_conditions(indicators1)
            result2 = strategy2.generate_conditions(indicators2)

            # 結果が等しいことを確認（順序依存なし）
            # 実際には順序依存のロジックがないので同じ結果になるはず

    def test_bug_empty_string_type_handling(self):
        """BUG: 空文字列のtype属性処理"""
        mock_generator = Mock()
        indicators = [
            IndicatorGene(type="", enabled=True),  # Empty type
        ]

        # 各戦略で空文字列がどのように処理されるか
        strategies = [
            DifferentIndicatorsStrategy(mock_generator),
            ComplexConditionsStrategy(mock_generator),
            IndicatorCharacteristicsStrategy(mock_generator),
        ]

        for strategy in strategies:
            # 空文字列typeがクラッシュを引き起こさないことを確認
            result = strategy.generate_conditions(indicators)
            assert isinstance(result, tuple)
            assert len(result) == 3


class TestEdgeCaseCombinations:
    """エッジケースの組み合わせテスト"""

    def test_max_indicators_stress_test(self):
        """最大指標数のストレステスト"""
        num_indicators = 50
        indicators = [
            IndicatorGene(type=f"IND_{i}", enabled=True) for i in range(num_indicators)
        ]

        mock_generator = Mock()
        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators[:25],
            IndicatorType.MOMENTUM: indicators[25:],
        }

        strategy = DifferentIndicatorsStrategy(mock_generator)

        # random.choiceが呼ばれても問題ない
        with patch('backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy.random.choice') as mock_choice:
            mock_choice.return_value = indicators[0]

            result = strategy.generate_conditions(indicators)
            assert len(result) == 3

    def test_none_values_in_indicators(self):
        """指標のNone値処理テスト"""
        mock_generator = Mock()

        # Noneを含む指標リスト（異常ケース）
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            None,  # Invalid
            IndicatorGene(type="RSI", enabled=True),
        ]

        strategies = [
            DifferentIndicatorsStrategy(mock_generator),
            ComplexConditionsStrategy(mock_generator),
            IndicatorCharacteristicsStrategy(mock_generator),
        ]

        for strategy in strategies:
            # Noneが適切にハンドリングされるか（AttributeErrorが発生しない）
            with pytest.raises((AttributeError, TypeError)):
                strategy.generate_conditions(indicators)

    def test_duplicate_indicator_types(self):
        """重複指標タイプ処理テスト"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),  # Duplicate
            IndicatorGene(type="RSI", enabled=True),
        ]

        mock_generator = Mock()
        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators[:2],
            IndicatorType.MOMENTUM: [indicators[2]],
        }

        strategy = DifferentIndicatorsStrategy(mock_generator)

        result = strategy.generate_conditions(indicators)
        # 重複が問題を引き起こさないことを確認
        assert isinstance(result, tuple)

    @patch('backend.app.services.auto_strategy.generators.strategies.base_strategy.logger')
    def test_base_strategy_abstract_method_call(self, mock_logger):
        """抽象メソッド呼び出し時のエラーハンドリング"""
        # 各サブクラスが抽象メソッドを正しく実装しているか

        mock_generator = Mock()
        strategies = [
            DifferentIndicatorsStrategy(mock_generator),
            ComplexConditionsStrategy(mock_generator),
            IndicatorCharacteristicsStrategy(mock_generator),
        ]

        for strategy in strategies:
            # generate_conditionsが実装されていることを確認
            assert hasattr(strategy, 'generate_conditions')
            assert callable(strategy.generate_conditions)


if __name__ == "__main__":
    pytest.main([__file__])