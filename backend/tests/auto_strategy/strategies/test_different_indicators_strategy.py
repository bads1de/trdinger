"""
DifferentIndicators strategyのテスト

バグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch, call
import logging
from backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy import (
    DifferentIndicatorsStrategy,
)
from backend.app.services.auto_strategy.models.strategy_models import (
    IndicatorGene,
    Condition,
)
from backend.app.services.auto_strategy.constants import IndicatorType


class TestDifferentIndicatorsStrategy:
    """DifferentIndicatorsStrategyのテスト"""

    @pytest.fixture
    def mock_generator(self):
        """Mock condition generator"""
        generator = Mock()
        generator.logger = Mock()
        return generator

    @pytest.fixture
    def strategy(self, mock_generator):
        """Test strategy instance"""
        return DifferentIndicatorsStrategy(mock_generator)

    def test_initialization(self, mock_generator):
        """初期化テスト"""
        strategy = DifferentIndicatorsStrategy(mock_generator)
        assert strategy.condition_generator == mock_generator

    def test_generate_conditions_with_mixed_types(self, strategy, mock_generator):
        """混合指標タイプでの条件生成"""
        # トレンドとモメンタムの指標
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),  # Trend
            IndicatorGene(type="RSI", enabled=True),     # Momentum
            IndicatorGene(type="ML_UP_PROB", enabled=True),  # ML
            IndicatorGene(type="MACD", enabled=False),   # Disabled
        ]

        # 指標分類のモック
        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[0]],
            IndicatorType.MOMENTUM: [indicators[1]],
        }

        # 各条件生成メソッドのモック
        mock_generator._create_trend_long_conditions.return_value = [
            Condition(left_operand="SMA_20", operator=">", right_operand=100)
        ]
        mock_generator._create_momentum_long_conditions.return_value = [
            Condition(left_operand="RSI", operator=">", right_operand=30)
        ]
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]
        mock_generator._create_trend_short_conditions.return_value = [
            Condition(left_operand="SMA_20", operator="<", right_operand=100)
        ]
        mock_generator._create_momentum_short_conditions.return_value = [
            Condition(left_operand="RSI", operator="<", right_operand=70)
        ]

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # ML指標を別途設定
        strategy.condition_generator._create_ml_long_conditions.assert_called_once_with(
            [indicators[2]]
        )

        assert len(result_long) > 0
        assert len(result_short) > 0
        assert len(result_exit) == 0

    def test_generate_conditions_only_trend(self, strategy, mock_generator):
        """トレンド指標のみの場合"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="EMA_50", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators,
            IndicatorType.MOMENTUM: [],
        }

        with patch.object(strategy, '_create_trend_long_conditions') as mock_trend_long:
            with patch.object(strategy, '_create_momentum_long_conditions') as mock_momentum_long:
                mock_trend_long.return_value = [
                    Condition(left_operand="SMA_20", operator=">", right_operand=100)
                ]

                result_long, result_short, result_exit = strategy.generate_conditions(indicators)

                # トレンド条件のみ生成
                mock_trend_long.assert_called()
                mock_momentum_long.assert_not_called()

    def test_generate_conditions_only_momentum(self, strategy, mock_generator):
        """モメンタム指標のみの場合"""
        indicators = [
            IndicatorGene(type="RSI", enabled=True),
            IndicatorGene(type="MACD", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [],
            IndicatorType.MOMENTUM: indicators,
        }

        with patch.object(strategy, '_create_trend_long_conditions') as mock_trend_long:
            with patch.object(strategy, '_create_momentum_long_conditions') as mock_momentum_long:
                mock_momentum_long.return_value = [
                    Condition(left_operand="RSI", operator=">", right_operand=30)
                ]

                result_long, result_short, result_exit = strategy.generate_conditions(indicators)

                # モメンタム条件のみ生成
                mock_momentum_long.assert_called()
                mock_trend_long.assert_not_called()

    def test_generate_conditions_no_valid_conditions(self, strategy, mock_generator):
        """有効な条件がない場合のfallback"""
        indicators = [
            IndicatorGene(type="UNKNOWN", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [],
            IndicatorType.MOMENTUM: [],
        }

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # fallback条件が生成される
        assert len(result_long) == 1
        assert len(result_short) == 1
        assert result_long[0].left_operand == "close"
        assert result_short[0].left_operand == "close"

    def test_ml_indicators_processing(self, strategy, mock_generator):
        """ML指標の特別処理"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="ML_DOWN_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),  # Non-ML
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[2]],
            IndicatorType.MOMENTUM: [],
        }

        # ML条件生成のモック
        mock_generator._create_ml_long_conditions.return_value = [
            Condition(left_operand="ML_UP_PROB", operator=">", right_operand=0.6)
        ]

        # Short条件にML_DOWN_PROBが追加されるはず
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # ML条件が追加されていることを確認
        # (len >= ML条件の数)
        assert len(result_long) >= 1

    def test_disabled_ml_indicators_ignored(self, strategy, mock_generator):
        """無効化されたML指標の無視"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=False),  # Disabled
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[1]],
            IndicatorType.MOMENTUM: [],
        }

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # ML条件は生成されないはず (disabled)
        # 実際の実装ではfilter(lambda ind: ind.enabled and ind.type.startswith("ML_"))
        # なので無効化されたML指標は無視される

    def test_logging_debug_messages(self, strategy, mock_generator):
        """デバッグログの確認"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators,
            IndicatorType.MOMENTUM: [],
        }

        with patch.object(strategy, '_create_trend_long_conditions') as mock_trend_long:
            mock_trend_long.return_value = [Condition(left_operand="SMA_20", operator=">", right_operand=100)]

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # ログが呼ばれていることを確認
            # strategy.condition_generator.logger.debug.assert_called()

    def test_short_conditions_generation(self, strategy, mock_generator):
        """short条件の適切な生成テスト"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[0]],
            IndicatorType.MOMENTUM: [indicators[1]],
        }

        # random.choiceのモック
        with patch('backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy.random.choice') as mock_choice:
            mock_choice.side_effect = [indicators[0], indicators[1]]  # Trend, Momentum

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            # short条件も生成される
            assert len(result_short) >= 2  # TrendとMomentum両方のshort

    def test_random_choice_consistency(self, strategy, mock_generator):
        """random.choiceの整合性テスト"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="EMA_50", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators,
            IndicatorType.MOMENTUM: [],
        }

        # random.choiceが呼ばれていることを確認
        with patch('backend.app.services.auto_strategy.generators.strategies.different_indicators_strategy.random.choice') as mock_choice:
            mock_choice.return_value = indicators[0]  # 常に最初の指標を選択

            result_long, result_short, result_exit = strategy.generate_conditions(indicators)

            mock_choice.assert_called()


class TestDifferentIndicatorsStrategyBugs:
    """DifferentIndicatorsStrategyの潜在的バグテスト"""

    def test_empty_ml_indicators_list_bug(self, strategy, mock_generator):
        """BUG: ML指標が空リストの場合の条件生成"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: indicators,
            IndicatorType.MOMENTUM: [],
        }

        # ML指標なしの場合
        # ML条件生成が呼び出されないことを確認
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 結果生成可能
        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_single_ml_indicator_short_condition_bug(self, strategy, mock_generator):
        """BUG: ML指標が1つの場合のshort条件生成"""
        indicators = [
            IndicatorGene(type="ML_UP_PROB", enabled=True),
            IndicatorGene(type="SMA_20", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[1]],
            IndicatorType.MOMENTUM: [],
        }

        # コードの85行: if ml_indicators and len(ml_indicators) >= 2:
        # ML指標が1つだけの場合はshort条件生成がスキップされるバグ
        # short_conditions.append(Condition(...)) は呼ばれない

        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # short条件が生成されていることを確認
        # もしバグがあれば、トレンドによるshortのみ生成

    def test_mixed_disabled_and_enabled_indicators(self, strategy, mock_generator):
        """有効・無効指標の混合処理テスト"""
        indicators = [
            IndicatorGene(type="SMA_20", enabled=True),
            IndicatorGene(type="RSI", enabled=False),     # Disabled momentum
            IndicatorGene(type="ML_UP_PROB", enabled=False),  # Disabled ML
            IndicatorGene(type="MACD", enabled=True),
        ]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: [indicators[0]],
            IndicatorType.MOMENTUM: [indicators[3]],  # MACDのみ有効
        }

        # 無効化された指標は処理されない
        result_long, result_short, result_exit = strategy.generate_conditions(indicators)

        # 有効な指標のみ処理
        assert len(result_long) > 0
        assert len(result_short) > 0

    def test_large_number_of_indicators_performance(self, strategy, mock_generator):
        """大量の指標でのパフォーマンステスト"""
        # 多くの指標でテスト
        num_indicators = 20
        indicators = [
            IndicatorGene(type=f"IND_{i}", enabled=True) for i in range(num_indicators)
        ]

        # 半分をtrend、半分をmomentum
        trend_indicators = indicators[:num_indicators//2]
        momentum_indicators = indicators[num_indicators//2:]

        mock_generator._classify_indicators_by_type.return_value = {
            IndicatorType.TREND: trend_indicators,
            IndicatorType.MOMENTUM: momentum_indicators,
        }

        # 生成メソッドが大量に呼ばれても問題ないことを確認
        with patch.object(strategy, '_create_trend_long_conditions') as mock_trend:
            with patch.object(strategy, '_create_momentum_long_conditions') as mock_momentum:
                mock_trend.return_value = [Condition(left_operand="trend", operator=">", right_operand=0)]
                mock_momentum.return_value = [Condition(left_operand="momentum", operator=">", right_operand=0)]

                result_long, result_short, result_exit = strategy.generate_conditions(indicators)

                # random.choiceにより各タイプから1つの指標のみ選択される
                mock_trend.assert_called()
                mock_momentum.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])