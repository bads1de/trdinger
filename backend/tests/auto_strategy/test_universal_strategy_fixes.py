"""
UniversalStrategy の修正のテスト

1. Position Sizing の動的計算（キャッシュ削除）
2. Stateful Condition の direction フィールド
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy
from app.services.auto_strategy.models.strategy_models import (
    StrategyGene,
    IndicatorGene,
    PositionSizingGene,
)
from app.services.auto_strategy.models.condition import Condition
from app.services.auto_strategy.models.stateful_condition import StatefulCondition
from app.services.auto_strategy.models.enums import PositionSizingMethod


class TestPositionSizingNoCaching:
    """Position Sizing のキャッシュが削除されていることをテスト"""

    @pytest.fixture
    def mock_broker(self):
        """Brokerのモック"""
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        return broker

    @pytest.fixture
    def mock_data(self):
        """Dataのモック"""
        data = MagicMock()
        data.Close = [100, 101, 102]
        data.High = [105, 106, 107]
        data.Low = [95, 96, 97]
        data.__len__.return_value = 3
        return data

    @pytest.fixture
    def gene_with_position_sizing(self):
        """PositionSizingGene付きの戦略遺伝子"""
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            exit_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            position_sizing_gene=PositionSizingGene(
                enabled=True,
                method=PositionSizingMethod.FIXED_RATIO,
                risk_per_trade=0.01,
            ),
            risk_management={},
            metadata={},
        )

    def test_position_size_recalculated_on_each_call(
        self, mock_broker, mock_data, gene_with_position_sizing
    ):
        """ポジションサイズが毎回再計算されることをテスト"""
        params = {"strategy_gene": gene_with_position_sizing}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.PositionSizingService"
        ) as MockPSService:
            mock_ps_service = MockPSService.return_value
            # 異なる値を返すように設定
            mock_ps_service.calculate_position_size_fast.side_effect = [
                0.05,
                0.08,
                0.10,
            ]

            strategy = UniversalStrategy(mock_broker, mock_data, params)

            # 3回呼び出して、毎回異なる値が返されることを確認
            size1 = strategy._calculate_position_size()
            size2 = strategy._calculate_position_size()
            size3 = strategy._calculate_position_size()

            # calculate_position_size_fast が3回呼ばれたことを確認
            assert mock_ps_service.calculate_position_size_fast.call_count == 3

            # 値がキャッシュされていないことを確認
            assert size1 != size2
            assert size2 != size3

    def test_no_cached_position_size_attribute(
        self, mock_broker, mock_data, gene_with_position_sizing
    ):
        """_cached_position_size 属性が設定されないことをテスト"""
        params = {"strategy_gene": gene_with_position_sizing}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.PositionSizingService"
        ) as MockPSService:
            mock_ps_service = MockPSService.return_value
            mock_ps_service.calculate_position_size_fast.return_value = 0.05

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy._calculate_position_size()

            # _cached_position_size 属性が存在しないことを確認
            assert not hasattr(strategy, "_cached_position_size")


class TestStatefulConditionDirection:
    """StatefulCondition の direction フィールドのテスト"""

    def test_stateful_condition_has_direction_field(self):
        """StatefulCondition に direction フィールドがあることを確認"""
        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand=100.0)

        # デフォルトは "long"
        stateful_long = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
        )
        assert stateful_long.direction == "long"

        # 明示的に "short" を指定
        stateful_short = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            direction="short",
        )
        assert stateful_short.direction == "short"

    def test_get_stateful_entry_direction_returns_correct_direction(self):
        """_get_stateful_entry_direction が正しい方向を返すことをテスト"""
        mock_broker = MagicMock()
        mock_data = MagicMock()
        mock_data.Close = [100, 101, 102]
        mock_data.__len__.return_value = 3

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)

        # ショート方向のステートフル条件
        stateful_short = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            direction="short",
        )

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            stateful_conditions=[stateful_short],
            risk_management={},
            metadata={},
        )

        params = {"strategy_gene": gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.evaluate_stateful_condition.return_value = True

            strategy = UniversalStrategy(mock_broker, mock_data, params)

            # ショート方向 (-1.0) が返されることを確認
            direction = strategy._get_stateful_entry_direction()
            assert direction == -1.0

    def test_stateful_condition_entry_with_direction(self):
        """ステートフル条件のみでエントリーが実行されることをテスト"""
        mock_broker = MagicMock()
        mock_data = MagicMock()
        mock_data.Close = [100, 101, 102]
        mock_data.High = [105, 106, 107]
        mock_data.Low = [95, 96, 97]
        mock_data.__len__.return_value = 3

        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)

        # ロング方向のステートフル条件
        stateful_long = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            direction="long",
        )

        gene = StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            entry_conditions=[],
            long_entry_conditions=[],
            short_entry_conditions=[],
            stateful_conditions=[stateful_long],
            risk_management={},
            metadata={},
        )

        params = {"strategy_gene": gene}

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            mock_evaluator = MockEval.return_value
            # 通常条件は不成立、ステートフル条件は成立
            mock_evaluator.evaluate_conditions.return_value = False
            mock_evaluator.evaluate_stateful_condition.return_value = True

            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()
            strategy.sell = MagicMock()

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_position:
                mock_position.return_value = None

                strategy.next()

            # ステートフル条件でロングエントリーが実行されたことを確認
            strategy.buy.assert_called_once()
            strategy.sell.assert_not_called()


class TestStatefulConditionSerialization:
    """StatefulCondition の direction フィールドのシリアライズテスト"""

    def test_direction_serialization(self):
        """direction フィールドが正しくシリアライズされることをテスト"""
        from app.services.auto_strategy.serializers.dict_converter import DictConverter

        converter = DictConverter()

        trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
        follow = Condition(left_operand="close", operator=">", right_operand=100.0)

        stateful = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            direction="short",
        )

        result = converter.stateful_condition_to_dict(stateful)

        assert result["direction"] == "short"

    def test_direction_deserialization(self):
        """direction フィールドが正しくデシリアライズされることをテスト"""
        from app.services.auto_strategy.serializers.dict_converter import DictConverter

        converter = DictConverter()

        data = {
            "trigger_condition": {
                "left_operand": "RSI",
                "operator": "<",
                "right_operand": 30.0,
            },
            "follow_condition": {
                "left_operand": "close",
                "operator": ">",
                "right_operand": 100.0,
            },
            "lookback_bars": 5,
            "cooldown_bars": 2,
            "direction": "short",
            "enabled": True,
        }

        result = converter.dict_to_stateful_condition(data)

        assert result.direction == "short"

    def test_direction_default_on_deserialization(self):
        """direction が省略された場合はデフォルト値 'long' が使用されることをテスト"""
        from app.services.auto_strategy.serializers.dict_converter import DictConverter

        converter = DictConverter()

        # direction を省略したデータ
        data = {
            "trigger_condition": {
                "left_operand": "RSI",
                "operator": "<",
                "right_operand": 30.0,
            },
            "follow_condition": {
                "left_operand": "close",
                "operator": ">",
                "right_operand": 100.0,
            },
            "lookback_bars": 5,
            "cooldown_bars": 2,
            "enabled": True,
        }

        result = converter.dict_to_stateful_condition(data)

        # デフォルト値 "long" が使用される
        assert result.direction == "long"
