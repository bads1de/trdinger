import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd

from app.services.auto_strategy.strategies.universal_strategy import UniversalStrategy
from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState
from app.services.auto_strategy.genes import (
    StrategyGene,
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
    PositionSizingGene,
)
from app.services.auto_strategy.genes.conditions import Condition, StatefulCondition
from app.services.auto_strategy.config.constants import PositionSizingMethod


class TestUniversalStrategyAll:
    """UniversalStrategy の包括的なテスト (基本、TPSLスライス、キャッシュ、Stateful)"""

    @pytest.fixture
    def mock_broker(self):
        broker = MagicMock()
        broker.orders = []
        broker.trades = []
        broker.commission = 0.001
        return broker

    @pytest.fixture
    def mock_data(self):
        """標準的なテストデータ (3バー)"""
        data = MagicMock()
        data.Close = [100, 101, 102]
        data.High = [105, 106, 107]
        data.Low = [95, 96, 97]
        data.__len__.return_value = 3
        return data

    @pytest.fixture
    def mock_data_large(self):
        """大規模なテストデータ (100バー)"""
        data = MagicMock()
        data_length = 100
        data.High = np.random.uniform(105, 115, data_length)
        data.Low = np.random.uniform(95, 105, data_length)
        data.Close = np.random.uniform(100, 110, data_length)
        data.df = pd.DataFrame(
            {"High": data.High, "Low": data.Low, "Close": data.Close},
            index=pd.date_range("2023-01-01", periods=data_length, freq="h"),
        )
        data.index = data.df.index
        data.__len__ = MagicMock(return_value=data_length)
        return data

    @pytest.fixture
    def valid_gene(self):
        return StrategyGene(
            indicators=[
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
            ],
            long_entry_conditions=[
                Condition(left_operand="close", operator=">", right_operand="sma")
            ],
            short_entry_conditions=[
                Condition(left_operand="close", operator="<", right_operand="sma")
            ],
            risk_management={},
            metadata={"test": True},
        )

    # ---------------------------------------------------------------------------
    # 基本機能テスト
    # ---------------------------------------------------------------------------

    def test_init_and_indicators(self, mock_broker, mock_data, valid_gene):
        params = {"strategy_gene": valid_gene}
        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.IndicatorCalculator"
        ) as MockCalc:
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.init()
            assert MockCalc.return_value.init_indicator.call_count == len(
                valid_gene.indicators
            )

    def test_entry_logic(self, mock_broker, mock_data, valid_gene):
        params = {"strategy_gene": valid_gene}
        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()
            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos:
                mock_pos.return_value = None
                MockEval.return_value.evaluate_conditions.side_effect = [
                    True,
                    False,
                ]  # Long: True
                strategy.next()
                strategy.buy.assert_called_once()

    def test_next_skips_entry_before_evaluation_start(
        self, mock_broker, mock_data, valid_gene
    ):
        mock_data.index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="h")
        params = {
            "strategy_gene": valid_gene,
            "evaluation_start": "2024-01-01 03:00:00",
        }

        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.ConditionEvaluator"
        ) as MockEval:
            strategy = UniversalStrategy(mock_broker, mock_data, params)
            strategy.buy = MagicMock()

            with patch.object(
                UniversalStrategy, "position", new_callable=PropertyMock
            ) as mock_pos:
                mock_pos.return_value = None
                strategy.next()

            strategy.buy.assert_not_called()
            MockEval.return_value.evaluate_conditions.assert_not_called()

    def test_runtime_state_properties_stay_in_sync(
        self, mock_broker, mock_data, valid_gene
    ):
        strategy = UniversalStrategy(
            mock_broker, mock_data, {"strategy_gene": valid_gene}
        )

        strategy.runtime_state.set_open_position(
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            direction=1.0,
        )

        assert strategy._entry_price == 100.0
        assert strategy._sl_price == 95.0
        assert strategy._tp_price == 110.0
        assert strategy._position_direction == 1.0

        strategy._tp_reached = True
        strategy._trailing_tp_sl = 108.0
        strategy._sl_price = 97.0

        assert isinstance(strategy.runtime_state, StrategyRuntimeState)
        assert strategy.runtime_state.tp_reached is True
        assert strategy.runtime_state.trailing_tp_sl == 108.0
        assert strategy.runtime_state.sl_price == 97.0

    # ---------------------------------------------------------------------------
    # TPSL & データスライス テスト
    # ---------------------------------------------------------------------------

    def test_tpsl_data_slice_size(self, mock_broker, mock_data_large, valid_gene):
        """atr_period に基づいてデータスライスサイズが動的に調整されるか"""
        valid_gene.long_tpsl_gene = TPSLGene(
            enabled=True, method=TPSLMethod.VOLATILITY_BASED, atr_period=50
        )
        strategy = UniversalStrategy(
            mock_broker, mock_data_large, {"strategy_gene": valid_gene}
        )

        with patch.object(
            strategy.tpsl_service, "calculate_tpsl_prices", return_value=(90.0, 110.0)
        ) as mock_tpsl:
            with patch.object(
                strategy.condition_evaluator, "evaluate_conditions", return_value=True
            ):
                with (
                    patch.object(strategy, "buy"),
                    patch.object(
                        UniversalStrategy, "position", new_callable=PropertyMock
                    ) as mock_pos,
                ):
                    mock_pos.return_value = None
                    strategy.next()
                    # スライスサイズが atr_period + 1 以上であることを確認
                    ohlc_data = mock_tpsl.call_args.kwargs["market_data"]["ohlc_data"]
                    assert len(ohlc_data) >= 51

    # ---------------------------------------------------------------------------
    # ポジションサイジング & キャッシュ無効化 テスト
    # ---------------------------------------------------------------------------

    def test_position_size_recalculation(self, mock_broker, mock_data, valid_gene):
        """キャッシュを使わず、呼び出しごとに再計算されるか"""
        valid_gene.position_sizing_gene = PositionSizingGene(
            enabled=True, method=PositionSizingMethod.FIXED_RATIO
        )
        with patch(
            "app.services.auto_strategy.strategies.universal_strategy.PositionSizingService"
        ) as MockPS:
            MockPS.return_value.calculate_position_size_fast.side_effect = [0.05, 0.08]
            strategy = UniversalStrategy(
                mock_broker, mock_data, {"strategy_gene": valid_gene}
            )

            assert strategy._calculate_position_size() == 0.05
            assert strategy._calculate_position_size() == 0.08
            assert not hasattr(strategy, "_cached_position_size")

    # ---------------------------------------------------------------------------
    # Stateful Condition テスト
    # ---------------------------------------------------------------------------

    def test_stateful_condition_direction(self, mock_broker, mock_data):
        """StatefulCondition の方向 (long/short) が正しく処理されるか"""
        trigger = Condition(left_operand=25.0, operator="<", right_operand=30.0)
        follow = Condition(left_operand=105.0, operator=">", right_operand=100.0)
        stateful_short = StatefulCondition(
            trigger_condition=trigger,
            follow_condition=follow,
            lookback_bars=5,
            direction="short",
        )

        gene = StrategyGene(indicators=[], stateful_conditions=[stateful_short])
        strategy = UniversalStrategy(mock_broker, mock_data, {"strategy_gene": gene})

        with patch.object(
            strategy.condition_evaluator,
            "evaluate_stateful_condition",
            return_value=True,
        ):
            assert strategy._get_stateful_entry_direction() == -1.0
