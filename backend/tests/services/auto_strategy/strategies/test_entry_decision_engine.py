from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.auto_strategy.config.constants import EntryType, PositionSizingMethod
from app.services.auto_strategy.genes import PositionSizingGene, TPSLGene, TPSLMethod
from app.services.auto_strategy.genes.conditions import Condition
from app.services.auto_strategy.genes.entry import EntryGene
from app.services.auto_strategy.strategies.entry_decision_engine import (
    EntryDecisionEngine,
)
from app.services.auto_strategy.strategies.runtime_state import StrategyRuntimeState


class TestEntryDecisionEngine:
    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.runtime_state = StrategyRuntimeState()
        strategy.ml_predictor = None
        strategy._current_bar_index = 7
        strategy.data.Close = [100.0]
        strategy.stateful_conditions_evaluator = MagicMock()
        strategy.entry_executor = MagicMock()
        strategy.order_manager = MagicMock()
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()
        strategy._precomputed_signals = {}
        strategy.condition_evaluator = MagicMock()
        strategy.tpsl_service = MagicMock()
        strategy.position_sizing_service = MagicMock()
        strategy.gene = MagicMock()
        strategy.gene.position_sizing_gene = None
        strategy.gene.long_entry_conditions = []
        strategy.gene.short_entry_conditions = []
        strategy.gene.tool_genes = []
        strategy.data.High = [105.0]
        strategy.data.Low = [95.0]
        strategy.data.Volume = [1000.0]
        strategy.data.index = [MagicMock()]
        strategy.data.__len__ = MagicMock(return_value=1)
        strategy._calculate_position_size.return_value = 0.25
        strategy._calculate_effective_tpsl_prices.return_value = (95.0, 110.0)
        strategy._get_effective_entry_gene.return_value = None
        strategy._ml_allows_entry.return_value = True
        strategy._get_effective_tpsl_gene.return_value = None
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return EntryDecisionEngine(strategy)

    def test_determine_entry_direction_prioritizes_regular_signals(
        self, engine, strategy
    ):
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[True, True])

        direction = engine.determine_entry_direction()

        assert direction == 1.0
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.assert_not_called()

    def test_determine_entry_direction_uses_stateful_fallback(self, engine, strategy):
        engine.tools_block_entry = MagicMock(return_value=False)
        engine.check_entry_conditions = MagicMock(side_effect=[False, False])
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.return_value = (
            -1.0
        )

        direction = engine.determine_entry_direction()

        assert direction == -1.0

    def test_execute_entry_updates_runtime_state_for_market_order(
        self, engine, strategy
    ):
        strategy._get_effective_entry_gene.return_value = None
        strategy.entry_executor.calculate_entry_params.return_value = {}
        engine.calculate_position_size = MagicMock(return_value=0.25)
        engine.calculate_effective_tpsl_prices = MagicMock(return_value=(95.0, 110.0))

        executed = engine.execute_entry(1.0)

        assert executed is True
        strategy.buy.assert_called_once_with(size=0.25)
        assert strategy.runtime_state.entry_price == 100.0
        assert strategy.runtime_state.sl_price == 95.0
        assert strategy.runtime_state.tp_price == 110.0
        assert strategy.runtime_state.position_direction == 1.0

    def test_execute_entry_creates_pending_order_for_non_market(self, engine, strategy):
        entry_gene = EntryGene(entry_type=EntryType.LIMIT, order_validity_bars=3)
        strategy._get_effective_entry_gene.return_value = entry_gene
        strategy.entry_executor.calculate_entry_params.return_value = {"limit": 99.0}
        engine.calculate_position_size = MagicMock(return_value=0.25)
        engine.calculate_effective_tpsl_prices = MagicMock(return_value=(95.0, 110.0))

        executed = engine.execute_entry(-1.0)

        assert executed is True
        strategy.sell.assert_not_called()
        strategy.order_manager.create_pending_order.assert_called_once_with(
            direction=-1.0,
            size=0.25,
            entry_params={"limit": 99.0},
            sl_price=95.0,
            tp_price=110.0,
            entry_gene=entry_gene,
            current_bar_index=7,
        )

    def test_execute_entry_stops_when_ml_rejects(self, engine, strategy):
        strategy.ml_predictor = MagicMock()
        strategy._ml_allows_entry.return_value = False

        executed = engine.execute_entry(1.0)

        assert executed is False
        strategy.buy.assert_not_called()
        strategy.order_manager.create_pending_order.assert_not_called()

    def test_check_entry_conditions_uses_precomputed_signal(self, engine, strategy):
        strategy._precomputed_signals = {1.0: np.array([False, True])}
        strategy.data = [object(), object()]

        result = engine.check_entry_conditions(1.0)

        assert result is True
        strategy.condition_evaluator.evaluate_conditions.assert_not_called()

    def test_check_entry_conditions_falls_back_when_cached_signal_is_scalar(
        self, engine, strategy
    ):
        strategy._precomputed_signals = {1.0: True}
        strategy.data = [object(), object()]
        strategy.gene.long_entry_conditions = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]
        strategy.condition_evaluator.evaluate_conditions.return_value = True

        result = engine.check_entry_conditions(1.0)

        assert result is True
        strategy.condition_evaluator.evaluate_conditions.assert_called_once_with(
            strategy.gene.long_entry_conditions,
            strategy,
        )

    def test_calculate_position_size_recomputes_from_service(self, engine, strategy):
        position_sizing_gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.FIXED_RATIO,
        )
        strategy.gene.position_sizing_gene = position_sizing_gene
        strategy.equity = 100000.0
        strategy.position_sizing_service.calculate_position_size_fast.side_effect = [
            0.05,
            0.08,
        ]
        strategy.data.Close = [50000.0, 51000.0]
        strategy.data.High = np.array([50500.0, 51500.0])
        strategy.data.Low = np.array([49500.0, 50500.0])
        strategy.data.__len__ = MagicMock(return_value=2)

        assert engine.calculate_position_size() == pytest.approx(0.0255)
        assert engine.calculate_position_size() == pytest.approx(0.0408)

    def test_calculate_position_size_preserves_gene_sized_quantity(
        self, engine, strategy
    ):
        position_sizing_gene = PositionSizingGene(
            enabled=True,
            method=PositionSizingMethod.FIXED_QUANTITY,
            min_position_size=0.001,
            max_position_size=500.0,
        )
        strategy.gene.position_sizing_gene = position_sizing_gene
        strategy.equity = 100000.0
        strategy.position_sizing_service.calculate_position_size_fast.return_value = (
            250.0
        )
        strategy.data.Close = [50000.0, 51000.0]
        strategy.data.High = np.array([50500.0, 51500.0])
        strategy.data.Low = np.array([49500.0, 50500.0])
        strategy.data.__len__ = MagicMock(return_value=2)

        assert engine.calculate_position_size() == pytest.approx(250.0)

    def test_calculate_effective_tpsl_prices_uses_precomputed_atr(
        self, engine, strategy
    ):
        tpsl_gene = TPSLGene(
            enabled=True,
            method=TPSLMethod.VOLATILITY_BASED,
            atr_period=14,
        )
        strategy._get_effective_tpsl_gene.return_value = tpsl_gene
        strategy._precomputed_tpsl_atr = {14: np.array([1.2])}
        strategy.data = MagicMock()
        strategy.data.Close = [100.0]
        strategy.data.High = [105.0]
        strategy.data.Low = [95.0]
        strategy.data.__len__ = MagicMock(return_value=1)
        strategy.tpsl_service.calculate_tpsl_prices.return_value = (95.0, 110.0)

        result = engine.calculate_effective_tpsl_prices(1.0, 100.0)

        assert result == (95.0, 110.0)
        market_data = strategy.tpsl_service.calculate_tpsl_prices.call_args.kwargs[
            "market_data"
        ]
        assert market_data["atr"] == 1.2
