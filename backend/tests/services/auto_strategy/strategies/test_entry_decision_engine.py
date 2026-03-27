from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config.constants import EntryType
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
        strategy._calculate_position_size.return_value = 0.25
        strategy._calculate_effective_tpsl_prices.return_value = (95.0, 110.0)
        strategy._get_effective_entry_gene.return_value = None
        strategy._ml_allows_entry.return_value = True
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return EntryDecisionEngine(strategy)

    def test_determine_entry_direction_prioritizes_regular_signals(self, engine, strategy):
        strategy._tools_block_entry.return_value = False
        strategy._check_entry_conditions.side_effect = [True, True]

        direction = engine.determine_entry_direction()

        assert direction == 1.0
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.assert_not_called()

    def test_determine_entry_direction_uses_stateful_fallback(self, engine, strategy):
        strategy._tools_block_entry.return_value = False
        strategy._check_entry_conditions.side_effect = [False, False]
        strategy.stateful_conditions_evaluator.get_stateful_entry_direction.return_value = (
            -1.0
        )

        direction = engine.determine_entry_direction()

        assert direction == -1.0

    def test_execute_entry_updates_runtime_state_for_market_order(self, engine, strategy):
        strategy._get_effective_entry_gene.return_value = None
        strategy.entry_executor.calculate_entry_params.return_value = {}

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
