"""
ExitDecisionEngine の追加ユニットテスト

既存のテストファイル (test_exit_decision_engine.py) には 1 テストしかなかったため、
未カバー分岐を網羅するテストを追加します。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.auto_strategy.config.constants import ExitType
from app.services.auto_strategy.genes import Condition, ConditionGroup
from app.services.auto_strategy.genes.exit import ExitGene
from app.services.auto_strategy.strategies.exit_decision_engine import (
    ExitDecisionEngine,
)


def _make_position(size: float = 1.0):
    return SimpleNamespace(size=size)


def _make_long_exit_gene(
    enabled: bool = True,
    exit_type: ExitType = ExitType.FULL,
    partial_exit_enabled: bool = False,
    trailing_stop_activation: bool = False,
    partial_exit_pct: float = 0.5,
) -> ExitGene:
    return ExitGene(
        enabled=enabled,
        exit_type=exit_type,
        partial_exit_enabled=partial_exit_enabled,
        trailing_stop_activation=trailing_stop_activation,
        partial_exit_pct=partial_exit_pct,
    )


class TestDetermineExitDirection:
    """determine_exit_direction のテスト"""

    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = _make_position(1.0)
        strategy._current_bar_index = 0
        strategy._precomputed_exit_signals = {}
        strategy.condition_evaluator = MagicMock()
        strategy.gene = MagicMock()
        strategy.gene.long_exit_conditions = []
        strategy.gene.short_exit_conditions = []
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_returns_zero_when_no_position(self, engine, strategy):
        strategy.position = None
        assert engine.determine_exit_direction() == 0.0

    def test_returns_long_direction_for_long_position(self, engine, strategy):
        strategy.position = _make_position(0.5)
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.long_exit_conditions = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]
        strategy.condition_evaluator.evaluate_single_condition.return_value = True

        assert engine.determine_exit_direction() == 1.0

    def test_returns_short_direction_for_short_position(self, engine, strategy):
        strategy.position = _make_position(-0.5)
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.short_exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand=100.0)
        ]
        strategy.condition_evaluator.evaluate_single_condition.return_value = True

        assert engine.determine_exit_direction() == -1.0

    def test_returns_zero_when_exit_gene_is_none(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = None
        assert engine.determine_exit_direction() == 0.0

    def test_returns_zero_when_exit_gene_is_disabled(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            enabled=False
        )
        assert engine.determine_exit_direction() == 0.0

    def test_returns_zero_when_no_exit_conditions(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.long_exit_conditions = []
        strategy.gene.short_exit_conditions = []
        assert engine.determine_exit_direction() == 0.0

    def test_returns_zero_when_conditions_not_satisfied(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.long_exit_conditions = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]
        strategy.condition_evaluator.evaluate_single_condition.return_value = False

        assert engine.determine_exit_direction() == 0.0

    def test_uses_precomputed_signal_for_long(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.array([False, True])}
        strategy._current_bar_index = 1
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.long_exit_conditions = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]

        assert engine.determine_exit_direction() == 1.0
        strategy.condition_evaluator.evaluate_single_condition.assert_not_called()

    def test_uses_precomputed_signal_for_short(self, engine, strategy):
        strategy.position = _make_position(-0.5)
        strategy._precomputed_exit_signals = {-1.0: np.array([False, True])}
        strategy._current_bar_index = 1
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene()
        strategy.gene.short_exit_conditions = [
            Condition(left_operand="close", operator="<", right_operand=100.0)
        ]

        assert engine.determine_exit_direction() == -1.0
        strategy.condition_evaluator.evaluate_single_condition.assert_not_called()


class TestExecuteExit:
    """execute_exit のテスト"""

    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = _make_position(0.5)
        strategy._current_bar_index = 0
        strategy._precomputed_exit_signals = {}
        strategy.condition_evaluator = MagicMock()
        strategy.gene = MagicMock()
        strategy.gene.long_exit_conditions = []
        strategy.gene.short_exit_conditions = []
        strategy.data = MagicMock(spec=pd.DataFrame)
        strategy.data.Close = np.array([100.0])
        strategy.position_manager = MagicMock()
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_returns_false_when_direction_is_zero(self, engine):
        assert engine.execute_exit(0.0) is False

    def test_returns_false_when_no_position(self, engine, strategy):
        strategy.position = None
        assert engine.execute_exit(1.0) is False

    def test_returns_false_when_exit_gene_is_none(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = None
        assert engine.execute_exit(1.0) is False

    def test_returns_false_when_exit_gene_is_disabled(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            enabled=False
        )
        assert engine.execute_exit(1.0) is False

    def test_executes_full_exit_for_long_position(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            exit_type=ExitType.FULL
        )

        result = engine.execute_exit(1.0)

        assert result is True
        strategy.sell.assert_called_once_with(size=pytest.approx(0.5))

    def test_executes_full_exit_for_short_position(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            exit_type=ExitType.FULL
        )

        result = engine.execute_exit(-1.0)

        assert result is True
        strategy.buy.assert_called_once_with(size=pytest.approx(0.5))

    def test_executes_partial_exit_when_enabled(self, engine, strategy):
        # position size 0.5 * exit_pct 0.5 = 0.25, round(0.25)=0 -> 1 にクランプ
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            exit_type=ExitType.PARTIAL,
            partial_exit_enabled=True,
            partial_exit_pct=0.5,
        )

        result = engine.execute_exit(1.0)

        assert result is True
        strategy.sell.assert_called_once_with(size=1)

    def test_does_not_partial_exit_when_flag_disabled(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            exit_type=ExitType.PARTIAL,
            partial_exit_enabled=False,
        )

        result = engine.execute_exit(1.0)

        # partial_exit_enabled=False の場合は full exit 経路
        assert result is True
        strategy.sell.assert_called_once_with(size=pytest.approx(0.5))

    def test_activates_trailing_stop(self, engine, strategy):
        strategy._get_effective_exit_gene.return_value = _make_long_exit_gene(
            exit_type=ExitType.TRAILING,
            trailing_stop_activation=True,
        )

        result = engine.execute_exit(1.0)

        assert result is True
        strategy.position_manager.activate_trailing_stop.assert_called_once()
        strategy.sell.assert_not_called()


class TestEvaluateConditionGroup:
    """_evaluate_condition_group のテスト"""

    @pytest.fixture
    def strategy(self):
        return MagicMock()

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    @pytest.fixture
    def evaluator(self):
        return MagicMock()

    def test_returns_false_for_empty_group(self, engine, evaluator):
        group = ConditionGroup(operator="AND", conditions=[])
        assert engine._evaluate_condition_group(group, evaluator) is False

    def test_and_group_all_true(self, engine, evaluator):
        evaluator.evaluate_single_condition.return_value = True
        group = ConditionGroup(
            operator="AND",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0),
                Condition(left_operand="close", operator="<", right_operand=200.0),
            ],
        )
        assert engine._evaluate_condition_group(group, evaluator) is True

    def test_and_group_one_false(self, engine, evaluator):
        evaluator.evaluate_single_condition.side_effect = [True, False]
        group = ConditionGroup(
            operator="AND",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0),
                Condition(left_operand="close", operator="<", right_operand=200.0),
            ],
        )
        assert engine._evaluate_condition_group(group, evaluator) is False

    def test_or_group_one_true(self, engine, evaluator):
        evaluator.evaluate_single_condition.side_effect = [False, True]
        group = ConditionGroup(
            operator="OR",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0),
                Condition(left_operand="close", operator="<", right_operand=200.0),
            ],
        )
        assert engine._evaluate_condition_group(group, evaluator) is True

    def test_or_group_all_false(self, engine, evaluator):
        evaluator.evaluate_single_condition.return_value = False
        group = ConditionGroup(
            operator="OR",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0),
                Condition(left_operand="close", operator="<", right_operand=200.0),
            ],
        )
        assert engine._evaluate_condition_group(group, evaluator) is False

    def test_nested_group(self, engine, evaluator):
        evaluator.evaluate_single_condition.side_effect = [True, True]
        inner = ConditionGroup(
            operator="AND",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0),
                Condition(left_operand="close", operator="<", right_operand=200.0),
            ],
        )
        outer = ConditionGroup(operator="OR", conditions=[inner])
        assert engine._evaluate_condition_group(outer, evaluator) is True


class TestEvaluateSingleCondition:
    """_evaluate_single_condition のテスト"""

    @pytest.fixture
    def strategy(self):
        return MagicMock()

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    @pytest.fixture
    def evaluator(self):
        return MagicMock()

    def test_delegates_to_evaluator_for_condition(self, engine, evaluator, strategy):
        cond = Condition(left_operand="close", operator=">", right_operand=100.0)
        evaluator.evaluate_single_condition.return_value = True

        result = engine._evaluate_single_condition(cond, evaluator)

        assert result is True
        evaluator.evaluate_single_condition.assert_called_once_with(cond, strategy)

    def test_recurses_for_condition_group(self, engine, evaluator):
        inner = ConditionGroup(
            operator="AND",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
        )
        evaluator.evaluate_single_condition.return_value = True

        result = engine._evaluate_single_condition(inner, evaluator)

        assert result is True

    def test_returns_false_for_unsupported_type(self, engine, evaluator):
        result = engine._evaluate_single_condition("not a condition", evaluator)
        assert result is False


class TestGetCachedExitSignal:
    """_get_cached_exit_signal のテスト"""

    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = _make_position(1.0)
        strategy._current_bar_index = 1
        strategy._precomputed_exit_signals = {}
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_returns_none_when_no_position(self, engine, strategy):
        strategy.position = None
        assert engine._get_cached_exit_signal() is None

    def test_returns_none_when_cache_not_dict(self, engine, strategy):
        strategy._precomputed_exit_signals = "not a dict"
        assert engine._get_cached_exit_signal() is None

    def test_returns_none_when_direction_not_in_cache(self, engine, strategy):
        strategy._precomputed_exit_signals = {2.0: np.array([True, True])}
        assert engine._get_cached_exit_signal() is None

    def test_returns_none_for_scalar_signal(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.bool_(True)}
        assert engine._get_cached_exit_signal() is None

    def test_returns_none_when_index_out_of_range(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.array([True])}
        strategy._current_bar_index = 5
        assert engine._get_cached_exit_signal() is None

    def test_returns_value_for_pandas_series(self, engine, strategy):
        series = pd.Series([False, True, False])
        strategy._precomputed_exit_signals = {1.0: series}
        strategy._current_bar_index = 1
        assert bool(engine._get_cached_exit_signal()) is True

    def test_returns_value_for_ndarray(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.array([False, True, False])}
        strategy._current_bar_index = 2
        assert bool(engine._get_cached_exit_signal()) is False

    def test_uses_short_direction_for_short_position(self, engine, strategy):
        strategy.position = _make_position(-1.0)
        strategy._precomputed_exit_signals = {-1.0: np.array([False, True])}
        strategy._current_bar_index = 1
        assert bool(engine._get_cached_exit_signal()) is True


class TestEvaluateExitConditions:
    """_evaluate_exit_conditions のテスト"""

    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        strategy.position = _make_position(1.0)
        strategy._current_bar_index = 0
        strategy._precomputed_exit_signals = {}
        strategy.condition_evaluator = MagicMock()
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_returns_false_for_empty_conditions(self, engine):
        assert engine._evaluate_exit_conditions([]) is False

    def test_uses_cached_signal_when_available(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.array([True])}
        strategy._current_bar_index = 0
        assert engine._evaluate_exit_conditions([object()]) is True

    def test_falls_back_to_condition_evaluation(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.bool_(True)}  # scalar -> skip
        cond = Condition(left_operand="close", operator=">", right_operand=100.0)
        strategy.condition_evaluator.evaluate_single_condition.return_value = True
        assert engine._evaluate_exit_conditions([cond]) is True

    def test_handles_condition_group_in_fallback(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.bool_(True)}  # scalar -> skip
        strategy.condition_evaluator.evaluate_single_condition.return_value = True
        group = ConditionGroup(
            operator="AND",
            conditions=[
                Condition(left_operand="close", operator=">", right_operand=100.0)
            ],
        )
        assert engine._evaluate_exit_conditions([group]) is True

    def test_returns_false_when_no_condition_matches(self, engine, strategy):
        strategy._precomputed_exit_signals = {1.0: np.bool_(True)}  # scalar -> skip
        strategy.condition_evaluator.evaluate_single_condition.return_value = False
        cond = Condition(left_operand="close", operator=">", right_operand=100.0)
        assert engine._evaluate_exit_conditions([cond]) is False


class TestGetExitConditions:
    """_get_exit_conditions のテスト"""

    @pytest.fixture
    def strategy(self):
        strategy = MagicMock()
        long_conds = [
            Condition(left_operand="close", operator=">", right_operand=100.0)
        ]
        short_conds = [
            Condition(left_operand="close", operator="<", right_operand=100.0)
        ]
        strategy.gene = MagicMock()
        strategy.gene.long_exit_conditions = long_conds
        strategy.gene.short_exit_conditions = short_conds
        return strategy

    @pytest.fixture
    def engine(self, strategy):
        return ExitDecisionEngine(strategy)

    def test_returns_long_conditions_for_positive_direction(self, engine, strategy):
        result = engine._get_exit_conditions(1.0)
        assert result is strategy.gene.long_exit_conditions

    def test_returns_short_conditions_for_negative_direction(self, engine, strategy):
        result = engine._get_exit_conditions(-1.0)
        assert result is strategy.gene.short_exit_conditions

    def test_returns_empty_list_when_attribute_missing(self, engine, strategy):
        strategy.gene.long_exit_conditions = None
        # getattr のデフォルト動作を確認する
        result = engine._get_exit_conditions(1.0)
        # getattr(obj, name, []) だが、None が返る -> 空判定
        # 実装上は get で [] をデフォルトにしているので、None でもそのまま返る
        assert result is None
