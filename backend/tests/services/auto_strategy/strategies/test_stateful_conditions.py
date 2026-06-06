"""
StatefulConditionsEvaluator のユニットテスト

UniversalStrategy から分離されたステートフル条件評価ロジックをテストします。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.genes.conditions import (
    Condition,
    StatefulCondition,
)
from app.services.auto_strategy.strategies.stateful_conditions import (
    StatefulConditionsEvaluator,
)


@pytest.fixture
def make_stateful() -> StatefulCondition:
    """テスト用 StatefulCondition のヘルパー"""
    trigger = Condition(left_operand="RSI", operator="<", right_operand=30.0)
    follow = Condition(left_operand="close", operator=">", right_operand="SMA_20")
    return StatefulCondition(
        trigger_condition=trigger,
        follow_condition=follow,
        lookback_bars=5,
    )


@pytest.fixture
def make_strategy(make_stateful: StatefulCondition) -> SimpleNamespace:
    """モック戦略(strategy)を構築"""
    strategy = SimpleNamespace()
    strategy.gene = SimpleNamespace(stateful_conditions=[make_stateful])
    strategy.condition_evaluator = MagicMock()
    strategy.state_tracker = MagicMock()
    strategy._current_bar_index = 10
    return strategy


class TestStatefulConditionsEvaluator:
    """StatefulConditionsEvaluator のテスト"""

    def test_init_sets_strategy(self, make_strategy: SimpleNamespace) -> None:
        """コンストラクタで strategy が保持される"""
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.strategy is make_strategy

    # ----------------------------------------------------------------
    # process_stateful_triggers
    # ----------------------------------------------------------------
    def test_process_stateful_triggers_calls_check_and_record(
        self,
        make_strategy: SimpleNamespace,
        make_stateful: StatefulCondition,
    ) -> None:
        """トリガー条件評価が条件ごとに呼ばれる"""
        evaluator = StatefulConditionsEvaluator(make_strategy)
        evaluator.process_stateful_triggers()

        make_strategy.condition_evaluator.check_and_record_trigger.assert_called_once_with(
            make_stateful,
            make_strategy,
            make_strategy.state_tracker,
            make_strategy._current_bar_index,
        )

    def test_process_stateful_triggers_skips_disabled(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """enabled=False の条件はスキップされる"""
        make_strategy.gene.stateful_conditions[0].enabled = False
        evaluator = StatefulConditionsEvaluator(make_strategy)
        evaluator.process_stateful_triggers()

        make_strategy.condition_evaluator.check_and_record_trigger.assert_not_called()

    def test_process_stateful_triggers_returns_early_when_gene_none(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """strategy.gene が None の場合は何もしない"""
        make_strategy.gene = None
        evaluator = StatefulConditionsEvaluator(make_strategy)
        # 例外なく終了することを確認
        evaluator.process_stateful_triggers()
        make_strategy.condition_evaluator.check_and_record_trigger.assert_not_called()

    def test_process_stateful_triggers_returns_early_when_no_stateful_attr(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """gene に stateful_conditions 属性がない場合は何もしない"""
        # stateful_conditions 属性を持たないモックに置き換え
        make_strategy.gene = SimpleNamespace()
        evaluator = StatefulConditionsEvaluator(make_strategy)
        evaluator.process_stateful_triggers()
        make_strategy.condition_evaluator.check_and_record_trigger.assert_not_called()

    def test_process_stateful_triggers_handles_multiple(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """複数のステートフル条件を全て処理する"""
        sc2 = StatefulCondition(
            trigger_condition=Condition(
                left_operand="MACD", operator=">", right_operand=0.0
            ),
            follow_condition=Condition(
                left_operand="close", operator="<", right_operand="SMA_20"
            ),
            lookback_bars=3,
        )
        make_strategy.gene.stateful_conditions.append(sc2)
        evaluator = StatefulConditionsEvaluator(make_strategy)
        evaluator.process_stateful_triggers()

        assert (
            make_strategy.condition_evaluator.check_and_record_trigger.call_count == 2
        )

    # ----------------------------------------------------------------
    # check_stateful_conditions
    # ----------------------------------------------------------------
    def test_check_stateful_conditions_returns_true_when_any_matches(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """いずれかの条件が成立すれば True"""
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            True
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)

        assert evaluator.check_stateful_conditions() is True

    def test_check_stateful_conditions_returns_false_when_none_matches(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """どの条件も成立しなければ False"""
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            False
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)

        assert evaluator.check_stateful_conditions() is False

    def test_check_stateful_conditions_short_circuits(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """最初に成立した時点で以降の評価を打ち切る"""
        sc2 = StatefulCondition(
            trigger_condition=Condition(
                left_operand="MACD", operator=">", right_operand=0.0
            ),
            follow_condition=Condition(
                left_operand="close", operator="<", right_operand="SMA_20"
            ),
            lookback_bars=3,
        )
        make_strategy.gene.stateful_conditions.append(sc2)
        # 1つ目だけ True
        make_strategy.condition_evaluator.evaluate_stateful_condition.side_effect = [
            True,
            False,
        ]
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.check_stateful_conditions() is True
        # 1回だけ呼ばれる(2回目は呼ばれない)
        assert (
            make_strategy.condition_evaluator.evaluate_stateful_condition.call_count
            == 1
        )

    def test_check_stateful_conditions_returns_false_when_gene_none(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """strategy.gene が None の場合は False"""
        make_strategy.gene = None
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.check_stateful_conditions() is False

    def test_check_stateful_conditions_skips_disabled(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """enabled=False の条件は評価されない"""
        make_strategy.gene.stateful_conditions[0].enabled = False
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.check_stateful_conditions() is False
        make_strategy.condition_evaluator.evaluate_stateful_condition.assert_not_called()

    # ----------------------------------------------------------------
    # get_stateful_entry_direction
    # ----------------------------------------------------------------
    def test_get_entry_direction_long_default(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """direction 属性がない、もしくは 'long' のとき 1.0 を返す"""
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            True
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() == 1.0

    def test_get_entry_direction_explicit_long(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """direction='long' 明示時も 1.0"""
        make_strategy.gene.stateful_conditions[0].direction = "long"
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            True
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() == 1.0

    def test_get_entry_direction_short(self, make_strategy: SimpleNamespace) -> None:
        """direction='short' のとき -1.0 を返す"""
        make_strategy.gene.stateful_conditions[0].direction = "short"
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            True
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() == -1.0

    def test_get_entry_direction_returns_none_when_not_satisfied(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """条件不成立なら None"""
        make_strategy.condition_evaluator.evaluate_stateful_condition.return_value = (
            False
        )
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() is None

    def test_get_entry_direction_returns_none_when_gene_none(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """strategy.gene が None の場合は None"""
        make_strategy.gene = None
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() is None

    def test_get_entry_direction_skips_disabled(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """enabled=False の条件は評価されない"""
        make_strategy.gene.stateful_conditions[0].enabled = False
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() is None
        make_strategy.condition_evaluator.evaluate_stateful_condition.assert_not_called()

    def test_get_entry_direction_no_stateful_attr(
        self, make_strategy: SimpleNamespace
    ) -> None:
        """gene に stateful_conditions 属性がない場合は None"""
        make_strategy.gene = SimpleNamespace()
        evaluator = StatefulConditionsEvaluator(make_strategy)
        assert evaluator.get_stateful_entry_direction() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
