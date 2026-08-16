"""
未知オペランドの評価セマンティクスのテスト

参照切れ（dangling reference）の条件オペランドは 0 ではなく NaN 扱いとし、
比較結果が常に偽になることを検証します。
"""

import math

import pandas as pd

from app.services.auto_strategy.core.evaluation.condition_evaluator import (
    ConditionEvaluator,
)
from app.services.auto_strategy.genes import Condition


class _FakeStrategy:
    """指標列を持たない最小の戦略インスタンス"""

    def __init__(self):
        self.data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 1100, 1200],
            }
        )
        self.indicators: dict = {}


class TestUnknownOperandSemantics:
    def setup_method(self):
        self.evaluator = ConditionEvaluator()
        self.strategy = _FakeStrategy()

    def test_unknown_operand_value_is_nan(self):
        """未知の識別子は 0 ではなく NaN を返す"""
        value = self.evaluator.get_condition_value("SMA_deadbeef", self.strategy)
        assert math.isnan(value)

    def test_close_vs_unknown_is_always_false(self):
        """``close > 未知列`` は常に偽（0 扱いだと常に真になってしまう）"""
        condition = Condition(
            left_operand="close", operator=">", right_operand="SMA_deadbeef"
        )
        assert (
            self.evaluator.evaluate_single_condition(condition, self.strategy) is False
        )

    def test_unknown_vs_threshold_is_always_false(self):
        """``未知列 > 50`` も常に偽"""
        condition = Condition(
            left_operand="SMA_deadbeef", operator=">", right_operand=50
        )
        assert (
            self.evaluator.evaluate_single_condition(condition, self.strategy) is False
        )

    def test_numeric_string_operand_still_resolves(self):
        """数値文字列オペランドは引き続き数値として解決される"""
        value = self.evaluator.get_condition_value("42.5", self.strategy)
        assert value == 42.5
