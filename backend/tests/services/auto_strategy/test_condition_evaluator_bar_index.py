"""
スカラー条件評価のバー位置セマンティクスのテスト

backtesting.py は ``strategy.data`` を現在バーまでに切り詰める一方、
インジケーターはフル長 Series のまま戦略インスタンスに登録される。
スカラー評価（ベクトル化フォールバック）は現在バー位置
``len(data) - 1`` でインジケーターを読む必要があり、末尾を読むと
データセット末尾＝未来の値を使う先読みになることを検証する。
"""

import math

import numpy as np
import pandas as pd

from app.services.auto_strategy.core.evaluation.condition_evaluator import (
    ConditionEvaluator,
)
from app.services.auto_strategy.genes import Condition


class _TruncatedDataStrategy:
    """data が現在バーまで切り詰められた状態を模した戦略インスタンス"""

    def __init__(self, data_length: int = 3, indicator_values=None):
        # backtesting.py の _Data 相当: len(data) が現在バー + 1
        self.data = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(data_length)],
                "High": [105.0 + i for i in range(data_length)],
                "Low": [95.0 + i for i in range(data_length)],
                "Close": [103.0 + i for i in range(data_length)],
                "Volume": [1000 + i for i in range(data_length)],
            }
        )
        if indicator_values is None:
            indicator_values = [10.0, 20.0, 30.0, 999.0, 999.0]
        self.indicators: dict = {"sma_test": pd.Series(indicator_values)}


class TestCurrentBarIndexRead:
    def setup_method(self):
        self.evaluator = ConditionEvaluator()

    def test_indicator_reads_current_bar_not_dataset_end(self):
        """インジケーターは現在バー位置の値を読み、末尾（未来）を読まない"""
        # data は 3 バー目まで（index 2 が現在バー）、sma はフル長 5
        # 末尾 2 件は 999.0 で、旧実装（[-1]）だと 999.0 を読んでいた
        strategy = _TruncatedDataStrategy(data_length=3)

        value = self.evaluator.get_condition_value("sma_test", strategy)

        assert value == 30.0  # index 2 の値（末尾 999.0 ではない）

    def test_condition_false_when_only_future_value_satisfies(self):
        """現在バーでは不成立・末尾（未来）でのみ成立する条件は False"""
        strategy = _TruncatedDataStrategy(data_length=3)
        # 現在バーの sma=30.0 は 50 を超えないが、末尾は 999.0
        condition = Condition(left_operand="sma_test", operator=">", right_operand=50.0)

        assert self.evaluator.evaluate_single_condition(condition, strategy) is False

    def test_condition_true_when_current_bar_satisfies(self):
        """現在バーの値で成立する条件は True"""
        strategy = _TruncatedDataStrategy(data_length=3)
        condition = Condition(left_operand="sma_test", operator=">", right_operand=25.0)

        assert self.evaluator.evaluate_single_condition(condition, strategy) is True

    def test_numpy_array_indicator_reads_current_bar(self):
        """ndadbge 型インジケーターも現在バー位置で読む"""
        strategy = _TruncatedDataStrategy(data_length=3)
        strategy.indicators["atr_test"] = np.array([1.0, 2.0, 3.0, 999.0, 999.0])

        value = self.evaluator.get_condition_value("atr_test", strategy)

        assert value == 3.0

    def test_out_of_range_bar_returns_nan_and_condition_false(self):
        """インジケーター長が現在バー位置に満たない場合は NaN→False（安全側）"""
        strategy = _TruncatedDataStrategy(data_length=5)
        strategy.indicators["short_series"] = pd.Series([1.0, 2.0])

        value = self.evaluator.get_condition_value("short_series", strategy)
        assert math.isnan(value)

        condition = Condition(
            left_operand="short_series", operator=">", right_operand=0.0
        )
        assert self.evaluator.evaluate_single_condition(condition, strategy) is False


class TestPreviousValueBarIndex:
    def setup_method(self):
        self.evaluator = ConditionEvaluator()

    def test_previous_value_reads_bar_before_current(self):
        """1つ前の値は現在バーの1つ前（index 1）で、末尾-2（未来側）を読まない"""
        # data 3 バー → 現在=2, prev=1（値 20.0）。旧実装の iloc[-2] は 999.0
        strategy = _TruncatedDataStrategy(data_length=3)

        prev = self.evaluator._get_previous_value("sma_test", strategy)

        assert prev == 20.0

    def test_cross_up_uses_causal_values(self):
        """CROSS_UP 判定も現在バー基準で行われる"""
        # index 1: left(10) <= right(20), index 2: left(30) > right(20) → CROSS_UP
        strategy = _TruncatedDataStrategy(data_length=3)
        strategy.indicators["left_series"] = pd.Series([5.0, 10.0, 30.0, 999.0, 999.0])
        strategy.indicators["sma_test"] = pd.Series([20.0, 20.0, 20.0, 1.0, 1.0])

        condition = Condition(
            left_operand="left_series",
            operator="CROSS_UP",
            right_operand="sma_test",
        )

        assert self.evaluator.evaluate_single_condition(condition, strategy) is True

    def test_cross_up_false_when_cross_only_in_future(self):
        """未来（末尾側）でしか交差しない場合は CROSS_UP にならない"""
        # 現在バー index 2 では 30 > 20 だが index 1 は 10 <= 20 で交差済み…
        # 末尾側（999 vs 1）でも交差するが、現在バーの交差は index 2 起点:
        # index1: 40<=20 False → index2: 30>20 かつ 40>20? 交差ではない
        strategy = _TruncatedDataStrategy(data_length=3)
        strategy.indicators["left_series"] = pd.Series([50.0, 40.0, 30.0, 999.0, 999.0])
        strategy.indicators["sma_test"] = pd.Series([20.0, 20.0, 20.0, 1.0, 1.0])

        condition = Condition(
            left_operand="left_series",
            operator="CROSS_UP",
            right_operand="sma_test",
        )

        assert self.evaluator.evaluate_single_condition(condition, strategy) is False


class TestNoDataFallback:
    def setup_method(self):
        self.evaluator = ConditionEvaluator()

    def test_without_data_attribute_reads_last_value(self):
        """data 属性がないコンテキストは旧来どおり末尾を読む（後方互換）"""

        class _NoDataStrategy:
            def __init__(self):
                self.indicators = {"sma_test": pd.Series([1.0, 2.0, 3.0])}

        strategy = _NoDataStrategy()
        value = self.evaluator.get_condition_value("sma_test", strategy)
        assert value == 3.0

        prev = self.evaluator._get_previous_value("sma_test", strategy)
        assert prev == 2.0

    def test_full_length_data_reads_last_value(self):
        """data がフル長（バックテスト外）の場合は末尾読みと同値"""
        # data 長 = インジケーター長 → bar_index = len-1 = 末尾
        strategy = _TruncatedDataStrategy(data_length=5)

        value = self.evaluator.get_condition_value("sma_test", strategy)

        assert value == 999.0
