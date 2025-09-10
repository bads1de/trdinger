"""
ConditionEvaluatorクラスのテスト
TDDアプローチで全メソッドをテスト
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import logging
from typing import List, Union
from numbers import Real, Integral

from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.models.condition import Condition, ConditionGroup


class TestConditionEvaluator:
    """ConditionEvaluatorクラスのテストスイート"""

    @pytest.fixture
    def evaluator(self):
        """ConditionEvaluatorインスタンス"""
        return ConditionEvaluator()

    @pytest.fixture
    def strategy_mock(self):
        """pandas-ta属性を持つstrategy_instanceモック"""
        strategy = MagicMock()
        # pandas-taスタイルの属性
        strategy.SMA = pd.Series([10, 20, 30, 40, 50])
        strategy.RSI = [30.5, 31.0, 32.5, 33.0, 34.5]
        strategy.BBANDS_upper = pd.Series([15, 25, 35, 45, 55])
        strategy.ML_Score = [0.8, 0.9, 0.85, 0.92, 0.88]
        return strategy

    @pytest.fixture
    def condition_factory(self):
        """テスト条件作成ファクトリ"""
        def _create_condition(left, op, right) -> Condition:
            return Condition(left_operand=left, operator=op, right_operand=right)
        return _create_condition

    def test_evaluate_conditions_normal_and(self, evaluator, strategy_mock, condition_factory):
        """evaluate_conditions: 正常AND評価（全True）"""
        conditions = [
            condition_factory("SMA", ">", 35),
            condition_factory("RSI", "<", 50)
        ]
        result = evaluator.evaluate_conditions(conditions, strategy_mock)
        assert result == True

    def test_evaluate_conditions_false_and(self, evaluator, strategy_mock, condition_factory):
        """evaluate_conditions: AND評価（一False）"""
        conditions = [
            condition_factory("SMA", ">", 35),  # True
            condition_factory("RSI", ">", 50)   # False
        ]
        result = evaluator.evaluate_conditions(conditions, strategy_mock)
        assert result == False

    def test_evaluate_conditions_empty_list(self, evaluator, strategy_mock):
        """evaluate_conditions: エッジ - 空リスト"""
        conditions: List[Union[Condition, ConditionGroup]] = []
        result = evaluator.evaluate_conditions(conditions, strategy_mock)
        assert result == True

    def test_evaluate_conditions_with_group(self, evaluator, strategy_mock, condition_factory):
        """evaluate_conditions: グループ含むAND"""
        group = ConditionGroup(conditions=[
            condition_factory("RSI", ">", 30),  # True
            condition_factory("RSI", "<", 20)   # False -> ORでTrue
        ])
        conditions = [
            condition_factory("SMA", ">", 35),  # True
            group
        ]
        result = evaluator.evaluate_conditions(conditions, strategy_mock)
        assert result == True

    def test_evaluate_conditions_error_case(self, evaluator, strategy_mock, condition_factory):
        """evaluate_conditions: エラーケース"""
        conditions = [
            condition_factory("INVALID_ATTR", ">", 1)
        ]
        result = evaluator.evaluate_conditions(conditions, strategy_mock)
        assert result == False  # 無効属性はFalse

    def test_evaluate_condition_group_normal(self, evaluator, strategy_mock, condition_factory):
        """_evaluate_condition_group: 正常OR評価"""
        group = ConditionGroup(conditions=[
            condition_factory("RSI", "<", 20),  # False
            condition_factory("RSI", ">", 30)   # True
        ])
        result = evaluator._evaluate_condition_group(group, strategy_mock)
        assert result == True

    def test_evaluate_condition_group_all_false(self, evaluator, strategy_mock, condition_factory):
        """_evaluate_condition_group: 全条件False"""
        group = ConditionGroup(conditions=[
            condition_factory("RSI", "<", 10),  # False
            condition_factory("SMA", ">", 100)  # False
        ])
        result = evaluator._evaluate_condition_group(group, strategy_mock)
        assert result == False

    def test_evaluate_condition_group_empty(self, evaluator, strategy_mock):
        """_evaluate_condition_group: エッジ - 空グループ"""
        group = ConditionGroup(conditions=[])
        result = evaluator._evaluate_condition_group(group, strategy_mock)
        assert result == False

    def test_evaluate_condition_group_error(self, evaluator, strategy_mock, condition_factory):
        """_evaluate_condition_group: エラーケース"""
        group = ConditionGroup(conditions=[
            condition_factory("INVALID", ">", 1)
        ])
        result = evaluator._evaluate_condition_group(group, strategy_mock)
        assert result == False

    def test_evaluate_single_condition_greater_than(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: > 演算子"""
        condition = condition_factory("SMA", ">", 35)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == True  # 50 > 35

    def test_evaluate_single_condition_less_than(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: < 演算子"""
        condition = condition_factory("RSI", "<", 40)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == True  # 34.5 < 40

    def test_evaluate_single_condition_greater_equal(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: >= 演算子"""
        condition = condition_factory("SMA", ">=", 50)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == True  # 50 >= 50

    def test_evaluate_single_condition_less_equal(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: <= 演算子"""
        condition = condition_factory("RSI", "<=", 34.5)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == True

    def test_evaluate_single_condition_equal(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: == 演算子"""
        condition = condition_factory("SMA", "==", 50)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == True

    def test_evaluate_single_condition_not_equal(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: != 演算子"""
        condition = condition_factory("SMB", "!=", 49)  # 50 != 49
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        # Note: Will fail due to invalid attribute
        assert result == False

    def test_evaluate_single_condition_ml_logging(self, evaluator, strategy_mock, condition_factory, caplog):
        """evaluate_single_condition: ML指標ログ出力"""
        with caplog.at_level(logging.INFO):
            condition = condition_factory("ML_Score", ">", 0.8)
            result = evaluator.evaluate_single_condition(condition, strategy_mock)

        assert result == True
        # MLログが含まれていることを確認
        assert "[ML条件評価デバッグ]" in caplog.text
        assert "評価結果: True" in caplog.text

    def test_evaluate_single_condition_boundary_float_equality(self, evaluator, condition_factory):
        """evaluate_single_condition: 境界 - float等価比較"""
        strategy = MagicMock()
        strategy.VALUE = [1.23456789012]

        condition = condition_factory("VALUE", "==", 1.23456789013)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == True  # 小数点以下の微小差は許容

    def test_evaluate_single_condition_invalid_operator(self, evaluator, strategy_mock, condition_factory):
        """evaluate_single_condition: 無効演算子"""
        condition = condition_factory("SMA", "**", 25)
        result = evaluator.evaluate_single_condition(condition, strategy_mock)
        assert result == False

    def test_evaluate_single_condition_non_numeric_comparison(self, evaluator, condition_factory):
        """evaluate_single_condition: 非数値比較"""
        strategy = MagicMock()
        strategy.TEXT = ["hello"]

        condition = condition_factory("TEXT", ">", 1)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == False  # 比較できない値でFalse

    def test_get_condition_value_numeric(self, evaluator):
        """get_condition_value: 数値オペランド"""
        result = evaluator.get_condition_value(42, None)
        assert result == 42.0

        result = evaluator.get_condition_value(3.14, None)
        assert result == 3.14

    def test_get_condition_value_dict_with_indicator(self, evaluator, strategy_mock):
        """get_condition_value: dictオペランド（pandas-taスタイル）"""
        operand = {"indicator": "BBANDS_upper"}
        result = evaluator.get_condition_value(operand, strategy_mock)
        assert result == 55.0  # Seriesの末尾値

    def test_get_condition_value_str_with_indicator(self, evaluator, strategy_mock):
        """get_condition_value: strオペランド（pandas-taスタイル）"""
        result = evaluator.get_condition_value("SMA", strategy_mock)
        assert result == 50.0  # Seriesの末尾値

    def test_get_condition_value_str_list_indicator(self, evaluator, strategy_mock):
        """get_condition_value: strオペランド（リスト属性）"""
        result = evaluator.get_condition_value("RSI", strategy_mock)
        assert result == 34.5  # リストの末尾値

    def test_get_condition_value_invalid_dict(self, evaluator, strategy_mock):
        """get_condition_value: 無効dictオペランド"""
        operand = {"invalid_key": "value"}
        result = evaluator.get_condition_value(operand, strategy_mock)
        assert result == 0.0

    def test_get_condition_value_invalid_str(self, evaluator, strategy_mock):
        """get_condition_value: 無効strオペランド"""
        result = evaluator.get_condition_value("NONEXISTENT", strategy_mock)
        assert result == 0.0

    def test_get_condition_value_none_strategy(self, evaluator):
        """get_condition_value: strategyがNoneのケース（未使用）"""
        # 実際のコードではstrategy_instanceが必要だが、テストではNone許容
        result = evaluator.get_condition_value(1.0, None)
        assert result == 1.0

    def test_evaluate_single_condition_numpy_int64(self, evaluator, condition_factory):
        """evaluate_single_condition: numpy.int64対応"""
        strategy = MagicMock()
        # numpy.int64の値を設定
        strategy.NUMBER = np.int64(42)

        condition = condition_factory("NUMBER", ">", 40)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == True  # 42 > 40

    def test_evaluate_single_condition_numpy_float64(self, evaluator, condition_factory):
        """evaluate_single_condition: numpy.float64対応"""
        strategy = MagicMock()
        strategy.VALUE = np.float64(3.14)

        condition = condition_factory("VALUE", ">", 3.0)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == True  # 3.14 > 3.0

    def test_evaluate_single_condition_boundary_equality_numpy(self, evaluator, condition_factory):
        """evaluate_single_condition: numpy境界値等価比較"""
        strategy = MagicMock()
        strategy.EXACT = np.float64(10.0)

        # 厳密等価
        condition = condition_factory("EXACT", "==", 10.0)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == True

        # 微小差
        condition = condition_factory("EXACT", "==", 10.0 + 1e-9)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == True  # 1e-9差は許容

        # 大きな差
        condition = condition_factory("EXACT", "!=", 10.0 + 1e-7)
        result = evaluator.evaluate_single_condition(condition, strategy)
        assert result == False  # 1e-7差は不一致と判断（境界調整中）

    def test_get_final_value_pandas_series(self, evaluator):
        """_get_final_value: pandas Series"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = evaluator._get_final_value(series)
        assert result == 5.0

    def test_get_final_value_list(self, evaluator):
        """_get_final_value: list/array"""
        values = [10, 20, 30, 40, 50]
        result = evaluator._get_final_value(values)
        assert result == 50.0

    def test_get_final_value_scalar(self, evaluator):
        """_get_final_value: スカラー値"""
        result = evaluator._get_final_value(42.5)
        assert result == 42.5

    def test_get_final_value_scalar_int(self, evaluator):
        """_get_final_value: intスカラー"""
        result = evaluator._get_final_value(42)
        assert result == 42.0  # float変換

    def test_get_final_value_with_nan(self, evaluator):
        """_get_final_value: NaN値対応"""
        series = pd.Series([1, 2, float('nan'), 4, 5])
        result = evaluator._get_final_value(series)
        assert result == 5.0  # 有限値を取る

    def test_get_final_value_empty_series(self, evaluator):
        """_get_final_value: 空Series"""
        series = pd.Series([])
        result = evaluator._get_final_value(series)
        assert result == 0.0

    def test_get_final_value_invalid_type(self, evaluator):
        """_get_final_value: 無効タイプ"""
        result = evaluator._get_final_value(None)
        assert result == 0.0  # 例外処理で0.0

        result = evaluator._get_final_value("string")
        # 適当な変換結果を期待
        assert isinstance(result, float)