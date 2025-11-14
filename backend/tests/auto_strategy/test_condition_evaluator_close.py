#!/usr/bin/env python3
"""
condition_evaluatorのcloseオペランド取得機能をテスト
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock

import pandas as pd

from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator
from app.services.auto_strategy.models.strategy_models import Condition


class TestConditionEvaluatorClose:
    """ConditionEvaluatorのcloseオペランド取得テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.evaluator = ConditionEvaluator()

    def test_close_operand_with_pandas_dataframe(self):
        """pandas DataFrameでのcloseオペランド取得テスト"""
        # Mock strategy instance with pandas DataFrame data
        mock_strategy = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_strategy.data = mock_data

        # Test close operand
        result = self.evaluator.get_condition_value("close", mock_strategy)

        # Should return the last close value
        assert result == 105.0

    def test_open_operand_with_pandas_dataframe(self):
        """pandas DataFrameでのopenオペランド取得テスト"""
        # Mock strategy instance with pandas DataFrame data
        mock_strategy = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_strategy.data = mock_data

        # Test open operand
        result = self.evaluator.get_condition_value("open", mock_strategy)

        # Should return the last open value
        assert result == 102.0

    def test_close_operand_with_backtesting_data(self):
        """backtesting.py形式のデータ構造でのcloseオペランド取得テスト"""
        # Mock strategy instance with backtesting.py style data
        mock_strategy = Mock()
        mock_data = Mock()
        # backtesting.pyでは属性アクセスでデータを取得
        mock_data.Close = pd.Series([103.0, 104.0, 105.0])
        mock_strategy.data = mock_data

        # Test close operand
        result = self.evaluator.get_condition_value("close", mock_strategy)

        # Should return the last close value
        assert result == 105.0

    def test_numeric_operand_unchanged(self):
        """数値オペランドは変更されないことをテスト"""
        mock_strategy = Mock()

        result = self.evaluator.get_condition_value(100.5, mock_strategy)

        assert result == 100.5

    def test_string_operand_attribute_access(self):
        """文字列オペランドが戦略インスタンスの属性としてアクセスされることをテスト"""
        mock_strategy = Mock()
        mock_strategy.sma_value = 123.45

        result = self.evaluator.get_condition_value("sma_value", mock_strategy)

        assert result == 123.45

    def test_unknown_operand_returns_zero(self):
        """未知のオペランドは0を返すことをテスト"""
        mock_strategy = Mock()

        result = self.evaluator.get_condition_value("unknown_indicator", mock_strategy)

        assert result == 0.0

    def test_condition_evaluation_with_close_operand(self):
        """closeオペランドを使用した条件評価テスト"""
        # Mock strategy with close data
        mock_strategy = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_strategy.data = mock_data

        # Create condition: close > 100
        condition = Condition(left_operand="close", operator=">", right_operand=100)

        # Evaluate condition
        result = self.evaluator.evaluate_single_condition(condition, mock_strategy)

        # Should be True because last close (105.0) > 100
        assert result

    def test_condition_evaluation_close_vs_open(self):
        """close vs openの条件評価テスト"""
        # Mock strategy with data where close > open
        mock_strategy = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],  # close > open
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_strategy.data = mock_data

        # Create condition: close > open
        condition = Condition(left_operand="close", operator=">", right_operand="open")

        # Evaluate condition
        result = self.evaluator.evaluate_single_condition(condition, mock_strategy)

        # Should be True because last close (105.0) > last open (102.0)
        assert result

    def test_error_handling_with_invalid_operand(self):
        """無効なオペランドでのエラーハンドリングテスト"""
        mock_strategy = Mock()

        # 無効なオペランドを渡す - デコレーターにより0.0が返されるはず
        result = self.evaluator.get_condition_value(None, mock_strategy)

        # エラーが発生してもデコレーターにより0.0が返される
        assert result == 0.0

    def test_error_handling_with_invalid_strategy_instance(self):
        """無効な戦略インスタンスでのエラーハンドリングテスト"""
        # 条件付きでNoneを渡すとエラーが発生するはず
        from app.services.auto_strategy.models.strategy_models import Condition

        condition = Condition(left_operand="close", operator=">", right_operand=100)

        # 無効な戦略インスタンスを渡す
        result = self.evaluator.evaluate_conditions([condition], None)

        # エラーが発生してもデコレーターによりFalseが返される
        assert not result

    def test_error_handling_in_condition_group(self):
        """条件グループでのエラーハンドリングテスト"""
        from app.services.auto_strategy.models.strategy_models import ConditionGroup

        # 空の条件グループを渡す
        group = ConditionGroup(conditions=[])
        mock_strategy = Mock()

        result = self.evaluator._evaluate_condition_group(group, mock_strategy)

        # 空のグループなのでFalseが返される
        assert not result
