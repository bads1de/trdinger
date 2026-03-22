import unittest
import numpy as np
from unittest.mock import MagicMock
from app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator


class TestConditionEvaluator0D(unittest.TestCase):
    def setUp(self):
        self.evaluator = ConditionEvaluator()

    def test_get_final_value_0d_array(self):
        """0次元Numpy配列（スカラー）を渡した場合のテスト"""
        val = np.array(10.5)
        result = self.evaluator._get_final_value(val)
        self.assertEqual(result, 10.5)
        self.assertIsInstance(result, float)

    def test_get_final_value_1d_array(self):
        """通常の1次元Numpy配列を渡した場合のテスト"""
        val = np.array([10.5, 20.5])
        result = self.evaluator._get_final_value(val)
        self.assertEqual(result, 20.5)

    def test_get_condition_value_0d_indicator(self):
        """インジケーターが0次元配列を返す場合のテスト"""
        strategy = MagicMock()
        strategy.indicators = {"test_ind": np.array(42.0)}

        result = self.evaluator.get_condition_value("test_ind", strategy)
        self.assertEqual(result, 42.0)

    def test_get_condition_value_0d_ohlcv(self):
        """OHLCVが0次元配列を返す場合の特殊なケース（通常はないが防止策のテスト）"""
        strategy = MagicMock()
        strategy.data.Close = np.array(100.0)

        result = self.evaluator.get_condition_value("close", strategy)
        self.assertEqual(result, 100.0)

    def test_get_previous_value_0d_array(self):
        """1つ前の値取得で0次元配列が渡された場合のテスト"""
        strategy = MagicMock()
        strategy.indicators = {"test_ind": np.array(10.0)}

        # 0次元配列には「1つ前」がないため NaN を返すべき
        result = self.evaluator._get_previous_value("test_ind", strategy)
        self.assertTrue(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
