"""
Indicator Serviceテスト

IndicatorCalculatorとIndicatorCalculatorの機能をテストします。
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from app.services.auto_strategy.services.indicator_service import IndicatorCalculator
from app.services.auto_strategy.models.strategy_models import IndicatorGene


class TestIndicatorCalculator(unittest.TestCase):
    """IndicatorCalculatorテスト"""

    def setUp(self):
        """セットアップ"""
        self.technical_indicator_service_mock = Mock()
        self.ml_orchestrator_mock = Mock()
        self.calculator = IndicatorCalculator(
            ml_orchestrator=self.ml_orchestrator_mock,
            technical_indicator_service=self.technical_indicator_service_mock
        )

    def test_calculate_indicator_valid_data(self):
        """有効なデータでの指標計算テスト"""
        # モックデータ作成
        mock_data = Mock()
        mock_data.df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })

        params = {'period': 14}
        self.technical_indicator_service_mock.calculate_indicator.return_value = np.array([10, 20, 30])

        result = self.calculator.calculate_indicator(mock_data, 'RSI', params)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
        self.technical_indicator_service_mock.calculate_indicator.assert_called_once_with(
            mock_data.df, 'RSI', params
        )

    def test_calculate_indicator_empty_data(self):
        """空データでの指標計算テスト - safe_operationがNone返すか確認"""
        mock_data = Mock()
        mock_data.df = pd.DataFrame()

        result = self.calculator.calculate_indicator(mock_data, 'RSI', {})

        self.assertIsNone(result)

    def test_calculate_indicator_null_data(self):
        """Nullデータオブジェクトでの指標計算テスト"""
        result = self.calculator.calculate_indicator(None, 'RSI', {})
        self.assertIsNone(result)

    def test_calculate_indicator_ml_indicator(self):
        """ML指標計算テスト"""
        mock_data = Mock()
        mock_data.df = pd.DataFrame({'Close': [100, 101, 102]})

        params = {'period': 14}
        self.ml_orchestrator_mock.calculate_single_ml_indicator.return_value = np.array([0.1, 0.2, 0.3])

        result = self.calculator.calculate_indicator(mock_data, 'ML_RSI', params)

        self.assertIsInstance(result, np.ndarray)
        self.ml_orchestrator_mock.calculate_single_ml_indicator.assert_called_once_with(
            'ML_RSI', mock_data.df, funding_rate_data=None, open_interest_data=None
        )

    def test_calculate_indicator_tuple_return(self):
        """複数出力指標（tuple戻り値）計算テスト"""
        mock_data = Mock()
        mock_data.df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })

        params = {}
        self.technical_indicator_service_mock.calculate_indicator.return_value = (
            np.array([10, 20, 30]),
            np.array([40, 50, 60])
        )

        result = self.calculator.calculate_indicator(mock_data, 'MACD', params)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_init_indicator_single_output(self):
        """単一出力指標初期化テスト"""
        # mock 戦略インスタンスを作成
        strategy_instance = MagicMock()
        strategy_instance.indicators = {}

        indicator_gene = IndicatorGene(type='RSI', parameters={'period': 14})

        mock_data = MagicMock()
        mock_data.df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        strategy_instance.data = mock_data

        self.technical_indicator_service_mock.calculate_indicator.return_value = np.array([50, 55, 60])

        # テスト実行
        self.calculator.init_indicator(indicator_gene, strategy_instance)

        # 指標が設定されているか確認
        self.assertTrue(hasattr(strategy_instance, 'RSI'))
        self.assertIn('RSI', strategy_instance.indicators)

    def test_init_indicator_tuple_output(self):
        """複数出力指標初期化テスト"""
        strategy_instance = MagicMock()
        strategy_instance.indicators = {}

        indicator_gene = IndicatorGene(type='MACD', parameters={})

        mock_data = MagicMock()
        mock_data.df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        strategy_instance.data = mock_data

        self.technical_indicator_service_mock.calculate_indicator.return_value = (
            np.array([10, 20, 30]),
            np.array([40, 50, 60])
        )

        self.calculator.init_indicator(indicator_gene, strategy_instance)

        # 複数指標が設定されているか確認
        self.assertTrue(hasattr(strategy_instance, 'MACD_0'))
        self.assertTrue(hasattr(strategy_instance, 'MACD_1'))
        self.assertIn('MACD_0', strategy_instance.indicators)
        self.assertIn('MACD_1', strategy_instance.indicators)

    def test_init_indicator_null_result(self):
        """Null結果での指標初期化テスト - safe_operationがNone返すか確認"""
        strategy_instance = MagicMock()
        strategy_instance.data = None

        indicator_gene = IndicatorGene(type='INVALID', parameters={})

        # 初期化はNone返す、raiseなし
        self.calculator.init_indicator(indicator_gene, strategy_instance)


if __name__ == '__main__':
    unittest.main()