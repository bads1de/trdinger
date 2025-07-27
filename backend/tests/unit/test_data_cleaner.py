#!/usr/bin/env python3
"""
DataCleanerクラスのテスト

新しく作成されたDataCleanerクラスのテストを行います。
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.utils import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """DataCleanerのテスト"""

    def test_interpolate_oi_fr_data(self):
        """OI/FRデータ補間テスト"""
        # 欠損値を含むテストデータ
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'open_interest': [1000.0, None, 1200.0],
            'funding_rate': [0.001, None, 0.003]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        result_df = DataCleaner.interpolate_oi_fr_data(test_df)

        # 欠損値が補間されたことを確認
        self.assertFalse(result_df['open_interest'].isna().any())
        self.assertFalse(result_df['funding_rate'].isna().any())

        # 前方補間が正しく動作したことを確認
        self.assertEqual(result_df.loc[result_df.index[1], 'open_interest'], 1000.0)
        self.assertEqual(result_df.loc[result_df.index[1], 'funding_rate'], 0.001)

    def test_interpolate_fear_greed_data(self):
        """Fear & Greedデータ補間テスト"""
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'fear_greed_value': [50.0, None, 60.0],
            'fear_greed_classification': ["Neutral", None, "Greed"]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        result_df = DataCleaner.interpolate_fear_greed_data(test_df)

        # 欠損値が補間されたことを確認
        self.assertFalse(result_df['fear_greed_value'].isna().any())
        self.assertFalse(result_df['fear_greed_classification'].isna().any())

        # 前方補間が正しく動作したことを確認
        self.assertEqual(result_df.loc[result_df.index[1], 'fear_greed_value'], 50.0)
        self.assertEqual(result_df.loc[result_df.index[1], 'fear_greed_classification'], "Neutral")

    def test_interpolate_all_data(self):
        """全データ補間テスト"""
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'open_interest': [1000.0, None, 1200.0],
            'funding_rate': [0.001, None, 0.003],
            'fear_greed_value': [50.0, None, 60.0],
            'fear_greed_classification': ["Neutral", None, "Greed"]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        result_df = DataCleaner.interpolate_all_data(test_df)

        # すべての欠損値が補間されたことを確認
        self.assertFalse(result_df['open_interest'].isna().any())
        self.assertFalse(result_df['funding_rate'].isna().any())
        self.assertFalse(result_df['fear_greed_value'].isna().any())
        self.assertFalse(result_df['fear_greed_classification'].isna().any())

    def test_optimize_dtypes(self):
        """データ型最適化テスト"""
        test_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        # 元のデータ型を確認
        self.assertEqual(test_df['Open'].dtype, 'float64')
        self.assertEqual(test_df['Volume'].dtype, 'int64')

        result_df = DataCleaner.optimize_dtypes(test_df)

        # データ型が最適化されたことを確認
        self.assertEqual(result_df['Open'].dtype, 'float32')
        self.assertEqual(result_df['Volume'].dtype, 'int32')

    def test_validate_ohlcv_data(self):
        """OHLCVデータ検証テスト"""
        # 正常なデータ
        valid_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [103.0, 104.0],
            'Volume': [1000.0, 1100.0]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        # 正常なデータは例外が発生しない
        try:
            DataCleaner.validate_ohlcv_data(valid_df)
        except ValueError:
            self.fail("正常なデータで例外が発生しました")

        # 異常なデータ（High < Low）
        invalid_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [95.0, 96.0],  # High < Low
            'Low': [105.0, 106.0],
            'Close': [103.0, 104.0],
            'Volume': [1000.0, 1100.0]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        with self.assertRaises(ValueError):
            DataCleaner.validate_ohlcv_data(invalid_df)

    def test_clean_and_validate_data(self):
        """データクリーニングと検証の統合テスト"""
        test_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000.0, 1100.0, 1200.0],
            'open_interest': [1000.0, None, 1200.0],
            'funding_rate': [0.001, None, 0.003]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'open_interest', 'funding_rate']
        
        result_df = DataCleaner.clean_and_validate_data(
            test_df, required_columns, interpolate=True, optimize=True
        )

        # 補間が実行されたことを確認
        self.assertFalse(result_df['open_interest'].isna().any())
        self.assertFalse(result_df['funding_rate'].isna().any())

        # データ型最適化が実行されたことを確認
        self.assertEqual(result_df['Open'].dtype, 'float32')

        # ソートが実行されたことを確認
        self.assertTrue(result_df.index.is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
