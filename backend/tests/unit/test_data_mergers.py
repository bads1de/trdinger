#!/usr/bin/env python3
"""
データマージャークラスのテスト

新しく作成されたOIMerger、FRMerger、FearGreedMergerクラスのテストを行います。
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.data_mergers import OIMerger, FRMerger, FearGreedMerger


class TestOIMerger(unittest.TestCase):
    """OIMergerのテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_oi_repo = Mock()
        self.merger = OIMerger(self.mock_oi_repo)

    def test_merge_oi_data_success(self):
        """OIデータマージ成功テスト"""
        # テストデータ
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='h'))

        # モックOIデータ
        mock_oi_data = [
            Mock(open_interest_value=1000, data_timestamp=datetime(2024, 1, 1)),
            Mock(open_interest_value=1100, data_timestamp=datetime(2024, 1, 1, 1)),
        ]
        self.mock_oi_repo.get_open_interest_data.return_value = mock_oi_data

        # テスト実行
        result_df = self.merger.merge_oi_data(
            test_df, "BTC/USDT", datetime(2024, 1, 1), datetime(2024, 1, 1, 3)
        )

        # 結果の検証
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('open_interest', result_df.columns)
        self.mock_oi_repo.get_open_interest_data.assert_called_once()

    def test_merge_oi_data_no_data(self):
        """OIデータが存在しない場合のテスト"""
        test_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        self.mock_oi_repo.get_open_interest_data.return_value = None

        result_df = self.merger.merge_oi_data(
            test_df, "BTC/USDT", datetime(2024, 1, 1), datetime(2024, 1, 1, 2)
        )

        self.assertIn('open_interest', result_df.columns)
        self.assertTrue(result_df['open_interest'].isna().all())


class TestFRMerger(unittest.TestCase):
    """FRMergerのテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_fr_repo = Mock()
        self.merger = FRMerger(self.mock_fr_repo)

    def test_merge_fr_data_success(self):
        """FRデータマージ成功テスト"""
        test_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        mock_fr_data = [
            Mock(funding_rate=0.001, funding_timestamp=datetime(2024, 1, 1)),
        ]
        self.mock_fr_repo.get_funding_rate_data.return_value = mock_fr_data

        result_df = self.merger.merge_fr_data(
            test_df, "BTC/USDT", datetime(2024, 1, 1), datetime(2024, 1, 1, 2)
        )

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('funding_rate', result_df.columns)
        self.mock_fr_repo.get_funding_rate_data.assert_called_once()


class TestFearGreedMerger(unittest.TestCase):
    """FearGreedMergerのテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.mock_fear_greed_repo = Mock()
        self.merger = FearGreedMerger(self.mock_fear_greed_repo)

    def test_merge_fear_greed_data_success(self):
        """Fear & Greedデータマージ成功テスト"""
        test_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))

        mock_fear_greed_data = [
            Mock(value=50, value_classification="Neutral", data_timestamp=datetime(2024, 1, 1)),
        ]
        self.mock_fear_greed_repo.get_fear_greed_data.return_value = mock_fear_greed_data

        result_df = self.merger.merge_fear_greed_data(
            test_df, datetime(2024, 1, 1), datetime(2024, 1, 1, 2)
        )

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('fear_greed_value', result_df.columns)
        self.assertIn('fear_greed_classification', result_df.columns)
        self.mock_fear_greed_repo.get_fear_greed_data.assert_called_once()

    def test_merge_fear_greed_data_detailed_logging(self):
        """Fear & Greedデータマージ（詳細ログ）テスト"""
        test_df = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [103],
            'Volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='h'))

        mock_fear_greed_data = [
            Mock(value=60, value_classification="Greed", data_timestamp=datetime(2024, 1, 1)),
        ]
        self.mock_fear_greed_repo.get_fear_greed_data.return_value = mock_fear_greed_data

        result_df = self.merger.merge_fear_greed_data(
            test_df, datetime(2024, 1, 1), datetime(2024, 1, 1, 1), detailed_logging=True
        )

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('fear_greed_value', result_df.columns)
        self.assertIn('fear_greed_classification', result_df.columns)


if __name__ == '__main__':
    unittest.main()
