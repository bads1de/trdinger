#!/usr/bin/env python3
"""
BacktestDataService リファクタリング後のテスト

重複コード解消後の動作確認を行います。
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.core.services.backtest_data_service import BacktestDataService


class TestBacktestDataServiceRefactor(unittest.TestCase):
    """BacktestDataService リファクタリング後のテスト"""

    def setUp(self):
        """テストセットアップ"""
        # モックリポジトリを作成
        self.mock_ohlcv_repo = Mock()
        self.mock_oi_repo = Mock()
        self.mock_fr_repo = Mock()
        self.mock_fear_greed_repo = Mock()

        # サービスインスタンスを作成
        self.service = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
            oi_repo=self.mock_oi_repo,
            fr_repo=self.mock_fr_repo,
            fear_greed_repo=self.mock_fear_greed_repo
        )

    def test_merge_fear_greed_data_unified_implementation(self):
        """Fear & Greedデータマージの統一実装テスト"""
        # テストデータの準備
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

        # Fear & Greedデータのモック
        mock_fear_greed_data = [
            Mock(timestamp=datetime(2024, 1, 1), value=50, classification="Neutral"),
            Mock(timestamp=datetime(2024, 1, 1, 1), value=55, classification="Neutral"),
        ]

        self.mock_fear_greed_repo.get_fear_greed_data.return_value = mock_fear_greed_data

        # テスト実行
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 3)

        # 詳細ログなしでテスト
        result_df = self.service._merge_fear_greed_data(test_df, start_date, end_date)

        # 結果の検証
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('fear_greed_value', result_df.columns)
        self.assertIn('fear_greed_classification', result_df.columns)

        # リポジトリが呼び出されたことを確認
        self.mock_fear_greed_repo.get_fear_greed_data.assert_called_once_with(
            start_time=start_date, end_time=end_date
        )

    def test_merge_fear_greed_data_with_details(self):
        """Fear & Greedデータマージの詳細ログ版テスト"""
        # テストデータの準備
        test_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1H'))

        # Fear & Greedデータのモック
        mock_fear_greed_data = [
            Mock(timestamp=datetime(2024, 1, 1), value=60, classification="Greed"),
        ]

        self.mock_fear_greed_repo.get_fear_greed_data.return_value = mock_fear_greed_data

        # テスト実行（詳細ログあり）
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 2)

        with patch('app.core.services.backtest_data_service.logger') as mock_logger:
            result_df = self.service._merge_fear_greed_data_with_details(
                test_df, start_date, end_date, detailed_logging=True
            )

            # ログが出力されたことを確認
            self.assertTrue(mock_logger.info.called)

        # 結果の検証
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('fear_greed_value', result_df.columns)
        self.assertIn('fear_greed_classification', result_df.columns)

    def test_fill_fear_greed_missing_values(self):
        """Fear & Greed欠損値補完テスト"""
        # 欠損値を含むテストデータ
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'fear_greed_value': [50.0, None, 60.0],
            'fear_greed_classification': ["Neutral", None, "Greed"]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

        # テスト実行
        result_df = self.service._fill_fear_greed_missing_values(test_df)

        # 欠損値が補完されたことを確認
        self.assertFalse(result_df['fear_greed_value'].isna().any())
        self.assertFalse(result_df['fear_greed_classification'].isna().any())

        # 前方補完が正しく動作したことを確認
        self.assertEqual(result_df.loc[result_df.index[1], 'fear_greed_value'], 50.0)
        self.assertEqual(result_df.loc[result_df.index[1], 'fear_greed_classification'], "Neutral")

    def test_fear_greed_repo_not_available(self):
        """Fear & Greedリポジトリが利用できない場合のテスト"""
        # リポジトリなしでサービスを作成
        service_no_fg = BacktestDataService(
            ohlcv_repo=self.mock_ohlcv_repo,
            oi_repo=self.mock_oi_repo,
            fr_repo=self.mock_fr_repo,
            fear_greed_repo=None
        )

        test_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1H'))

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 2)

        # テスト実行
        result_df = service_no_fg._merge_fear_greed_data(test_df, start_date, end_date)

        # デフォルト値が設定されたことを確認
        self.assertIn('fear_greed_value', result_df.columns)
        self.assertIn('fear_greed_classification', result_df.columns)
        
        # 欠損値が適切に補完されたことを確認
        self.assertTrue((result_df['fear_greed_value'] == 50.0).all())
        self.assertTrue((result_df['fear_greed_classification'] == "Neutral").all())


if __name__ == '__main__':
    unittest.main()
