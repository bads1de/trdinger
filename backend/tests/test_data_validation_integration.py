"""
DataSanitizer統合機能テスト
DataValidatorに統合されたDataSanitizerの機能をテストします。
"""

import pytest
from datetime import datetime, timezone
import pandas as pd

from app.utils.data_validation import DataValidator


class TestDataValidationIntegration:
    """DataSanitizer統合機能テスト"""

    def test_validate_ohlcv_records_simple_valid(self):
        """OHLCVレコードのシンプル検証 - 有効なデータ"""
        ohlcv_records = [
            {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'timestamp': datetime.now(timezone.utc),
                'open': 50000.0,
                'high': 51000.0,
                'low': 49500.0,
                'close': 50500.0,
                'volume': 100.0
            }
        ]

        result = DataValidator.validate_ohlcv_records_simple(ohlcv_records)
        assert result is True

    def test_validate_ohlcv_records_simple_invalid(self):
        """OHLCVレコードのシンプル検証 - 無効なデータ"""
        # 必須フィールドが不足
        invalid_records = [
            {
                'symbol': 'BTC/USDT',
                'timestamp': datetime.now(timezone.utc),
                'open': 50000.0,
                # 'timeframe', 'high', 'low', 'close', 'volume' が不足
            }
        ]

        result = DataValidator.validate_ohlcv_records_simple(invalid_records)
        assert result is False

    def test_validate_fear_greed_data_valid(self):
        """Fear & Greed データ検証 - 有効なデータ"""
        fear_greed_records = [
            {
                'value': 75,
                'value_classification': 'Greed',
                'data_timestamp': datetime.now(timezone.utc)
            }
        ]

        result = DataValidator.validate_fear_greed_data(fear_greed_records)
        assert result is True

    def test_validate_fear_greed_data_invalid_value(self):
        """Fear & Greed データ検証 - 無効な値"""
        fear_greed_records = [
            {
                'value': 150,  # 0-100の範囲外
                'value_classification': 'Extreme Greed',
                'data_timestamp': datetime.now(timezone.utc)
            }
        ]

        result = DataValidator.validate_fear_greed_data(fear_greed_records)
        assert result is False

    def test_sanitize_ohlcv_data(self):
        """OHLCVデータサニタイズテスト"""
        ohlcv_records = [
            {
                'symbol': 'btc/usdt',  # 小文字
                'timeframe': '1H',     # 大文字
                'timestamp': '2023-01-01T00:00:00Z',  # 文字列
                'open': '50000.0',     # 文字列
                'high': '51000.0',
                'low': '49500.0',
                'close': '50500.0',
                'volume': '100.0'
            }
        ]

        sanitized = DataValidator.sanitize_ohlcv_data(ohlcv_records)

        assert len(sanitized) == 1
        record = sanitized[0]

        # 正規化されたことを確認
        assert record['symbol'] == 'BTC/USDT'  # 大文字に変換
        assert record['timeframe'] == '1h'     # 小文字に変換
        assert isinstance(record['timestamp'], datetime)  # datetimeに変換
        assert isinstance(record['open'], float)  # floatに変換
        assert record['open'] == 50000.0

    def test_validate_ohlcv_dataframe_valid(self):
        """DataFrame OHLCV検証 - 有効なデータ"""
        df = pd.DataFrame({
            'Open': [50000.0, 50500.0],
            'High': [51000.0, 51500.0],
            'Low': [49500.0, 50000.0],
            'Close': [50500.0, 51000.0],
            'Volume': [100.0, 150.0]
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result['is_valid'] is True
        assert result['data_quality_score'] > 80  # 高い品質スコア

    def test_validate_ohlcv_dataframe_missing_columns(self):
        """DataFrame OHLCV検証 - 必須カラム不足"""
        df = pd.DataFrame({
            'Open': [50000.0, 50500.0],
            # High, Low, Close, Volume が不足
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result['is_valid'] is False
        assert '必要なカラムが不足' in str(result['errors'])

    def test_validate_ohlcv_dataframe_negative_volume(self):
        """DataFrame OHLCV検証 - 負の出来高"""
        df = pd.DataFrame({
            'Open': [50000.0, 50500.0],
            'High': [51000.0, 51500.0],
            'Low': [49500.0, 50000.0],
            'Close': [50500.0, 51000.0],
            'Volume': [100.0, -50.0]  # 負の出来高
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result['is_valid'] is False
        assert result['negative_volumes'] > 0
        assert result['data_quality_score'] < 100