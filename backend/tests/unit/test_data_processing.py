"""
統合データ処理テスト

データ変換、プロセッサ、バリデーションの機能を統合テスト
TDD原則に基づき、各機能を包括的にテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock

# データ変換関連
from app.utils.data_conversion import (
    OHLCVDataConverter,
    FundingRateDataConverter,
    OpenInterestDataConverter,
    ensure_list,
    DataConversionError
)

# データ処理関連
from backend.app.utils.data_processing import data_processor

# データバリデーション関連
from app.utils.data_validation import (
    OHLCVDataModel,
    DataValidator,
    validate_dataframe_with_config,
    clean_dataframe_with_config,
    OHLCV_VALIDATION_CONFIG,
    EXTENDED_MARKET_DATA_VALIDATION_CONFIG
)


class TestDataConversionIntegrated:
    """データ変換機能の統合テスト"""

    def test_ohlcv_conversion_complete_flow(self):
        """OHLCV変換の完全フロー"""
        ohlcv_data = [
            [1638360000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.5],
            [1638360060000, 50500.0, 51500.0, 50000.0, 51000.0, 95.0]
        ]

        # CCXT形式からDB形式へ
        db_records = OHLCVDataConverter.ccxt_to_db_format(ohlcv_data, "BTC/USDT", "1h")
        assert len(db_records) == 2

        # DB形式からAPI形式へ
        db_mocks = []
        for record in db_records:
            mock = Mock()
            mock.timestamp = record["timestamp"]
            mock.open = record["open"]
            mock.high = record["high"]
            mock.low = record["low"]
            mock.close = record["close"]
            mock.volume = record["volume"]
            db_mocks.append(mock)

        api_data = OHLCVDataConverter.db_to_api_format(db_mocks)
        assert len(api_data) == 2
        assert api_data[0] == [1638360000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.5]

    def test_funding_rate_conversion_complete_flow(self):
        """ファンディングレート変換の完全フロー"""
        funding_data = [{
            "datetime": "2023-10-01T12:00:00.000Z",
            "fundingRate": 0.0001,
            "nextFundingDatetime": "2023-10-01T13:00:00.000Z"
        }]

        result = FundingRateDataConverter.ccxt_to_db_format(funding_data, "BTC/USDT")
        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["funding_rate"] == 0.0001
        assert isinstance(result[0]["data_timestamp"], datetime)

    def test_open_interest_conversion_complete_flow(self):
        """オープンインタレスト変換の完全フロー"""
        oi_data = [{
            "timestamp": 1638360000000,
            "openInterestAmount": 1000000.0
        }]

        result = OpenInterestDataConverter.ccxt_to_db_format(oi_data, "BTC/USDT")
        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/USDT"
        assert result[0]["open_interest_value"] == 1000000.0

    def test_ensure_list_comprehensive(self):
        """ensure_list関数の包括テスト"""
        test_cases = [
            ([1, 2, 3], [1, 2, 3]),
            ("hello", ["h", "e", "l", "l", "o"]),
            (42, [42]),
            ({}, []),
            (None, []),
            ((1, 2), [1, 2])
        ]

        for input_val, expected in test_cases:
            assert ensure_list(input_val) == expected


class TestDataProcessingIntegrated:
    """データ処理機能の統合テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        np.random.seed(42)

        # OHLCの関係を満たすようにデータを生成
        base_prices = np.random.uniform(100, 110, 100)
        volatility = np.random.uniform(0.01, 0.05, 100)

        opens = base_prices
        highs = base_prices * (1 + volatility)
        lows = base_prices * (1 - volatility)
        closes = base_prices + np.random.uniform(-volatility, volatility, 100) * base_prices

        # 確実にlow <= open/close <= highを満たす
        lows = np.minimum(lows, np.minimum(opens, closes))
        highs = np.maximum(highs, np.maximum(opens, closes))

        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, 100),
        }
        return pd.DataFrame(data)

    def test_data_processor_complete_workflow(self, sample_data):
        """DataProcessorの完全ワークフロー"""
        # 必須カラム
        required_columns = ['open', 'high', 'low', 'close']

        # 1. クリーニングと検証
        cleaned_data = data_processor.clean_and_validate_data(
            sample_data, required_columns
        )
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0

        # 2. 前処理パイプライン
        processed_data = data_processor.preprocess_with_pipeline(cleaned_data)
        assert isinstance(processed_data, pd.DataFrame)

        # 3. 効率的なデータ処理
        efficient_data = data_processor.process_data_efficiently(cleaned_data)
        assert isinstance(efficient_data, pd.DataFrame)

    def test_data_processor_pipeline_management(self, sample_data):
        """パイプライン管理機能"""
        # パイプライン作成
        pipeline = data_processor.create_optimized_pipeline()
        assert pipeline is not None

        # パイプライン情報取得
        info = data_processor.get_pipeline_info("nonexistent")
        assert info["exists"] is False

        # キャッシュクリア
        data_processor.clear_cache()
        assert len(data_processor.fitted_pipelines) == 0


class TestDataValidationIntegrated:
    """データバリデーション機能の統合テスト"""

    def test_ohlcv_validation_complete_flow(self):
        """OHLCVバリデーションの完全フロー"""
        # 有効なデータ
        valid_data = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        # Pydanticモデル検証
        for idx, row in valid_data.iterrows():
            model = OHLCVDataModel(**row.to_dict())
            assert model.Open == row["Open"]

        # DataFrameバリデーション
        is_valid, errors = validate_dataframe_with_config(valid_data, OHLCV_VALIDATION_CONFIG)
        assert is_valid is True
        assert len(errors) == 0

        # データクリーニング
        cleaned = clean_dataframe_with_config(valid_data, OHLCV_VALIDATION_CONFIG)
        assert len(cleaned) == 2

    def test_data_validator_comprehensive_validation(self):
        """DataValidatorの包括的検証"""
        # 正常なデータ
        df = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        # OHLCVデータ検証
        result = DataValidator.validate_ohlcv_data(df)
        assert result["is_valid"] is True
        assert result["data_quality_score"] == 100.0

        # OHLCVレコード検証
        ohlcv_records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": datetime.now(timezone.utc),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]
        assert DataValidator.validate_ohlcv_records_simple(ohlcv_records) is True

    def test_data_validator_sanitization(self):
        """DataValidatorのサニタイズ機能"""
        # サニタイズ対象データ
        records = [{
            "symbol": "btc/usdt",  # 小文字
            "timeframe": " 1h ",   # 前後スペース
            "timestamp": "2023-10-01T12:00:00Z",
            "open": "50000",       # 文字列
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "100"
        }]

        sanitized = DataValidator.sanitize_ohlcv_data(records)

        assert sanitized[0]["symbol"] == "BTC/USDT"  # 正規化
        assert sanitized[0]["timeframe"] == "1h"     # スペース除去
        assert isinstance(sanitized[0]["timestamp"], datetime)
        assert sanitized[0]["open"] == 50000.0      # 数値変換


class TestDataProcessingErrorHandling:
    """エラーハンドリングの統合テスト"""

    def test_data_conversion_error_handling(self):
        """データ変換のエラーハンドリング"""
        # 無効なOHLCVデータ
        invalid_ohlcv = [["invalid", 50000.0]]

        with pytest.raises(ValueError):
            OHLCVDataConverter.ccxt_to_db_format(invalid_ohlcv, "BTC/USDT", "1h")

        # 無効なタイムスタンプ
        with pytest.raises(ValueError):
            OHLCVDataConverter.db_to_api_format([])

    def test_data_processor_error_handling(self):
        """DataProcessorのエラーハンドリング"""
        # 空のDataFrame
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            data_processor.clean_and_validate_data(empty_df, ['open', 'high'])

    def test_data_validator_error_handling(self):
        """DataValidatorのエラーハンドリング"""
        # 無効なPydanticモデル
        with pytest.raises(ValueError):
            OHLCVDataModel(Open=-100.0, High=51000.0, Low=49000.0, Close=50500.0, Volume=100.0)

        # 空のDataFrame
        result = DataValidator.validate_ohlcv_data(pd.DataFrame())
        assert result["is_valid"] is False


if __name__ == "__main__":
    pytest.main([__file__])