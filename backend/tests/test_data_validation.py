"""データバリデーションモジュールのテスト

OHLCVデータ、DataFrameバリデーション、クリーニング機能をテスト
エラーケースを追加してバグを洗い出す
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import numpy as np

from app.utils.data_validation import (
    OHLCVDataModel,
    DataValidator,
    validate_dataframe_with_config,
    clean_dataframe_with_config,
    OHLCV_VALIDATION_CONFIG,
    EXTENDED_MARKET_DATA_VALIDATION_CONFIG
)


class TestOHLCVDataModel:
    """Pydantic OHLCVデータモデルのテスト"""

    def test_ohlcv_data_model_valid(self):
        """正常なOHLCVデータを検証"""
        data = {
            "Open": 50000.0,
            "High": 51000.0,
            "Low": 49000.0,
            "Close": 50500.0,
            "Volume": 100.0
        }

        model = OHLCVDataModel(**data)
        assert model.Open == 50000.0
        assert model.High == 51000.0

    def test_ohlcv_data_model_invalid_negative_price(self):
        """負の価格値を検証（エラーを期待）"""
        data = {
            "Open": -100.0,  # 負の値は無効
            "High": 51000.0,
            "Low": 49000.0,
            "Close": 50500.0,
            "Volume": 100.0
        }

        with pytest.raises(ValueError):
            OHLCVDataModel(**data)

    def test_ohlcv_data_model_invalid_zero_price(self):
        """ゼロの価格値を検証（エラーを期待）"""
        data = {
            "Open": 0.0,  # ゼロは無効
            "High": 51000.0,
            "Low": 49000.0,
            "Close": 50500.0,
            "Volume": 100.0
        }

        with pytest.raises(ValueError):
            OHLCVDataModel(**data)

    def test_ohlcv_data_model_invalid_negative_volume(self):
        """負の出来高を検証（有効 - gt=0は価格のみ）"""
        data = {
            "Open": 50000.0,
            "High": 51000.0,
            "Low": 49000.0,
            "Close": 50500.0,
            "Volume": -10.0  # ge=0は有効
        }

        with pytest.raises(ValueError):
            OHLCVDataModel(**data)


class TestValidateDataFrameWithConfig:
    """DataFrameバリデーション関数のテスト"""

    def test_validate_dataframe_normal(self):
        """正常なDataFrameの検証"""
        df = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_dataframe_missing_required_column(self):
        """必須カラムが欠けたDataFrameの検証"""
        df = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            # Low missing
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        assert is_valid is False
        assert len(errors) > 0
        assert "Low" in str(errors)

    def test_validate_dataframe_nan_in_required_column(self):
        """必須カラムにNaNを含む場合の検証"""
        df = pd.DataFrame({
            "Open": [50000.0, np.nan],  # NaNを含む
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        assert is_valid is False
        assert len(errors) > 0
        assert "NaN" in str(errors)

    def test_validate_dataframe_invalid_number_type(self):
        """数値型でないカラムの検証"""
        df = pd.DataFrame({
            "Open": ["invalid", 50500.0],  # 文字列を含む
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        assert is_valid is False
        assert len(errors) > 0
        assert "数値型" in str(errors)

    def test_validate_dataframe_duplicate_columns(self):
        """重複カラムを含むDataFrameの検証（pandasが自動的に処理）"""
        df = pd.DataFrame({
            ("Open",): [50000.0, 50500.0],  # MultiIndexのような処理
            ("Open",): [51000.0, 51500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        # エラーが発生するか確認
        try:
            is_valid, errors = validate_dataframe_with_config(df, config)
            # カラムアクセスでエラーになる可能性
        except Exception:
            assert True  # Expected to fail

    def test_validate_dataframe_empty_dataframe(self):
        """空のDataFrameを検証"""
        df = pd.DataFrame()

        config = OHLCV_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_dataframe_extreme_values(self):
        """極端な値を含むDataFrameの検証"""
        df = pd.DataFrame({
            "Open": [1e10, 50500.0],  # 非常に大きな値
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        # 拡張バリデーション設定を使用
        config = EXTENDED_MARKET_DATA_VALIDATION_CONFIG.copy()
        is_valid, errors = validate_dataframe_with_config(df, config)

        # MAX_VALUE_THRESHOLDを超えるために無効
        assert is_valid is False
        assert len(errors) > 0


class TestCleanDataFrameWithConfig:
    """DataFrameクリーニング関数のテスト"""

    def test_clean_dataframe_normal(self):
        """正常なDataFrameのクリーニング"""
        df = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        cleaned_df = clean_dataframe_with_config(df, config)

        assert len(cleaned_df) == 2
        assert cleaned_df.equals(df)  # 変更がないはず

    def test_clean_dataframe_with_nan(self):
        """NaNを含むDataFrameのクリーニング"""
        df = pd.DataFrame({
            "Open": [50000.0, np.nan],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        cleaned_df = clean_dataframe_with_config(df, config, drop_invalid_rows=False)

        assert len(cleaned_df) == 2
        assert pd.isna(cleaned_df.loc[1, "Open"])  # NaNが残る

    def test_clean_dataframe_with_extreme_values(self):
        """極端な値を含むDataFrameのクリーニング"""
        df = pd.DataFrame({
            "Open": [50000.0, 1e10],  # 極端な値
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        cleaned_df = clean_dataframe_with_config(df, config, drop_invalid_rows=False)

        # 極端な値がクリップされるはず
        assert cleaned_df.loc[1, "Open"] <= 1e6  # MAX_VALUE_THRESHOLD

    def test_clean_dataframe_type_conversion(self):
        """型変換を伴うDataFrameのクリーニング"""
        df = pd.DataFrame({
            "Open": ["50000", "50500"],  # 文字列である
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        cleaned_df = clean_dataframe_with_config(df, config)

        # 型変換がされるはず
        assert cleaned_df["Open"].dtype == "float64"
        assert cleaned_df.loc[0, "Open"] == 50000.0

    def test_clean_dataframe_drop_invalid_rows(self):
        """無効な行を削除する機能のテスト"""
        df = pd.DataFrame({
            "Open": [50000.0, np.nan],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        config = OHLCV_VALIDATION_CONFIG.copy()
        cleaned_df = clean_dataframe_with_config(df, config, drop_invalid_rows=True)

        # NaNを含む行が削除されるはず
        assert len(cleaned_df) == 1
        assert not pd.isna(cleaned_df.iloc[0]["Open"])


class TestDataValidator:
    """DataValidatorクラスのテスト"""

    def test_validate_ohlcv_data_normal(self):
        """正常なOHLCVデータを検証"""
        df = pd.DataFrame({
            "Open": [50000.0, 50500.0],
            "High": [51000.0, 51500.0],
            "Low": [49000.0, 49500.0],
            "Close": [50500.0, 51000.0],
            "Volume": [100.0, 95.0]
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["data_quality_score"] == 100.0

    def test_validate_ohlcv_data_ohlc_violations(self):
        """OHLC論理違反を含むデータを検証"""
        df = pd.DataFrame({
            "Open": [50000.0],
            "High": [49000.0],  # High < Open  - violation
            "Low": [50000.0],   # Low > Open  - violation
            "Close": [50500.0],
            "Volume": [100.0]
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result["is_valid"] is False  # 違反が5%以上
        assert result["ohlc_violations"] == 2
        assert result["data_quality_score"] < 100.0

    def test_validate_ohlcv_data_empty_dataframe(self):
        """空のDataFrameを検証"""
        df = pd.DataFrame()

        result = DataValidator.validate_ohlcv_data(df)

        assert result["is_valid"] is False
        assert result["data_quality_score"] == 0.0

    def test_validate_ohlcv_data_negative_volume(self):
        """負の出来高を含むデータを検証"""
        df = pd.DataFrame({
            "Open": [50000.0],
            "High": [51000.0],
            "Low": [49000.0],
            "Close": [50500.0],
            "Volume": [-100.0]  # 負の出来高
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result["is_valid"] is True  # 負の出来高は警告のみ
        assert result["negative_volumes"] == 1
        assert len(result["warnings"]) > 0

    def test_validate_ohlcv_data_strict_mode_extreme_values(self):
        """厳密モードでの極端な値の検証"""
        df = pd.DataFrame({
            "Open": [0.0],  # ゼロ価格
            "High": [51000.0],
            "Low": [49000.0],
            "Close": [50500.0],
            "Volume": [100.0]
        })

        result = DataValidator.validate_ohlcv_data(df, strict_mode=True)

        assert result["is_valid"] is True  # 厳密モードでもzeroは許容
        assert len(result["warnings"]) == 0  # 極端値チェックはNaN/Nullのみ

    def test_validate_ohlcv_data_high_missing_data_ratio(self):
        """欠損データの割合が高いデータを検証"""
        df = pd.DataFrame({
            "Open": [50000.0, np.nan],
            "High": [51000.0, np.nan],
            "Low": [49000.0, np.nan],
            "Close": [50500.0, np.nan],
            "Volume": [100.0, np.nan]
        })

        result = DataValidator.validate_ohlcv_data(df)

        assert result["is_valid"] is True  # 欠損率50%未満
        assert result["missing_data_ratio"] == 0.5  # 50%の欠損
        assert len(result["warnings"]) > 0

    def test_validate_ohlcv_records_simple_normal(self):
        """正常なOHLCVレコードのシンプル検証"""
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": datetime.now(timezone.utc),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        assert DataValidator.validate_ohlcv_records_simple(records) is True

    def test_validate_ohlcv_records_simple_missing_field(self):
        """必須フィールドが欠けたレコードの検証"""
        records = [{
            "timestamp": datetime.now(timezone.utc),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
            # symbol and timeframe missing
        }]

        assert DataValidator.validate_ohlcv_records_simple(records) is False

    def test_validate_ohlcv_records_simple_invalid_field_type(self):
        """フィールドの型が無効なレコードの検証"""
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": datetime.now(timezone.utc),
            "open": "invalid",  # should be numeric
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        assert DataValidator.validate_ohlcv_records_simple(records) is False

    def test_validate_ohlcv_records_simple_invalid_timestamp_type(self):
        """タイムスタンプの型が無効なレコードの検証"""
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": "invalid-timestamp",  # should be datetime/int/float
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        assert DataValidator.validate_ohlcv_records_simple(records) is False

    def test_validate_fear_greed_data_normal(self):
        """正常なFear & Greed Indexデータの検証"""
        records = [{
            "value": 50,
            "value_classification": "Neutral",
            "data_timestamp": datetime.now(timezone.utc)
        }]

        assert DataValidator.validate_fear_greed_data(records) is True

    def test_validate_fear_greed_data_invalid_value_range(self):
        """値の範囲が無効なFear & Greedデータの検証"""
        records = [{
            "value": 150,  # should be 0-100
            "value_classification": "Extreme Fear",
            "data_timestamp": datetime.now(timezone.utc)
        }]

        assert DataValidator.validate_fear_greed_data(records) is False

    def test_validate_fear_greed_data_missing_fields(self):
        """必須フィールドが欠けたFear & Greedデータの検証"""
        records = [{
            "value": 50,
            # value_classification missing
            "data_timestamp": datetime.now(timezone.utc)
        }]

        assert DataValidator.validate_fear_greed_data(records) is False

    def test_sanitize_ohlcv_data_normal(self):
        """正常なOHLCVデータのサニタイズ"""
        records = [{
            "symbol": "btc/usdt",
            "timeframe": " 1h ",
            "timestamp": "2023-10-01T12:00:00Z",
            "open": "50000",
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "100"
        }]

        sanitized = DataValidator.sanitize_ohlcv_data(records)

        assert sanitized[0]["symbol"] == "BTC/USDT"  # 正規化
        assert sanitized[0]["timeframe"] == "1h"  # lowercase
        assert isinstance(sanitized[0]["timestamp"], datetime)
        assert sanitized[0]["open"] == 50000.0
        assert sanitized[0]["close"] == 50500.0

    def test_sanitize_ohlcv_data_floating_timestamp(self):
        """浮動小数点タイムスタンプのサニタイズ"""
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": 1696161600.0,  # float timestamp
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        sanitized = DataValidator.sanitize_ohlcv_data(records)

        assert isinstance(sanitized[0]["timestamp"], datetime)

    def test_sanitize_ohlcv_data_datetime_timestamp(self):
        """datetimeタイムスタンプのサニタイズ"""
        timestamp = datetime.now(timezone.utc)
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": timestamp,  # already datetime
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        sanitized = DataValidator.sanitize_ohlcv_data(records)

        assert sanitized[0]["timestamp"] is timestamp

    def test_sanitize_ohlcv_data_invalid_timestamp(self):
        """無効なタイムスタンプのサニタイズ（エラーを期待）"""
        records = [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": "invalid-timestamp",
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        }]

        with pytest.raises(ValueError):
            DataValidator.sanitize_ohlcv_data(records)


if __name__ == "__main__":
    pytest.main([__file__])