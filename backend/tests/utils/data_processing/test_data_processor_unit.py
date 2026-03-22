import pytest
import numpy as np
import pandas as pd
from app.utils.data_processing.data_processor import DataProcessor


class TestDataProcessorUnit:
    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def sample_ohlcv(self):
        rows = 100
        df = pd.DataFrame(
            {
                "Open": np.random.normal(100, 1, rows),
                "High": np.random.normal(101, 1, rows),
                "Low": np.random.normal(99, 1, rows),
                "Close": np.random.normal(100, 1, rows),
                "Volume": np.random.normal(1000, 10, rows),
                "funding_rate": np.random.normal(0, 0.01, rows),
            },
            index=pd.date_range("2023-01-01", periods=rows, freq="h"),
        )
        return df

    def test_clean_and_validate_data_success(self, processor, sample_ohlcv):
        # 正常系
        required = ["open", "high", "low", "close", "volume"]
        result = processor.clean_and_validate_data(sample_ohlcv, required)

        assert isinstance(result, pd.DataFrame)
        # カラム名が小文字になっていること
        assert all(col in result.columns for col in required)
        # データ型が最適化されていること (float64 -> float32)
        assert result["open"].dtype == "float32"

    def test_clean_and_validate_empty_data(self, processor):
        # 空データのハンドリング
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = processor.clean_and_validate_data(df, ["open"])
        assert result.empty

    def test_clean_and_validate_data_error(self, processor, sample_ohlcv):
        # バリデーションエラーの発生 (補間を無効にして不整合データを残す)
        bad_df = sample_ohlcv.copy()
        bad_df.columns = bad_df.columns.str.lower()
        bad_df.loc[bad_df.index[0], "low"] = 500  # Low > High

        with pytest.raises(ValueError, match="データ検証に失敗しました"):
            processor.clean_and_validate_data(
                bad_df, ["open", "high", "low", "close"], interpolate=False
            )

    def test_interpolate_data_logic(self, processor):
        # 補間ロジックのテスト
        df = pd.DataFrame(
            {
                "open": [100.0, np.nan, 102.0],
                "high": [105.0, 105.0, 105.0],
                "low": [95.0, 95.0, 95.0],
                "close": [101.0, 101.0, 101.0],
                "cat": ["A", np.nan, "A"],
            }
        )

        result = processor._interpolate_data(df)
        # 数値の補間 (ffillが優先される実装)
        assert not result["open"].isnull().any()
        assert result.loc[1, "open"] == 100.0  # ffill の結果
        # カテゴリカルの補間 (最頻値)
        assert result.loc[1, "cat"] == "A"

    def test_ohlc_relationship_fix(self, processor):
        # OHLCの不整合修正テスト
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [105.0],  # 異常: Low > High
                "close": [102.0],
            }
        )

        result = processor._interpolate_data(df)
        assert result.loc[0, "low"] <= result.loc[0, "open"]
        assert result.loc[0, "low"] <= result.loc[0, "close"]
        assert result.loc[0, "high"] >= result.loc[0, "open"]
        assert result.loc[0, "high"] >= result.loc[0, "close"]

    def test_clip_extended_data_ranges(self, processor):
        # 範囲クリップのテスト
        df = pd.DataFrame(
            {
                "funding_rate": [1.5, -2.0, np.inf, 0.5],
                "open_interest": [100, -10, np.nan, 50],
            }
        )

        result = processor._clip_extended_data_ranges(df)
        # funding_rate: [-1, 1]
        assert (result["funding_rate"] <= 1.0).all()
        assert (result["funding_rate"] >= -1.0).all()
        # open_interest: >= 0
        assert (result["open_interest"] >= 0).all()

    def test_clear_cache(self, processor):
        # メソッドが存在し、エラーにならないこと
        processor.clear_cache()
