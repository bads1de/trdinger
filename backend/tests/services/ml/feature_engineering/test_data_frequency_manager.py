
import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.data_frequency_manager import DataFrequencyManager

class TestDataFrequencyManager:
    """DataFrequencyManagerのテスト"""

    @pytest.fixture
    def manager(self):
        return DataFrequencyManager()

    def test_detect_ohlcv_timeframe(self, manager):
        """タイムフレーム自動検出のテスト"""
        # 1h データ
        ts_1h = pd.date_range("2024-01-01", periods=10, freq="1h")
        df_1h = pd.DataFrame({"dummy": 0}, index=ts_1h)
        assert manager.detect_ohlcv_timeframe(df_1h) == "1h"
        
        # 1m データ
        ts_1m = pd.date_range("2024-01-01", periods=10, freq="1min")
        df_1m = pd.DataFrame({"dummy": 0}, index=ts_1m)
        assert manager.detect_ohlcv_timeframe(df_1m) == "1m"
        
        # 1d データ
        ts_1d = pd.date_range("2024-01-01", periods=10, freq="1D")
        df_1d = pd.DataFrame({"dummy": 0}, index=ts_1d)
        assert manager.detect_ohlcv_timeframe(df_1d) == "1d"

    def test_align_data_frequencies_upsampling(self, manager):
        """低頻度データを高頻度OHLCVに合わせる（アップサンプリング）テスト"""
        # OHLCV: 1h 間隔
        ohlcv_ts = pd.date_range("2024-01-01", periods=24, freq="1h")
        ohlcv_df = pd.DataFrame({"close": np.random.randn(24)}, index=ohlcv_ts)
        
        # Funding Rate: 8h 間隔
        fr_ts = pd.date_range("2024-01-01", periods=3, freq="8h")
        fr_df = pd.DataFrame({"funding_rate": [0.0001, 0.0002, 0.0003]}, index=fr_ts)
        
        # Open Interest: 4h 間隔
        oi_ts = pd.date_range("2024-01-01", periods=6, freq="4h")
        oi_df = pd.DataFrame({"open_interest": [100, 110, 120, 130, 140, 150]}, index=oi_ts)
        
        aligned_fr, aligned_oi = manager.align_data_frequencies(
            ohlcv_data=ohlcv_df,
            funding_rate_data=fr_df,
            open_interest_data=oi_df
        )
        
        # 結果の行数がOHLCVと一致すること
        assert len(aligned_fr) == len(ohlcv_df)
        assert len(aligned_oi) == len(ohlcv_df)
        
        # FRが前方補完(ffill)されていること
        # 最初の8時間は 0.0001 のはず
        assert aligned_fr["funding_rate"].iloc[0] == 0.0001
        assert aligned_fr["funding_rate"].iloc[7] == 0.0001
        assert aligned_fr["funding_rate"].iloc[8] == 0.0002

    def test_align_data_frequencies_downsampling(self, manager):
        """高頻度データを低頻度OHLCVに合わせる（ダウンサンプリング）テスト"""
        # OHLCV: 1d 間隔
        ohlcv_ts = pd.date_range("2024-01-01", periods=2, freq="1D")
        ohlcv_df = pd.DataFrame({"close": [100, 110]}, index=ohlcv_ts)
        
        # Open Interest: 1h 間隔
        oi_ts = pd.date_range("2024-01-01", periods=48, freq="1h")
        oi_df = pd.DataFrame({"open_interest": np.linspace(100, 147, 48)}, index=oi_ts)
        
        aligned_fr, aligned_oi = manager.align_data_frequencies(
            ohlcv_data=ohlcv_df,
            open_interest_data=oi_df
        )
        
        assert len(aligned_oi) == 2
        # ダウンサンプリング時は平均値などで集約されるはず
        # 1日目の平均が最初の行にくる
        expected_mean = oi_df["open_interest"].iloc[:24].mean()
        # 実装では resample().mean() を使っている
        assert np.isclose(aligned_oi["open_interest"].iloc[0], expected_mean)

    def test_normalize_timeframe(self, manager):
        """タイムフレーム文字列の正規化テスト"""
        assert manager._normalize_timeframe("1m") == "1min"
        assert manager._normalize_timeframe("1h") == "1h"
        assert manager._normalize_timeframe("15m") == "15min"

    def test_empty_input(self, manager):
        """空入力のテスト"""
        ohlcv_df = pd.DataFrame()
        aligned_fr, aligned_oi = manager.align_data_frequencies(ohlcv_df)
        assert aligned_fr is None
        assert aligned_oi is None
