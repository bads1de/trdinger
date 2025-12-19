import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.services.ml.feature_engineering.data_frequency_manager import DataFrequencyManager

class TestDataFrequencyManagerUnit:
    @pytest.fixture
    def manager(self):
        return DataFrequencyManager()

    def test_normalize_timeframe(self, manager):
        assert manager._normalize_timeframe("1m") == "1min"
        assert manager._normalize_timeframe("15m") == "15min"
        assert manager._normalize_timeframe("1h") == "1h"
        assert manager._normalize_timeframe("15min") == "15min"

    def test_detect_ohlcv_timeframe(self, manager):
        # 1. データ不足
        assert manager.detect_ohlcv_timeframe(pd.DataFrame()) == "1h"
        assert manager.detect_ohlcv_timeframe(pd.DataFrame({"close": [100]})) == "1h"
        
        # 2. 1分足
        df_1m = pd.DataFrame(index=pd.date_range("2023-01-01", periods=5, freq="1min"))
        assert manager.detect_ohlcv_timeframe(df_1m) == "1m"
        
        # 3. 1時間足
        df_1h = pd.DataFrame(index=pd.date_range("2023-01-01", periods=5, freq="1h"))
        assert manager.detect_ohlcv_timeframe(df_1h) == "1h"
        
        # 4. 日足
        df_1d = pd.DataFrame(index=pd.date_range("2023-01-01", periods=5, freq="1D"))
        assert manager.detect_ohlcv_timeframe(df_1d) == "1d"
        
        # 5. timestampカラムがある場合
        df_ts = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="15min")
        })
        assert manager.detect_ohlcv_timeframe(df_ts) == "15m"

    def test_align_data_frequencies_comprehensive(self, manager):
        # OHLCV (1h)
        ohlcv = pd.DataFrame({
            "close": np.arange(10, 20)
        }, index=pd.date_range("2023-01-01", periods=10, freq="1h"))
        
        # FR (8h) - アップサンプリングされるはず
        fr = pd.DataFrame({
            "funding_rate": [0.01, 0.02]
        }, index=pd.date_range("2023-01-01", periods=2, freq="8h"))
        
        # OI (1h) - そのまま reindex
        oi = pd.DataFrame({
            "open_interest": np.arange(100, 110)
        }, index=pd.date_range("2023-01-01", periods=10, freq="1h"))
        
        aligned_fr, aligned_oi = manager.align_data_frequencies(ohlcv, fr, oi)
        
        assert len(aligned_fr) == 10
        assert len(aligned_oi) == 10
        assert "funding_rate" in aligned_fr.columns
        assert not aligned_fr["funding_rate"].isnull().any()

    def test_align_data_frequencies_downsampling(self, manager):
        # OHLCV (4h) - ダウンサンプリングパス
        ohlcv = pd.DataFrame({
            "close": [10, 11]
        }, index=pd.date_range("2023-01-01", periods=2, freq="4h"))
        
        # 1時間ごとのFRデータを4時間に集約
        fr = pd.DataFrame({
            "funding_rate": [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02]
        }, index=pd.date_range("2023-01-01", periods=8, freq="1h"))
        
        aligned_fr, _ = manager.align_data_frequencies(ohlcv, funding_rate_data=fr, ohlcv_timeframe="4h")
        assert len(aligned_fr) == 2
        # 平均値が計算されていること (0.01 と 0.02)
        assert aligned_fr.iloc[0]["funding_rate"] == 0.01

    def test_resample_internal_methods(self, manager):
        # _resample_funding_rate の各分岐
        fr = pd.DataFrame({"funding_rate": [0.1]}, index=[datetime(2023,1,1)])
        
        # 1h (ffill)
        res_1h = manager._resample_funding_rate(fr, "1h")
        assert len(res_1h) == 1
        
        # 1d (mean)
        res_1d = manager._resample_funding_rate(fr, "1d")
        assert len(res_1d) == 1

        # _resample_open_interest の各分岐
        oi = pd.DataFrame({"open_interest": [100]}, index=[datetime(2023,1,1)])
        assert len(manager._resample_open_interest(oi, "1h")) == 1
        assert len(manager._resample_open_interest(oi, "4h")) == 1

    def test_validate_data_alignment(self, manager):
        ohlcv = pd.DataFrame({"close": [100]}, index=[datetime(2023,1,1)])
        # 正常
        res = manager.validate_data_alignment(ohlcv)
        assert res["is_valid"] is True
        
        # 異常: 空
        res_empty = manager.validate_data_alignment(pd.DataFrame())
        assert res_empty["is_valid"] is False
        assert "OHLCVデータが空です" in res_empty["errors"]

    def test_detect_ohlcv_timeframe_thresholds(self, manager):
        # 各時間軸の境界値をテスト
        # 1m (<= 1.5)
        df_1m = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="1.5min"))
        assert manager.detect_ohlcv_timeframe(df_1m) == "1m"
        # 5m (<= 7.5)
        df_5m = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="7.5min"))
        assert manager.detect_ohlcv_timeframe(df_5m) == "5m"
        # 15m (<= 22.5)
        df_15m = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="22.5min"))
        assert manager.detect_ohlcv_timeframe(df_15m) == "15m"
        # 30m (<= 45)
        df_30m = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="45min"))
        assert manager.detect_ohlcv_timeframe(df_30m) == "30m"
        # 1h (<= 120)
        df_1h = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="120min"))
        assert manager.detect_ohlcv_timeframe(df_1h) == "1h"
        # 4h (<= 360)
        df_4h = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="360min"))
        assert manager.detect_ohlcv_timeframe(df_4h) == "4h"
        # 1d (> 360)
        df_1d = pd.DataFrame(index=pd.date_range("2023-01-01", periods=2, freq="1440min"))
        assert manager.detect_ohlcv_timeframe(df_1d) == "1d"

    def test_resample_full_branches(self, manager):
        # _resample_funding_rate の全 elif パス
        fr = pd.DataFrame({"funding_rate": [0.1]}, index=[datetime(2023,1,1)])
        # 30m
        assert len(manager._resample_funding_rate(fr, "30m")) == 1
        # 4h
        assert len(manager._resample_funding_rate(fr, "4h")) == 1
        # unknown (default ffill)
        assert len(manager._resample_funding_rate(fr, "99h")) == 1
        
        # _resample_open_interest の全 elif パス
        oi = pd.DataFrame({"open_interest": [100]}, index=[datetime(2023,1,1)])
        # 5m
        assert len(manager._resample_open_interest(oi, "5m")) == 1
        # 1d
        assert len(manager._resample_open_interest(oi, "1d")) == 1
        # unknown
        assert len(manager._resample_open_interest(oi, "99h")) == 1

    def test_align_data_frequencies_index_variants(self, manager):
        # 最初からインデックスが DatetimeIndex の場合 (正常系の別パターン)
        idx = pd.date_range("2023-01-01", periods=2, freq="1h")
        ohlcv = pd.DataFrame({"close": [100, 101]}, index=idx)
        fr = pd.DataFrame({"funding_rate": [0.01, 0.02]}, index=idx)
        
        aligned_fr, _ = manager.align_data_frequencies(ohlcv, funding_rate_data=fr)
        assert len(aligned_fr) == 2
        assert "funding_rate" in aligned_fr.columns



