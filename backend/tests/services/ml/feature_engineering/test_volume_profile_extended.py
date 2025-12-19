import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.services.ml.feature_engineering.volume_profile_features import (
    VolumeProfileFeatureCalculator,
    _numba_calc_bins,
    _numba_rolling_volume_profile,
    _numba_detect_volume_nodes_signed,
    _numba_vp_skewness_kurtosis
)


class TestVolumeProfileExtended:
    @pytest.fixture
    def ohlcv_basic(self):
        n = 100
        return pd.DataFrame({
            "high": np.linspace(100, 110, n),
            "low": np.linspace(90, 100, n),
            "close": np.linspace(95, 105, n),
            "volume": np.ones(n) * 100
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))

    def test_numba_calc_bins_edge_cases(self):
        # ビン計算の境界値テスト
        w_high = np.array([105.0, 110.0])
        w_low = np.array([95.0, 100.0])
        w_vol = np.array([100.0, 200.0])
        price_min = 90.0
        bin_step = 2.0
        num_bins = 15
        
        bins = _numba_calc_bins(w_high, w_low, w_vol, price_min, bin_step, num_bins)
        assert len(bins) == num_bins
        assert bins.sum() > 0

    def test_rolling_vp_zero_variance(self):
        # 価格が全く動かない場合 (p_min == p_max)
        n = 20
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)
        vol = np.full(n, 100.0)
        
        poc, vah, val = _numba_rolling_volume_profile(high, low, close, vol, 5, 10)
        # i >= window (5) の部分は close_arr[i] (100.0) で埋められるはず
        assert poc[10] == 100.0
        assert vah[10] == 100.0
        assert val[10] == 100.0

    def test_detect_nodes_edge_cases(self):
        n = 20
        high = np.linspace(100, 110, n)
        low = np.linspace(90, 100, n)
        close = np.linspace(95, 105, n)
        vol = np.ones(n) * 100
        
        hvn, lvn = _numba_detect_volume_nodes_signed(high, low, close, vol, 5, 10)
        assert len(hvn) == n
        assert len(lvn) == n

    def test_vp_skewness_kurtosis_zero_vol(self):
        # 出来高0の期間がある場合
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        vol = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        skew, kurt = _numba_vp_skewness_kurtosis(close, vol, 2)
        assert (skew == 0).all()
        assert (kurt == 0).all()

    def test_calculator_with_nan_handling(self, ohlcv_basic):
        # シリーズの最初に NaN がある状況（safe_fill の導通）
        calc = VolumeProfileFeatureCalculator(lookback_period=10)
        res = calc.calculate_features(ohlcv_basic)
        # 最初の lookback_period 分は safe_fill によって 0.0 等で埋められる
        assert "POC_Distance_10" in res.columns
        assert not res.isnull().any().any()


    def test_poc_distance_zero_poc(self, ohlcv_basic):
        # POCが0になる（理論上ありえないが、コード上は存在する）場合の回避
        # calculator内部のリスト内包表記の if poc_f[i]!=0 else 0.0 を通す
        with patch("app.services.ml.feature_engineering.volume_profile_features._numba_rolling_volume_profile", 
                   return_value=(np.zeros(100), np.zeros(100), np.zeros(100))):
            calc = VolumeProfileFeatureCalculator(lookback_period=10)
            res = calc.calculate_features(ohlcv_basic, lookback_periods=[10])
            assert (res["POC_Distance_10"] == 0.0).all()
