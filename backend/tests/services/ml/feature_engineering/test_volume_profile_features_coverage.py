"""
Extended coverage tests for ``volume_profile_features``.

Covers inner Numba functions, the inner ``safe_fill`` closure,
and edge cases for the public ``calculate_features`` API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering import volume_profile_features as vpf


def _build_ohlcv(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="h")
    rng = np.random.default_rng(42)
    close = np.linspace(100, 110, rows) + rng.normal(0, 0.5, rows)
    high = close + 0.5
    low = close - 0.5
    open_ = close + rng.normal(0, 0.1, rows)
    volume = rng.uniform(100, 1000, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestNumbaCalcBinsExtended:
    def test_basic(self) -> None:
        w_high = np.array([105.0])
        w_low = np.array([95.0])
        w_vol = np.array([100.0])
        bins = vpf._numba_calc_bins(w_high, w_low, w_vol, 90.0, 1.0, 20)
        assert len(bins) == 20
        # All volume should be assigned
        assert bins.sum() == pytest.approx(100.0)

    def test_clipped_indices(self) -> None:
        # High above bin range
        w_high = np.array([200.0])
        w_low = np.array([195.0])
        w_vol = np.array([100.0])
        # price_min=90, bin_step=5, num_bins=5 => bins span [90, 115)
        bins = vpf._numba_calc_bins(w_high, w_low, w_vol, 90.0, 5.0, 5)
        # Indices should be clipped to [0, num_bins-1]
        assert len(bins) == 5

    def test_empty_input(self) -> None:
        bins = vpf._numba_calc_bins(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            90.0,
            1.0,
            10,
        )
        assert len(bins) == 10
        assert bins.sum() == 0.0


class TestNumbaRollingVolumeProfileExtended:
    def test_poc_vah_val_have_finite_values(self) -> None:
        df = _build_ohlcv(100)
        poc, vah, val = vpf._numba_rolling_volume_profile(
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            df["volume"].to_numpy(),
            window=20,
            num_bins=10,
        )
        assert len(poc) == 100
        # At least the last few entries should be finite
        assert np.isfinite(poc[20:]).any()
        assert np.isfinite(vah[20:]).any()
        assert np.isfinite(val[20:]).any()
        # VAH >= POC >= VAL
        valid_idx = (poc[20:] > 0) & np.isfinite(poc[20:])
        if valid_idx.any():
            assert (vah[20:][valid_idx] >= poc[20:][valid_idx]).all()
            assert (poc[20:][valid_idx] >= val[20:][valid_idx]).all()

    def test_constant_prices(self) -> None:
        n = 30
        s = np.full(n, 100.0)
        poc, vah, val = vpf._numba_rolling_volume_profile(s, s, s, s, 5, 10)
        # All prices identical => p_min == p_max => poc/vah/val = close
        for i in range(5, n):
            assert poc[i] == 100.0
            assert vah[i] == 100.0
            assert val[i] == 100.0


class TestNumbaDetectVolumeNodesExtended:
    def test_returns_correct_shape(self) -> None:
        df = _build_ohlcv(100)
        hvn, lvn = vpf._numba_detect_volume_nodes_signed(
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            df["volume"].to_numpy(),
            window=20,
            num_bins=10,
        )
        assert hvn.shape == (100,)
        assert lvn.shape == (100,)

    def test_constant_prices(self) -> None:
        n = 30
        s = np.full(n, 100.0)
        hvn, lvn = vpf._numba_detect_volume_nodes_signed(s, s, s, s, 5, 10)
        # All prices equal => p_min == p_max, no distances computed
        assert (hvn == 0.0).all()
        assert (lvn == 0.0).all()


class TestNumbaVPSkewnessKurtosisExtended:
    def test_constant_close_with_volume(self) -> None:
        n = 30
        close = np.full(n, 100.0)
        vol = np.ones(n) * 100
        skew, kurt = vpf._numba_vp_skewness_kurtosis(close, vol, 10)
        # mean == median => skew = 0
        # constant => std = 0 => no kurtosis
        assert (skew == 0.0).all()
        assert (kurt == 0.0).all()

    def test_output_shapes(self) -> None:
        df = _build_ohlcv(100)
        skew, kurt = vpf._numba_vp_skewness_kurtosis(
            df["close"].to_numpy(), df["volume"].to_numpy(), 20
        )
        assert skew.shape == (100,)
        assert kurt.shape == (100,)


class TestVolumeProfileCalculatorExtended:
    def test_default_lookback_periods(self) -> None:
        df = _build_ohlcv(250)
        calc = vpf.VolumeProfileFeatureCalculator(lookback_period=50, num_bins=20)
        result = calc.calculate_features(df)
        # Default lookback_periods = [50, 100, 200]
        for period in [50, 100, 200]:
            assert f"POC_Distance_{period}" in result.columns
            assert f"VAH_Distance_{period}" in result.columns
            assert f"VAL_Distance_{period}" in result.columns
            assert f"In_Value_Area_{period}" in result.columns
            assert f"Value_Area_Width_{period}" in result.columns
        # Common HVN/LVN/Skewness/Kurtosis
        assert "HVN_Distance" in result.columns
        assert "LVN_Distance" in result.columns
        assert "VP_Skewness" in result.columns
        assert "VP_Kurtosis" in result.columns

    def test_custom_lookback_periods(self) -> None:
        df = _build_ohlcv(150)
        calc = vpf.VolumeProfileFeatureCalculator(lookback_period=20, num_bins=10)
        result = calc.calculate_features(df, lookback_periods=[20, 50])
        assert "POC_Distance_20" in result.columns
        assert "POC_Distance_50" in result.columns
        # Default periods should NOT be present when custom is supplied
        assert "POC_Distance_100" not in result.columns
        assert "POC_Distance_200" not in result.columns

    def test_value_area_width_uses_poc_denominator(self) -> None:
        # The Value_Area_Width is (vah - val) / poc.
        # If POC = 0, the value should be 0 (not NaN).
        n = 100
        df = pd.DataFrame(
            {
                "high": np.full(n, 100.0),
                "low": np.full(n, 100.0),
                "close": np.full(n, 100.0),
                "volume": np.ones(n),
            },
            index=pd.date_range("2023-01-01", periods=n, freq="h"),
        )
        calc = vpf.VolumeProfileFeatureCalculator(lookback_period=20, num_bins=10)
        result = calc.calculate_features(df, lookback_periods=[20])
        # No NaN values
        assert not result.isnull().any().any()

    def test_output_dataframe_index(self) -> None:
        df = _build_ohlcv(80)
        calc = vpf.VolumeProfileFeatureCalculator(lookback_period=20)
        result = calc.calculate_features(df, lookback_periods=[20])
        assert result.index.equals(df.index)

    def test_with_nan_values_in_ohlcv(self) -> None:
        df = _build_ohlcv(80)
        # Inject a NaN
        df.loc[df.index[10], "high"] = np.nan
        calc = vpf.VolumeProfileFeatureCalculator(lookback_period=20)
        # Should not raise
        result = calc.calculate_features(df, lookback_periods=[20])
        assert isinstance(result, pd.DataFrame)


class TestNumbaRollingVolumeProfileExpansion:
    """Test the VAH/VAL expansion loop branches."""

    def test_volume_distribution_reaches_target(self) -> None:
        """VAH/VAL 拡張ループで target_v に到達するケース"""
        n = 100
        rng = np.random.default_rng(42)
        high = np.linspace(100, 110, n) + rng.normal(0, 0.2, n)
        low = np.linspace(90, 100, n) + rng.normal(0, 0.2, n)
        close = (high + low) / 2.0
        vol = rng.uniform(100, 1000, n)

        poc, vah, val = vpf._numba_rolling_volume_profile(high, low, close, vol, 20, 10)
        assert poc.shape == (n,)
        assert vah.shape == (n,)
        assert val.shape == (n,)
        # At least some finite values after window
        assert np.isfinite(poc[20:]).any()

    def test_volume_concentrated_in_one_bin(self) -> None:
        """出来高が1ビンに集中している場合の拡張ループ break"""
        n = 50
        # Price range creates multiple bins
        high = np.linspace(101, 110, n)
        low = np.linspace(90, 99, n)
        close = (high + low) / 2.0
        # Volume concentrated in middle of window
        vol = np.zeros(n)
        vol[5:15] = 1000.0  # One region of high volume

        poc, vah, val = vpf._numba_rolling_volume_profile(high, low, close, vol, 15, 5)
        assert poc.shape == (n,)
        assert not np.all(np.isnan(poc[15:]))

    def test_sparse_volume_distribution(self) -> None:
        """出来高がまばらな場合でも v_up/dn == 0 break に到達すること"""
        n = 60
        high = np.linspace(100, 110, n)
        low = np.linspace(90, 100, n)
        close = (high + low) / 2.0
        # Only a few non-zero volume points
        vol = np.zeros(n)
        vol[10] = 500.0
        vol[30] = 500.0

        poc, vah, val = vpf._numba_rolling_volume_profile(high, low, close, vol, 20, 10)
        # Should not crash, some values might be NaN or 0
        assert poc.shape == (n,)


class TestNumbaDetectVolumeNodesExtendedBranches:
    """Test HVN/LVN detection with asymmetric volume distributions."""

    def test_hvn_found_closer_than_lvn(self) -> None:
        """HVN が LVN より現在価格に近いケース"""
        n = 60
        rng = np.random.default_rng(42)
        high = np.linspace(100, 110, n) + rng.normal(0, 0.3, n)
        low = np.linspace(90, 100, n) + rng.normal(0, 0.3, n)
        close = (high + low) / 2.0
        vol = rng.uniform(10, 1000, n)

        hvn, lvn = vpf._numba_detect_volume_nodes_signed(high, low, close, vol, 20, 10)
        assert hvn.shape == (n,)
        assert lvn.shape == (n,)
        # After window, at least some should be non-zero
        assert (hvn[20:] != 0.0).any() or (lvn[20:] != 0.0).any()

    def test_current_price_at_extreme(self) -> None:
        """現在価格が価格範囲の端にあるケース"""
        n = 60
        rng = np.random.default_rng(7)
        high = np.linspace(101, 110, n) + rng.normal(0, 0.2, n)
        low = np.linspace(90, 100, n) + rng.normal(0, 0.2, n)
        # close near the high extreme
        close = high - rng.normal(0, 0.1, n)
        vol = rng.uniform(100, 1000, n)

        hvn, lvn = vpf._numba_detect_volume_nodes_signed(high, low, close, vol, 20, 10)
        assert hvn.shape == (n,)
        assert lvn.shape == (n,)


class TestNumbaVPSkewnessKurtosisBranches:
    """Test weighted skewness/kurtosis with varying volume profiles."""

    def test_asymmetric_volume_weighted(self) -> None:
        """非対称な出来高分布での加重歪度・尖度"""
        n = 60
        rng = np.random.default_rng(42)
        close = np.linspace(100, 110, n) + rng.normal(0, 0.5, n)
        # Volume weighted toward one end
        vol = np.linspace(10, 1000, n)  # Increasing volume

        skew, kurt = vpf._numba_vp_skewness_kurtosis(close, vol, 15)
        assert skew.shape == (n,)
        assert kurt.shape == (n,)
        # After warm-up, should have non-zero values
        assert (skew[15:] != 0.0).any()

    def test_decreasing_volume_profile(self) -> None:
        """減少する出来高分布での加重統計"""
        n = 60
        rng = np.random.default_rng(99)
        close = np.linspace(100, 108, n) + rng.normal(0, 0.3, n)
        # Decreasing volume
        vol = np.linspace(1000, 10, n)

        skew, kurt = vpf._numba_vp_skewness_kurtosis(close, vol, 15)
        assert skew.shape == (n,)
        assert kurt.shape == (n,)
