"""
Coverage gap tests for the original indicator modules.

The main coverage suite (``test_original_indicators_coverage.py``) exercises
the Numba-based indicators.  This file targets the modules and branches that
remain uncovered when Numba JIT is disabled (``NUMBA_DISABLE_JIT=1``):

- The pure-pandas indicators (GRI / RWI / TII / TTF) that lacked dedicated
  tests in this directory.
- Defensive branches inside the Numba loops (zero denominators, constant
  input, short-data guards, ...).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _import_original_module(name: str) -> Any:
    """Import an original indicator submodule via importlib."""
    return importlib.import_module(
        f"app.services.indicators.technical_indicators.original.{name}"
    )


def _make_ohlcv(rows: int = 200) -> pd.DataFrame:
    """Deterministic OHLCV frame with linear trend and small noise."""
    idx = pd.date_range("2023-01-01", periods=rows, freq="h")
    base = np.linspace(100.0, 110.0, rows)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, rows)
    close = base + noise
    high = close + 0.5
    low = close - 0.5
    open_ = close + rng.normal(0, 0.05, rows)
    volume = rng.integers(100, 1000, rows).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# GRI (Gopalakrishnan Range Index)
# ---------------------------------------------------------------------------


class TestGRIGapCoverage:
    """GRI is pure pandas and had no dedicated tests in this directory."""

    def test_valid_data_returns_series(self) -> None:
        m = _import_original_module("gri")
        df = _make_ohlcv(60)
        result = m.gri(df["high"], df["low"], df["close"], length=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert not result.isna().all()

    def test_offset_shifts_result(self) -> None:
        m = _import_original_module("gri")
        df = _make_ohlcv(60)
        base = m.gri(df["high"], df["low"], df["close"], length=5)
        shifted = m.gri(df["high"], df["low"], df["close"], length=5, offset=1)
        # offset=1 shifts the series forward by one row.
        assert np.isnan(shifted.iloc[0])
        assert np.isclose(shifted.iloc[5], base.iloc[4])

    def test_empty_input_returns_empty(self) -> None:
        m = _import_original_module("gri")
        empty = pd.Series([], dtype=float)
        result = m.gri(empty, empty, empty)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_invalid_length_raises(self) -> None:
        m = _import_original_module("gri")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="length must be positive"):
            m.gri(df["high"], df["low"], df["close"], length=-1)


# ---------------------------------------------------------------------------
# RWI (Random Walk Index)
# ---------------------------------------------------------------------------


class TestRWIGapCoverage:
    """RWI is pure pandas and had no dedicated tests in this directory."""

    def test_invalid_length_raises(self) -> None:
        m = _import_original_module("random_walk_index")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.rwi(df["high"], df["low"], df["close"], length=1)

    def test_valid_data_returns_named_pair(self) -> None:
        m = _import_original_module("random_walk_index")
        df = _make_ohlcv(120)
        rwi_high, rwi_low = m.rwi(df["high"], df["low"], df["close"], length=14)
        assert isinstance(rwi_high, pd.Series)
        assert isinstance(rwi_low, pd.Series)
        assert rwi_high.name == "RWI_HIGH"
        assert rwi_low.name == "RWI_LOW"
        assert len(rwi_high) == len(df)
        assert len(rwi_low) == len(df)

    def test_constant_prices_all_nan(self) -> None:
        m = _import_original_module("random_walk_index")
        const = pd.Series(np.full(40, 100.0))
        rwi_high, rwi_low = m.rwi(const, const, const, length=5)
        # Zero true-range => 0/0 => NaN candidates, then max() collapses to NaN.
        assert rwi_high.dropna().empty
        assert rwi_low.dropna().empty

    def test_true_range_helper(self) -> None:
        m = _import_original_module("random_walk_index")
        df = _make_ohlcv(60)
        tr = m._calculate_true_range(df["high"], df["low"], df["close"])
        assert isinstance(tr, pd.Series)
        assert len(tr) == len(df)
        # max(axis=1) skips NaN, so the first row is the finite high-low term.
        assert np.isclose(tr.iloc[0], df["high"].iloc[0] - df["low"].iloc[0])
        expected = max(
            df["high"].iloc[1] - df["low"].iloc[1],
            abs(df["high"].iloc[1] - df["close"].iloc[0]),
            abs(df["low"].iloc[1] - df["close"].iloc[0]),
        )
        assert np.isclose(tr.iloc[1], expected)


# ---------------------------------------------------------------------------
# TII (Trend Intensity Index)
# ---------------------------------------------------------------------------


class TestTIIGapCoverage:
    """TII is pure pandas and had no dedicated tests in this directory."""

    def test_invalid_sma_length_raises(self) -> None:
        m = _import_original_module("trend_intensity_index")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="sma_length must be >= 1"):
            m.trend_intensity_index(
                df["close"], df["high"], df["low"], length=14, sma_length=0
            )

    def test_valid_data_bounded(self) -> None:
        m = _import_original_module("trend_intensity_index")
        df = _make_ohlcv(80)
        result = m.trend_intensity_index(
            df["close"], df["high"], df["low"], length=14, sma_length=30
        )
        assert result.name == "TII_14_30"
        valid = result.dropna()
        assert ((valid >= 0.0) & (valid <= 100.0)).all()

    def test_insufficient_data_returns_nan(self) -> None:
        m = _import_original_module("trend_intensity_index")
        df = _make_ohlcv(10)
        result = m.trend_intensity_index(
            df["close"], df["high"], df["low"], length=14, sma_length=30
        )
        assert isinstance(result, pd.Series)
        assert result.name == "TII_14_30"
        assert result.isna().all()


# ---------------------------------------------------------------------------
# TTF (Trend Trigger Factor)
# ---------------------------------------------------------------------------


class TestTTFGapCoverage:
    """TTF is pure pandas and had no dedicated tests in this directory."""

    def test_invalid_length_raises(self) -> None:
        m = _import_original_module("trend_trigger_factor")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.ttf(df["high"], df["low"], df["close"], length=1)

    def test_valid_data_finite(self) -> None:
        m = _import_original_module("trend_trigger_factor")
        df = _make_ohlcv(120)
        result = m.ttf(df["high"], df["low"], df["close"], length=15)
        assert result.name == "TTF_15"
        assert np.isfinite(result.dropna()).all()

    def test_constant_prices_zero_denominator(self) -> None:
        m = _import_original_module("trend_trigger_factor")
        const = pd.Series(np.full(40, 100.0))
        result = m.ttf(const, const, const, length=5)
        # buy_power == sell_power == 0 => denominator == 0 => zero_mask => 0.0
        assert result.iloc[:9].isna().all()
        assert np.allclose(result.iloc[9:], 0.0)

    def test_empty_input_returns_empty(self) -> None:
        m = _import_original_module("trend_trigger_factor")
        empty = pd.Series([], dtype=float)
        result = m.ttf(empty, empty, empty)
        assert isinstance(result, pd.Series)
        assert result.empty


# ---------------------------------------------------------------------------
# Defensive branches inside Numba loops
# ---------------------------------------------------------------------------


class TestWindowHelpersVarianceClamp:
    """Cover the ``variance < 0`` numerical guard in the shared helpers."""

    def test_large_magnitude_values_do_not_yield_negative_std(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([1e8, 1e8, 1e8, 1e8, 1e8 + 0.1])
        mean, std = m._window_mean_and_std(arr, 0, 5)
        assert std >= 0.0
        assert not np.isnan(std)
        assert mean > 0.0

        mean_f, std_f, count = m._window_mean_and_std_finite(arr, 0, 5)
        assert count == 5
        assert std_f >= 0.0
        assert not np.isnan(std_f)


class TestEntropyVolatilityNoMatchBranch:
    """Cover the ``a_count == 0 or b_count == 0`` fallback (EVI = 0)."""

    def test_no_matching_subsequences_returns_zero(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        # Monotonic returns: no subsequence pair falls inside a tight threshold.
        returns = np.arange(60, dtype=float)
        result = m._njit_entropy_volatility_loop(returns, 30, 2, 0.01)
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()


class TestAdaptiveEntropyShortInput:
    """Cover the ``n < window`` early-return in the entropy loop."""

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("adaptive_entropy")
        data = np.arange(5, dtype=float)
        result = m._njit_entropy_loop(data, 14)
        assert np.isnan(result).all()


class TestDeMarkerConstantPrices:
    """Cover the zero-denominator fallback (DeMarker = 50)."""

    def test_constant_prices_produce_50(self) -> None:
        m = _import_original_module("demarker")
        const = np.full(30, 100.0)
        result = m._njit_demarker_loop(const, const, 14)
        valid = result[~np.isnan(result)]
        assert (valid == 50.0).all()


class TestFRAMAConstantPrices:
    """Cover the ``dimen = 1.0`` fallback for flat windows."""

    def test_constant_prices_follow_price(self) -> None:
        m = _import_original_module("frama")
        prices = np.full(40, 100.0)
        result = m._njit_frama_loop(
            prices,
            length=16,
            half=8,
            log2=np.log(2.0),
            w=0.5,
            alpha_min=0.01,
            alpha_max=1.0,
        )
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, 100.0)


class TestVortexRSIShortInput:
    """Cover the ``n < length + 1`` early-return in the vortex RSI loop."""

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("vortex_rsi")
        high = np.full(10, 101.0)
        low = np.full(10, 99.0)
        close = np.full(10, 100.0)
        result = m._njit_vortex_rsi_loop(high, low, close, 14)
        assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# Short-data guards in other Numba loops
# ---------------------------------------------------------------------------


class TestRMIShortInput:
    """Cover the ``n < length + momentum`` early-return in the RMI loop."""

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("rmi")
        prices = np.linspace(100, 110, 10)
        result = m._njit_rmi_loop(prices, 14, 5)
        assert np.isnan(result).all()


class TestDamianiShortInput:
    """Cover the ``n < min_len + 1`` early-return in the volatmeter loop."""

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("damiani_volatmeter")
        high = np.linspace(101, 110, 10)
        low = np.linspace(99, 90, 10)
        close = np.linspace(100, 105, 10)
        result = m._njit_damiani_volatmeter_loop(high, low, close, 13, 20, 40, 100)
        assert np.isnan(result).all()


class TestMMIShortInput:
    """Cover the ``n < length or length < 3`` early-return in the MMI loop."""

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("market_meanness_index")
        prices = np.linspace(100, 110, 10)
        result = m._njit_mmi_loop(prices, 20)
        assert np.isnan(result).all()


class TestSAMConstantAndShortInput:
    """Cover the zero-volatility path and the outer NaN fallback for SAM."""

    def test_constant_prices_zero_momentum(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        prices = np.full(40, 100.0)
        result = m._njit_sam_loop(prices, 14, 5)
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()

    def test_short_data_returns_all_nan(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        prices = np.linspace(100, 110, 10)
        result = m._njit_sam_loop(prices, 14, 5)
        assert np.isnan(result).all()

    def test_outer_insufficient_data_returns_nan(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        close = _make_ohlcv(10)["close"]
        result = m.smoothed_adaptive_momentum(close, length=14, smooth_length=5)
        assert isinstance(result, pd.Series)
        assert result.name == "SAM_14_5"
        assert result.isna().all()


class TestFibonacciNegativeReturns:
    """Cover the negative sign branch (``sign_sum -= 1``)."""

    def test_descending_prices_produce_negative_output(self) -> None:
        m = _import_original_module("fibonacci_cycle")
        prices = np.linspace(110, 100, 100)
        cycle_periods = np.array([8, 13, 21, 34, 55], dtype=np.int64)
        fib_ratios = np.array([0.618, 1.0, 1.618, 2.618])
        result = m._njit_fibonacci_cycle_loop(
            prices, cycle_periods, fib_ratios, max_period=55
        )
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert (valid <= 0.0).all()


class TestQuantumFlowConstantPrices:
    """Cover the zero-correlation branch (``denom <= 1e-12``)."""

    def test_constant_prices_zero_correlation(self) -> None:
        m = _import_original_module("quantum_flow")
        n = 30
        prices = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        volumes = np.full(n, 500.0)
        wavelet = m._simple_wavelet_transform(prices, 14)
        result = m._njit_quantum_flow_loop(prices, highs, lows, volumes, 14, wavelet)
        assert result.shape == (n,)
        assert np.isfinite(result).all()


class TestEntropyVolatilityEmptyInput:
    """Cover the outer NaN fallback for EVI (empty series)."""

    def test_empty_input_returns_nan(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        close = pd.Series([], dtype=float)
        result = m.entropy_volatility_index(close)
        assert isinstance(result, pd.Series)
        assert result.empty


class TestTIIInvalidLength:
    """Cover the ``length < 1`` validation branch for TII."""

    def test_invalid_length_raises(self) -> None:
        m = _import_original_module("trend_intensity_index")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="length must be >= 1"):
            m.trend_intensity_index(df["close"], df["high"], df["low"], length=0)


class TestHurstShortAndGuard:
    """Cover the ``continue`` guards in the Hurst loop."""

    def test_win_too_small_for_max_lag(self) -> None:
        m = _import_original_module("hurst_exponent")
        prices = np.linspace(100, 110, 200)
        result = m._njit_hurst_loop(prices, 10, 20)
        # m = 9 < max_lag + 2 = 22 => always ``continue`` => all NaN.
        assert np.isnan(result).all()

    def test_too_few_lags(self) -> None:
        m = _import_original_module("hurst_exponent")
        prices = np.linspace(100, 110, 200)
        result = m._njit_hurst_loop(prices, 100, 2)
        # lags = arange(2, min(3, 49)) has len 1 < 3 => all NaN.
        assert np.isnan(result).all()

    def test_constant_prices_zero_hurst(self) -> None:
        m = _import_original_module("hurst_exponent")
        prices = np.full(200, 100.0)
        result = m._njit_hurst_loop(prices, 100, 10)
        valid = result[~np.isnan(result)]
        # Constant prices => all log_rs are 0 => ss_xy == 0 => Hurst == 0.
        if len(valid) > 0:
            assert np.allclose(valid, 0.0)
