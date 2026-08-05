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


# ---------------------------------------------------------------------------
# Remaining branches in the 90%-modules (Connors RSI, CTFD, FRAMA, HRI, PFE,
# Prime Oscillator, Quantum Flow, RWI, RMI, Vortex RSI)
# ---------------------------------------------------------------------------


class TestConnorsRSIResidualBranches:
    """Cover the outer happy path and the partial-count / clamp branches."""

    def test_outer_function_valid_data(self) -> None:
        m = _import_original_module("connors_rsi")
        df = _make_ohlcv(150)
        result = m.connors_rsi(
            df["close"], rsi_periods=3, streak_periods=2, rank_periods=30
        )
        assert isinstance(result, pd.Series)
        assert result.name == "CONNORS_RSI_3_2_30"
        assert not result[30:].isna().all()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_extreme_prices_nan_component_and_upper_clamp(self) -> None:
        m = _import_original_module("connors_rsi")
        # Alternating +/- 1e308 overflows every close-RSI change to +/- inf,
        # so close_rsi becomes NaN and the outer loop takes the count == 2
        # path, where (total / 2) * 1.5 can exceed 100 and get clamped.
        prices = pd.Series(np.array([1e308, -1e308] * 60, dtype=float))
        result = m.connors_rsi(prices, rsi_periods=2, streak_periods=2, rank_periods=2)
        valid = result.iloc[2:].to_numpy()
        assert np.isfinite(valid).all()
        assert ((valid >= 0.0) & (valid <= 100.0)).all()
        assert (valid == 100.0).any()


class TestChaosFractalDimensionResidualBranches:
    """Cover the remaining defensive branches of the CTFD helpers."""

    def test_correlation_dimension_negative_points_guard(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        # 10 points with embedding_dim=5 and time_delay=3 make n_points <= 0.
        prices = np.arange(10, dtype=float)
        result = m._calculate_correlation_dimension_impl(prices, 5, 3)
        assert result == 1.0

    def test_constant_prices_singular_coeffs(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        df = _make_ohlcv(60)
        const = pd.Series(np.full(60, 100.0), index=df.index)
        ctf, sig = m.chaos_fractal_dimension(
            const, const, const, const, length=15, signal_length=2
        )
        # Constant windows => correlation dimension falls back to 1.0 and the
        # quadratic fit is singular => chaos_score = 1.0 * 0.7 + 0.3, so the
        # raw score is 1 / (1 + 1.0) = 0.5 and normalization keeps it.
        valid = ctf.dropna()
        assert np.allclose(valid, 0.5)
        assert sig.name == "CTFD_SIGNAL"

    def test_normalize_clamps_lower_bound(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        # A drop below the recent mean by many standard deviations clamps to -1.
        raw = np.array([0.8, 0.9] * 6 + [0.2] * 30, dtype=float)
        result = m._njit_ctfd_normalize(raw, 2)
        assert (result == -1.0).any()

    def test_normalize_clamps_upper_bound(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        # A jump above the recent mean by many standard deviations clamps to 1.
        raw = np.array([0.2, 0.3] * 6 + [0.9] * 30, dtype=float)
        result = m._njit_ctfd_normalize(raw, 2)
        assert (result == 1.0).any()


class TestFRAMAResidualBranches:
    """Cover the remaining FRAMA loop branches."""

    def test_varying_prices_normal_dimen_branch(self) -> None:
        m = _import_original_module("frama")
        df = _make_ohlcv(80)
        result = m.frama(df["close"], length=16, slow=200)
        assert result.name == "FRAMA"
        assert not result[16:].isna().all()

    def test_choppy_prices_alpha_min_clamp(self) -> None:
        m = _import_original_module("frama")
        rng = np.random.default_rng(7)
        prices = np.cumsum(rng.normal(0, 1, 300)) + 100.0
        w = 2.303 * np.log(2.0 / 201.0)
        result = m._njit_frama_loop(prices, 16, 8, np.log(2.0), w, 0.01, 1.0)
        assert np.isfinite(result[16:]).all()

    def test_alpha_max_clamp_via_small_max(self) -> None:
        m = _import_original_module("frama")
        # Flat windows give dimen == 1.0 => alpha == 1.0; an alpha_max below
        # 1.0 forces the upper clamp branch.
        prices = np.full(40, 100.0)
        result = m._njit_frama_loop(prices, 16, 8, np.log(2.0), 0.5, 0.01, 0.5)
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, 100.0)

    def test_nan_seed_price_prev_filt_fallback(self) -> None:
        m = _import_original_module("frama")
        # A NaN right at the seed index (length - 1) makes the next iteration
        # read a non-finite prev_filt and fall back to the raw price.
        prices = np.concatenate([np.full(15, 100.0), [np.nan], np.full(10, 110.0)])
        result = m._njit_frama_loop(prices, 16, 8, np.log(2.0), 0.5, 0.01, 1.0)
        assert result.shape == (26,)
        assert np.isnan(result[15])
        assert np.isclose(result[16], prices[16])

    def test_frama_is_adaptive_average_not_raw_price(self) -> None:
        m = _import_original_module("frama")
        # Regression test: the min/max window helper returns (min, max); a
        # swapped unpacking used to make n1/n2/n3 negative so dimen was always
        # 1.0 and FRAMA returned the raw price.  It must now smooth the series
        # while staying within the price range.
        rng = np.random.default_rng(11)
        prices = np.cumsum(rng.normal(0, 1, 200)) + 100.0
        w = 2.303 * np.log(2.0 / 201.0)
        result = m._njit_frama_loop(prices, 16, 8, np.log(2.0), w, 0.01, 1.0)
        valid = result[~np.isnan(result)]
        assert not np.allclose(valid, prices[-len(valid) :])
        assert valid.min() >= prices.min()
        assert valid.max() <= prices.max()


class TestHarmonicResonanceResidualBranches:
    """Cover the remaining HRI branches."""

    def test_find_dominant_freqs_short_magnitude(self) -> None:
        m = _import_original_module("harmonic_resonance")
        # n in [4, 5] gives a magnitude of length < 3 => default frequencies.
        res = m._njit_find_dominant_freqs(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.allclose(res, [0.1, 0.2, 0.3])

    def test_loop_empty_freqs_path(self) -> None:
        m = _import_original_module("harmonic_resonance")
        # length=2 makes every window too short to find dominant frequencies,
        # exercising the ``len(dominant_freqs) == 0`` early path.
        prices = np.arange(10, dtype=float) + 100.0
        result = m._njit_harmonic_resonance_loop(prices, 2, 5, min_period=2)
        assert np.isnan(result[:2]).all()
        assert (result[2:] == 0.0).all()

    def test_constant_prices_zero_resonance(self) -> None:
        m = _import_original_module("harmonic_resonance")
        df = _make_ohlcv(80)
        const = pd.Series(np.full(80, 100.0), index=df.index)
        hri, sig = m.harmonic_resonance(const, const, const, length=20)
        # No variation => resonance scores stay 0 => final pass keeps 0.
        assert hri[30:].eq(0.0).all()
        assert sig.name == "HRI_SIGNAL"


class TestPFEResidualBranches:
    """Cover the remaining PFE branches."""

    def test_constant_prices_zero_direction(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        prices = np.full(60, 100.0)
        result = m._njit_pfe_loop(prices, 10, 5)
        # direction == 0 => value forced to 0.
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()

    def test_nan_input_returns_all_nan(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        # NaN propagates through the cumulative sum => no finite raw value =>
        # the ``first_valid == -1`` early return.
        prices = np.concatenate([np.full(5, 100.0), np.full(55, np.nan)])
        result = m._njit_pfe_loop(prices, 10, 5)
        assert np.isnan(result).all()

    def test_outer_function_valid_data(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        df = _make_ohlcv(80)
        result = m.pfe(df["close"], length=10, smoothing_length=5)
        assert result.name == "PFE_10_5"
        assert not result.dropna().empty

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        close = pd.Series(np.linspace(100, 105, 5))
        result = m.pfe(close, length=10, smoothing_length=5)
        assert isinstance(result, pd.Series)
        assert result.name == "PFE_10_5"
        assert result.isna().all()


class TestPrimeOscillatorResidualBranches:
    """Cover the remaining prime oscillator branches."""

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("prime_oscillator")
        close = pd.Series(np.arange(10, dtype=float) + 100.0)
        osc, sig = m.prime_oscillator(close, length=14)
        assert isinstance(osc, pd.Series)
        assert isinstance(sig, pd.Series)
        assert osc.isna().all()
        assert sig.isna().all()

    def test_outer_function_invalid_signal_length(self) -> None:
        m = _import_original_module("prime_oscillator")
        close = pd.Series(np.linspace(100, 110, 60))
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            m.prime_oscillator(close, length=14, signal_length=1)

    def test_constant_prices_zero_variance(self) -> None:
        m = _import_original_module("prime_oscillator")
        close = pd.Series(np.full(60, 100.0))
        osc, sig = m.prime_oscillator(close, length=14)
        # Constant prices => chg == 0 => variance 0 => raw oscillator kept.
        valid = osc.dropna()
        assert (valid == 0.0).all()

    def test_zero_prices_zero_count(self) -> None:
        m = _import_original_module("prime_oscillator")
        prices = np.zeros(60)
        primes = np.array(m._get_prime_sequence(14), dtype=np.int64)
        result = m._njit_prime_oscillator_loop(prices, primes, lookback_limit=200)
        # p_prev == 0 for every bar => bar_counts == 0 => zero-count fallback.
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()


class TestQuantumFlowResidualBranches:
    """Cover the remaining quantum flow validation branches."""

    def test_invalid_length_raises(self) -> None:
        m = _import_original_module("quantum_flow")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="length must be >= 5"):
            m.quantum_flow(df["close"], df["high"], df["low"], df["volume"], length=4)

    def test_invalid_flow_length_raises(self) -> None:
        m = _import_original_module("quantum_flow")
        df = _make_ohlcv(60)
        with pytest.raises(ValueError, match="flow_length must be >= 3"):
            m.quantum_flow(
                df["close"], df["high"], df["low"], df["volume"], flow_length=2
            )


class TestRWIRemainingBranches:
    """Cover the RWI validation-failure path (nan pair)."""

    def test_insufficient_data_returns_nan_pair(self) -> None:
        m = _import_original_module("random_walk_index")
        df = _make_ohlcv(10)
        rwi_high, rwi_low = m.rwi(df["high"], df["low"], df["close"], length=14)
        assert rwi_high.name == "RWI_HIGH"
        assert rwi_low.name == "RWI_LOW"
        assert rwi_high.isna().all()
        assert rwi_low.isna().all()


class TestRMIOuterHappyPath:
    """Cover the RMI public-function happy path."""

    def test_outer_function_valid_data(self) -> None:
        m = _import_original_module("rmi")
        close = pd.Series(np.linspace(100, 110, 100))
        result = m.rmi(close, length=14, momentum=5)
        assert isinstance(result, pd.Series)
        assert result.name == "RMI_14_5"
        valid = result.dropna()
        assert not valid.empty
        assert ((valid >= 0) & (valid <= 100)).all()


class TestVortexRSIRemainingBranches:
    """Cover the ``sum_vi <= 1e-12`` fallback inside the vortex loop."""

    def test_zero_vortex_sum_returns_50(self) -> None:
        m = _import_original_module("vortex_rsi")
        # Inverted OHLC bars make VM+ == VM- == 0 while TR stays positive,
        # hitting the sum_vi == 0 fallback (result = 50.0).
        high = np.array([100.0, 90.0])
        low = np.array([90.0, 100.0])
        close = np.array([95.0, 95.0])
        result = m._njit_vortex_rsi_loop(high, low, close, 1)
        assert np.isnan(result[0])
        assert np.isclose(result[1], 50.0)
