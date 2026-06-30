"""
Coverage tests for the under-tested OriginalIndicators modules.

These tests target the Numba-internal functions and edge cases that
the existing wrapper-level tests do not exercise, in order to
improve line/branch coverage of the original/ indicator package.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_original_module(name: str) -> Any:
    """Import an original indicator submodule via importlib.

    Some decorator chains (e.g. ``handle_pandas_ta_errors``) make the
    ``from ... import name`` form behave unexpectedly, so we always
    use ``importlib.import_module`` to get the actual module object.
    """
    return importlib.import_module(
        f"app.services.indicators.technical_indicators.original.{name}"
    )


def _make_series(values: list[float] | np.ndarray) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr, index=pd.date_range("2023-01-01", periods=len(arr), freq="h"))


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
# Hurst Exponent (lowest coverage)
# ---------------------------------------------------------------------------


class TestHurstExponentDeepCoverage:
    """Target the inner Numba loop and validation branches."""

    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("hurst_exponent")
        # Random walk-like prices: at least 2 * max_lag + 1 values needed.
        rng = np.random.default_rng(0)
        prices = np.cumsum(rng.normal(0, 1, 250)) + 100.0
        result = m._njit_hurst_loop(prices, 100, 20)
        assert result.shape == (250,)
        # The first 99 entries should be NaN.
        assert np.isnan(result[:99]).all()
        # After warm-up, we expect at least some finite values.
        assert np.isfinite(result[100:]).any()

    def test_inner_loop_handles_constant_segment(self) -> None:
        m = _import_original_module("hurst_exponent")
        # A flat segment should make std_seg < 1e-12, the loop must skip cleanly.
        prices = np.full(220, 100.0)
        result = m._njit_hurst_loop(prices, 100, 10)
        assert result.shape == (220,)

    def test_inner_loop_small_max_lag(self) -> None:
        m = _import_original_module("hurst_exponent")
        prices = np.linspace(100, 110, 200)
        # min(max_lag + 1, m // 2) when m is large enough
        result = m._njit_hurst_loop(prices, 50, 5)
        assert result.shape == (200,)

    def test_outer_function_validates_length(self) -> None:
        m = _import_original_module("hurst_exponent")
        close = _make_series([100.0, 101.0, 102.0, 103.0, 104.0] * 50)
        with pytest.raises(ValueError, match="length must be >= 10"):
            m.hurst_exponent(close, length=5)

    def test_outer_function_validates_max_lag(self) -> None:
        m = _import_original_module("hurst_exponent")
        close = _make_series([100.0, 101.0, 102.0, 103.0, 104.0] * 50)
        with pytest.raises(ValueError, match="max_lag must be >= 2"):
            m.hurst_exponent(close, length=20, max_lag=1)

    def test_outer_function_insufficient_data_returns_nan(self) -> None:
        m = _import_original_module("hurst_exponent")
        close = _make_series([100.0, 101.0, 102.0, 103.0])
        result = m.hurst_exponent(close, length=20)
        assert isinstance(result, pd.Series)
        assert result.isna().all()
        assert result.name == "HURST_20_20"

    def test_outer_function_returns_named_series(self) -> None:
        m = _import_original_module("hurst_exponent")
        close = _make_series(np.linspace(100, 110, 200))
        result = m.hurst_exponent(close, length=20, max_lag=10)
        assert isinstance(result, pd.Series)
        assert result.name == "HURST_20_10"
        # First (length-1) values are NaN.
        assert result.iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# Connors RSI
# ---------------------------------------------------------------------------


class TestConnorsRSIDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("connors_rsi")
        prices = np.array(
            [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 100, 101] * 20,
            dtype=float,
        )
        result = m._njit_connors_rsi_loop(prices, 3, 2, 100)
        assert result.shape == prices.shape
        # After max(rsi, streak, rank)=100, we expect a finite value at index 100
        assert np.isfinite(result[100:]).any()

    def test_streak_reset_on_equal_prices(self) -> None:
        m = _import_original_module("connors_rsi")
        # Identical consecutive prices reset streak to 0.
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0] * 30)
        result = m._njit_connors_rsi_loop(prices, 3, 2, 30)
        # Should still produce something finite at the end.
        assert np.isfinite(result[30:]).any()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("connors_rsi")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="rsi_periods must be >= 2"):
            m.connors_rsi(close, rsi_periods=1)
        with pytest.raises(ValueError, match="streak_periods must be >= 1"):
            m.connors_rsi(close, streak_periods=0)
        with pytest.raises(ValueError, match="rank_periods must be >= 2"):
            m.connors_rsi(close, rank_periods=1)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("connors_rsi")
        close = _make_series([100.0] * 5)
        result = m.connors_rsi(close)
        assert isinstance(result, pd.Series)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Harmonic Resonance
# ---------------------------------------------------------------------------


class TestHarmonicResonanceDeepCoverage:
    def test_inner_helpers(self) -> None:
        m = _import_original_module("harmonic_resonance")
        x = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
        mag = m._njit_dft_magnitude(x)
        # DFT magnitude length is n // 2.
        assert mag.shape == (4,)

    def test_find_dominant_freqs_short_signal(self) -> None:
        m = _import_original_module("harmonic_resonance")
        # n < 4: returns zeros array
        res = m._njit_find_dominant_freqs(np.array([1.0, 2.0, 3.0]))
        assert isinstance(res, np.ndarray)
        assert res.shape == (0,)

    def test_find_dominant_freqs_returns_peak(self) -> None:
        m = _import_original_module("harmonic_resonance")
        # A pure sinusoid: 64 samples at freq = 4 cycles.
        t = np.arange(64)
        prices = np.cos(2.0 * np.pi * 4.0 * t / 64.0)
        res = m._njit_find_dominant_freqs(prices)
        assert res.shape[0] > 0
        # All returned frequencies are in (0, 0.5)
        assert ((res > 0.0) & (res < 0.5)).all()

    def test_find_dominant_freqs_flat_signal(self) -> None:
        m = _import_original_module("harmonic_resonance")
        prices = np.zeros(64)
        # Constant input has no peaks.
        res = m._njit_find_dominant_freqs(prices)
        # Returns the default fallback
        assert np.allclose(res, [0.1, 0.2, 0.3])

    def test_bandpass_filter_identity(self) -> None:
        m = _import_original_module("harmonic_resonance")
        x = np.linspace(0, 1, 50)
        y = m._njit_apply_bandpass_res(x, freq=0.1, q=2.0)
        assert y.shape == x.shape

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("harmonic_resonance")
        df = _make_ohlcv(200)
        with pytest.raises(ValueError, match="length must be >= 10"):
            m.harmonic_resonance(df["close"], df["high"], df["low"], length=5)
        with pytest.raises(ValueError, match="resonance_bands must be between"):
            m.harmonic_resonance(
                df["close"],
                df["high"],
                df["low"],
                length=20,
                resonance_bands=2,
            )
        with pytest.raises(ValueError, match="resonance_bands must be between"):
            m.harmonic_resonance(
                df["close"],
                df["high"],
                df["low"],
                length=20,
                resonance_bands=11,
            )
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            m.harmonic_resonance(
                df["close"],
                df["high"],
                df["low"],
                length=20,
                signal_length=1,
            )

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("harmonic_resonance")
        df = _make_ohlcv(5)
        hri, sig = m.harmonic_resonance(df["close"], df["high"], df["low"])
        assert isinstance(hri, pd.Series)
        assert isinstance(sig, pd.Series)
        assert hri.isna().all()
        assert sig.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("harmonic_resonance")
        df = _make_ohlcv(120)
        hri, sig = m.harmonic_resonance(df["close"], df["high"], df["low"])
        assert hri.name == "HARMONIC_RESONANCE"
        assert sig.name == "HRI_SIGNAL"


# ---------------------------------------------------------------------------
# Chaos Fractal Dimension
# ---------------------------------------------------------------------------


class TestChaosFractalDimensionDeepCoverage:
    def test_correlation_dimension_impl_short(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        # n_prices < embedding_dim * 2
        result = m._calculate_correlation_dimension_impl(
            np.array([1.0, 2.0, 3.0]), embedding_dim=3, time_delay=1
        )
        assert result == 1.0

    def test_correlation_dimension_impl_returns_finite(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        rng = np.random.default_rng(0)
        prices = np.cumsum(rng.normal(0, 1, 60)) + 100.0
        result = m._calculate_correlation_dimension_impl(prices, 3, 1)
        assert 1.0 <= result <= 5.0

    def test_solve_3x3_singular(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        b = np.array([1.0, 2.0, 3.0])
        x = m._njit_solve_3x3(A, b)
        assert np.allclose(x, 0.0)

    def test_solve_3x3_known_solution(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        # 3 * x + 0 * y + 0 * z = 6
        # 0 * x + 3 * y + 0 * z = 12
        # 0 * x + 0 * y + 3 * z = 18  =>  x=2, y=4, z=6
        A = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        b = np.array([6.0, 12.0, 18.0])
        x = m._njit_solve_3x3(A, b)
        assert np.allclose(x, [2.0, 4.0, 6.0])

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        df = _make_ohlcv(200)
        with pytest.raises(ValueError, match="length must be >= 15"):
            m.chaos_fractal_dimension(
                df["close"],
                df["high"],
                df["low"],
                df["volume"],
                length=10,
            )
        with pytest.raises(ValueError, match="embedding_dim must be between"):
            m.chaos_fractal_dimension(
                df["close"],
                df["high"],
                df["low"],
                df["volume"],
                length=20,
                embedding_dim=1,
            )
        with pytest.raises(ValueError, match="embedding_dim must be between"):
            m.chaos_fractal_dimension(
                df["close"],
                df["high"],
                df["low"],
                df["volume"],
                length=20,
                embedding_dim=10,
            )
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            m.chaos_fractal_dimension(
                df["close"],
                df["high"],
                df["low"],
                df["volume"],
                length=20,
                signal_length=1,
            )

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        df = _make_ohlcv(5)
        ctf, sig = m.chaos_fractal_dimension(
            df["close"], df["high"], df["low"], df["volume"]
        )
        assert ctf.isna().all()
        assert sig.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("chaos_fractal_dimension")
        df = _make_ohlcv(120)
        ctf, sig = m.chaos_fractal_dimension(
            df["close"], df["high"], df["low"], df["volume"]
        )
        assert ctf.name == "CHAOS_FRACTAL_DIM"
        assert sig.name == "CTFD_SIGNAL"


# ---------------------------------------------------------------------------
# Vortex RSI
# ---------------------------------------------------------------------------


class TestVortexRSIDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("vortex_rsi")
        rng = np.random.default_rng(0)
        n = 80
        high = np.cumsum(rng.normal(0, 1, n)) + 102
        low = np.cumsum(rng.normal(0, 1, n)) + 98
        close = (high + low) / 2.0
        result = m._njit_vortex_rsi_loop(high, low, close, 14)
        # Values before length are 0 (initial state), the result should be
        # within [0, 100] or NaN.
        valid = result[~np.isnan(result)]
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_inner_loop_constant_input(self) -> None:
        m = _import_original_module("vortex_rsi")
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)
        result = m._njit_vortex_rsi_loop(high, low, close, 14)
        # Zero TR path => 50.0 fallback
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.allclose(valid, 50.0)

    def test_outer_function_invalid_length(self) -> None:
        m = _import_original_module("vortex_rsi")
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match="length must be >= 1"):
            m.vortex_rsi(df["high"], df["low"], df["close"], length=0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("vortex_rsi")
        df = _make_ohlcv(5)
        result = m.vortex_rsi(df["high"], df["low"], df["close"])
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("vortex_rsi")
        df = _make_ohlcv(80)
        result = m.vortex_rsi(df["high"], df["low"], df["close"], length=10)
        assert result.name == "VRSI_10"
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Polarized Fractal Efficiency
# ---------------------------------------------------------------------------


class TestPFEDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        prices = np.array([100.0 + i * 0.5 for i in range(60)])
        result = m._njit_pfe_loop(prices, 10, 5)
        # Trending prices => positive PFE
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()

    def test_inner_loop_descending_prices(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        prices = np.array([100.0 - i * 0.5 for i in range(60)])
        result = m._njit_pfe_loop(prices, 10, 5)
        valid = result[~np.isnan(result)]
        assert (valid < 0).all()

    def test_inner_loop_short_input(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        prices = np.array([100.0, 101.0, 102.0])
        result = m._njit_pfe_loop(prices, 10, 5)
        # Too short: all NaN
        assert np.isnan(result).all()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.pfe(close, length=1)
        with pytest.raises(ValueError, match="smoothing_length must be >= 1"):
            m.pfe(close, length=10, smoothing_length=0)

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("polarized_fractal_efficiency")
        close = _make_series(np.linspace(100, 110, 200))
        result = m.pfe(close, length=20, smoothing_length=5)
        assert result.name == "PFE_20_5"


# ---------------------------------------------------------------------------
# Smoothed Adaptive Momentum
# ---------------------------------------------------------------------------


class TestSmoothedAdaptiveMomentumDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        prices = np.linspace(100, 110, 100)
        result = m._njit_sam_loop(prices, 14, 5)
        # After warm-up (length + smooth_length - 1 = 18), at least one finite value.
        assert np.isfinite(result[20:]).any()

    def test_inner_loop_too_short(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        result = m._njit_sam_loop(prices, 14, 5)
        assert np.isnan(result).all()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.smoothed_adaptive_momentum(close, length=1)
        with pytest.raises(ValueError, match="smooth_length must be >= 1"):
            m.smoothed_adaptive_momentum(close, length=10, smooth_length=0)

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("smoothed_adaptive_momentum")
        close = _make_series(np.linspace(100, 110, 200))
        result = m.smoothed_adaptive_momentum(close, length=14, smooth_length=5)
        assert result.name == "SAM_14_5"


# ---------------------------------------------------------------------------
# RMI (Relative Momentum Index)
# ---------------------------------------------------------------------------


class TestRMIDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("rmi")
        prices = np.linspace(100, 110, 100)
        result = m._njit_rmi_loop(prices, 14, 5)
        # Values are in [0, 100]
        valid = result[~np.isnan(result)]
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_inner_loop_no_losses(self) -> None:
        m = _import_original_module("rmi")
        # Monotonically increasing prices => RMI = 100
        prices = np.linspace(100, 200, 100)
        result = m._njit_rmi_loop(prices, 14, 5)
        # After seed, value is 100
        seed_idx = 5 + 14 - 1
        assert np.isclose(result[seed_idx], 100.0)

    def test_inner_loop_no_gains(self) -> None:
        m = _import_original_module("rmi")
        prices = np.linspace(200, 100, 100)
        result = m._njit_rmi_loop(prices, 14, 5)
        seed_idx = 5 + 14 - 1
        assert np.isclose(result[seed_idx], 0.0)

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("rmi")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.rmi(close, length=1)
        with pytest.raises(ValueError, match="momentum must be >= 1"):
            m.rmi(close, length=14, momentum=0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("rmi")
        close = _make_series([100.0, 101.0, 102.0, 103.0])
        result = m.rmi(close)
        assert isinstance(result, pd.Series)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Prime Oscillator
# ---------------------------------------------------------------------------


class TestPrimeOscillatorDeepCoverage:
    def test_is_prime_values(self) -> None:
        m = _import_original_module("prime_oscillator")
        assert m._njit_is_prime(2)
        assert m._njit_is_prime(3)
        assert m._njit_is_prime(5)
        assert m._njit_is_prime(7)
        assert m._njit_is_prime(11)
        assert m._njit_is_prime(13)
        assert not m._njit_is_prime(1)
        assert not m._njit_is_prime(0)
        assert not m._njit_is_prime(4)
        assert not m._njit_is_prime(9)
        assert not m._njit_is_prime(15)

    def test_get_prime_sequence(self) -> None:
        m = _import_original_module("prime_oscillator")
        primes = m._get_prime_sequence(5)
        assert primes == [2, 3, 5, 7, 11]
        primes10 = m._get_prime_sequence(10)
        assert primes10[:5] == [2, 3, 5, 7, 11]
        assert primes10[5:] == [13, 17, 19, 23, 29]

    def test_inner_loop_too_short(self) -> None:
        m = _import_original_module("prime_oscillator")
        primes = np.array([2, 3, 5, 7, 11])
        # 10-element input but max prime is 11 > n
        prices = np.arange(10, dtype=float) + 100
        result = m._njit_prime_oscillator_loop(prices, primes, lookback_limit=200)
        # n < max_p (11) => all NaN
        assert np.isnan(result).all()

    def test_outer_function_invalid_length(self) -> None:
        m = _import_original_module("prime_oscillator")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.prime_oscillator(close, length=1)

    def test_outer_function_returns_named_pair(self) -> None:
        m = _import_original_module("prime_oscillator")
        close = _make_series(np.linspace(100, 110, 200))
        osc, sig = m.prime_oscillator(close, length=14, signal_length=3)
        assert osc.name.startswith("PRIME_OSC")
        assert sig.name.startswith("PRIME_SIGNAL")


# ---------------------------------------------------------------------------
# Ehlers Cyber Cycle
# ---------------------------------------------------------------------------


class TestEhlersCyberCycleDeepCoverage:
    def test_inner_loop_too_short(self) -> None:
        m = _import_original_module("ehlers_cyber_cycle")
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        result = m._njit_cyber_cycle_loop(prices, 14, 0.07)
        assert np.isnan(result).all()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("ehlers_cyber_cycle")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="length must be >= 2"):
            m.ehlers_cyber_cycle(close, length=1)
        with pytest.raises(ValueError, match="alpha must be between"):
            m.ehlers_cyber_cycle(close, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be between"):
            m.ehlers_cyber_cycle(close, alpha=1.0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("ehlers_cyber_cycle")
        close = _make_series([100.0, 101.0, 102.0])
        result = m.ehlers_cyber_cycle(close)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("ehlers_cyber_cycle")
        close = _make_series(np.linspace(100, 110, 200))
        result = m.ehlers_cyber_cycle(close, length=14, alpha=0.07)
        assert result.name == "CC_14"


# ---------------------------------------------------------------------------
# Ehlers Instantaneous Trendline
# ---------------------------------------------------------------------------


class TestEhlersInstantaneousTrendlineDeepCoverage:
    def test_inner_loop_too_short(self) -> None:
        m = _import_original_module("ehlers_instantaneous_trendline")
        prices = np.array([100.0, 101.0, 102.0])
        result = m._njit_instantaneous_trendline_loop(prices, 0.07)
        assert np.isnan(result).all()

    def test_inner_loop_basic(self) -> None:
        m = _import_original_module("ehlers_instantaneous_trendline")
        prices = np.linspace(100, 110, 50)
        result = m._njit_instantaneous_trendline_loop(prices, 0.07)
        # From index 4 onward, values should be finite
        assert np.isfinite(result[4:]).all()

    def test_outer_function_invalid_alpha(self) -> None:
        m = _import_original_module("ehlers_instantaneous_trendline")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="alpha must be between"):
            m.ehlers_instantaneous_trendline(close, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be between"):
            m.ehlers_instantaneous_trendline(close, alpha=1.0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("ehlers_instantaneous_trendline")
        close = _make_series([100.0, 101.0, 102.0])
        result = m.ehlers_instantaneous_trendline(close)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("ehlers_instantaneous_trendline")
        close = _make_series(np.linspace(100, 110, 200))
        result = m.ehlers_instantaneous_trendline(close, alpha=0.07)
        assert result.name == "IT_0.07"


# ---------------------------------------------------------------------------
# Fibonacci Cycle
# ---------------------------------------------------------------------------


class TestFibonacciCycleDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("fibonacci_cycle")
        prices = np.linspace(100, 110, 100)
        cycle_periods = np.array([8, 13, 21, 34, 55], dtype=np.int64)
        fib_ratios = np.array([0.618, 1.0, 1.618, 2.618])
        result = m._njit_fibonacci_cycle_loop(
            prices, cycle_periods, fib_ratios, max_period=55
        )
        assert result.shape == (100,)

    def test_outer_function_invalid_lists(self) -> None:
        m = _import_original_module("fibonacci_cycle")
        close = _make_series(np.linspace(100, 110, 200))
        with pytest.raises(ValueError, match="cycle_periods must not be empty"):
            m.fibonacci_cycle(close, cycle_periods=[])
        with pytest.raises(ValueError, match="fib_ratios must not be empty"):
            m.fibonacci_cycle(close, fib_ratios=[])

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("fibonacci_cycle")
        close = _make_series([100.0, 101.0, 102.0])
        cycle, sig = m.fibonacci_cycle(close)
        assert isinstance(cycle, pd.Series)
        assert isinstance(sig, pd.Series)
        assert cycle.isna().all()
        assert sig.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("fibonacci_cycle")
        close = _make_series(np.linspace(100, 110, 200))
        cycle, sig = m.fibonacci_cycle(close)
        assert cycle.name.startswith("FIBO_CYCLE")
        assert sig.name.startswith("FIBO_SIGNAL")


# ---------------------------------------------------------------------------
# Entropy Volatility Index
# ---------------------------------------------------------------------------


class TestEntropyVolatilityIndexDeepCoverage:
    def test_inner_loop_constant_input(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        # Constant returns: std is 0 => EVI = 0
        returns = np.zeros(50)
        result = m._njit_entropy_volatility_loop(returns, 30, 2, 0.2)
        assert result.shape == (50,)
        valid = result[~np.isnan(result)]
        # For std=0, output is 0
        assert (valid == 0.0).all()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        close = _make_series(np.linspace(100, 110, 100))
        with pytest.raises(ValueError, match="length must be >= 1"):
            m.entropy_volatility_index(close, length=0)
        with pytest.raises(ValueError, match="m_val must be >= 1"):
            m.entropy_volatility_index(close, m_val=0)
        with pytest.raises(ValueError, match="r_val must be > 0"):
            m.entropy_volatility_index(close, r_val=0.0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        close = _make_series([100.0, 101.0])
        result = m.entropy_volatility_index(close)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("entropy_volatility_index")
        close = _make_series(np.linspace(100, 110, 100))
        result = m.entropy_volatility_index(close, length=30, m_val=2, r_val=0.2)
        assert result.name.startswith("EVI_")


# ---------------------------------------------------------------------------
# FRAMA
# ---------------------------------------------------------------------------


class TestFRAMADeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("frama")
        prices = np.linspace(100, 110, 100)
        result = m._njit_frama_loop(
            prices,
            length=16,
            half=8,
            log2=np.log(2.0),
            w=0.5,
            alpha_min=0.01,
            alpha_max=1.0,
        )
        # After length-1, values should be finite
        assert np.isfinite(result[16:]).all()

    def test_outer_function_auto_clamps_length(self) -> None:
        m = _import_original_module("frama")
        close = _make_series(np.linspace(100, 110, 50))
        # length < 4 is clamped to 4
        result = m.frama(close, length=2)
        assert isinstance(result, pd.Series)
        # length is odd => it gets rounded up
        result_odd = m.frama(close, length=5)
        assert isinstance(result_odd, pd.Series)
        # slow < 1 is clamped to 1
        result_slow = m.frama(close, length=10, slow=0)
        assert isinstance(result_slow, pd.Series)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("frama")
        close = _make_series([100.0, 101.0, 102.0])
        result = m.frama(close, length=16)
        assert isinstance(result, pd.Series)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Market Meanness Index
# ---------------------------------------------------------------------------


class TestMMIDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("market_meanness_index")
        # Constant price => MMI = 0
        prices = np.full(30, 100.0)
        result = m._njit_mmi_loop(prices, 20)
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()

    def test_inner_loop_alternating(self) -> None:
        m = _import_original_module("market_meanness_index")
        # Strictly alternating up/down: every pair is meanness.
        prices = np.array([100, 101, 100, 101, 100, 101, 100, 101, 100, 101] * 5)
        result = m._njit_mmi_loop(prices, 5)
        # All flips => MMI = 100
        valid = result[~np.isnan(result)]
        assert (valid == 100.0).all()

    def test_outer_function_invalid_length(self) -> None:
        m = _import_original_module("market_meanness_index")
        close = _make_series(np.linspace(100, 110, 50))
        with pytest.raises(ValueError, match="length must be >= 3"):
            m.mmi(close, length=2)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("market_meanness_index")
        close = _make_series([100.0, 101.0])
        result = m.mmi(close, length=20)
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("market_meanness_index")
        close = _make_series(np.linspace(100, 110, 50))
        result = m.mmi(close, length=20)
        assert result.name == "MMI_20"


# ---------------------------------------------------------------------------
# Damiani Volatmeter
# ---------------------------------------------------------------------------


class TestDamianiVolatmeterDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("damiani_volatmeter")
        rng = np.random.default_rng(0)
        n = 150
        high = np.cumsum(rng.normal(0, 1, n)) + 102
        low = np.cumsum(rng.normal(0, 1, n)) + 98
        close = (high + low) / 2.0
        result = m._njit_damiani_volatmeter_loop(high, low, close, 13, 20, 40, 100)
        assert result.shape == (n,)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("damiani_volatmeter")
        df = _make_ohlcv(10)
        osc, thr = m.damiani_volatmeter(df["high"], df["low"], df["close"])
        assert isinstance(osc, pd.Series)
        assert isinstance(thr, pd.Series)
        assert osc.isna().all()
        # Threshold line is always the constant value
        assert (thr == 1.4).all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("damiani_volatmeter")
        df = _make_ohlcv(150)
        osc, thr = m.damiani_volatmeter(df["high"], df["low"], df["close"])
        assert osc.name.startswith("DAMIANI_")
        assert thr.name.startswith("DAMIANI_THR_")


# ---------------------------------------------------------------------------
# Adaptive Entropy
# ---------------------------------------------------------------------------


class TestAdaptiveEntropyDeepCoverage:
    def test_inner_loop_constant_input(self) -> None:
        m = _import_original_module("adaptive_entropy")
        data = np.full(50, 100.0)
        result = m._njit_entropy_loop(data, 14)
        # Constant data has 0 entropy
        valid = result[~np.isnan(result)]
        assert (valid == 0.0).all()

    def test_outer_function_invalid_params(self) -> None:
        m = _import_original_module("adaptive_entropy")
        close = _make_series(np.linspace(100, 110, 100))
        with pytest.raises(ValueError, match="short_length must be >= 5"):
            m.adaptive_entropy(close, short_length=3, long_length=28)
        with pytest.raises(ValueError, match="long_length must be >= 10"):
            m.adaptive_entropy(close, short_length=14, long_length=5)
        with pytest.raises(ValueError, match="signal_length must be >= 2"):
            m.adaptive_entropy(close, short_length=14, long_length=28, signal_length=1)
        with pytest.raises(ValueError, match="short_length must be < long_length"):
            m.adaptive_entropy(close, short_length=30, long_length=28, signal_length=3)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("adaptive_entropy")
        close = _make_series([100.0, 101.0, 102.0, 103.0, 104.0])
        osc, sig, ratio = m.adaptive_entropy(close)
        assert osc.isna().all()
        assert sig.isna().all()
        assert ratio.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("adaptive_entropy")
        close = _make_series(np.linspace(100, 110, 100))
        osc, sig, ratio = m.adaptive_entropy(
            close, short_length=14, long_length=28, signal_length=5
        )
        assert osc.name.startswith("ADAPTIVE_ENTROPY_OSC")
        assert sig.name.startswith("ADAPTIVE_ENTROPY_SIGNAL")
        assert ratio.name.startswith("ADAPTIVE_ENTROPY_RATIO")


# ---------------------------------------------------------------------------
# Kairi Relative Index
# ---------------------------------------------------------------------------


class TestKairiRelativeIndexDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("kairi_relative_index")
        prices = np.array([100.0] * 5 + [110.0, 120.0, 130.0, 140.0, 150.0])
        result = m._njit_kairi_loop(prices, 5)
        # At index 9, the SMA is ~120 and the price is 150
        # KRI = (150 - 120) / 120 * 100 = 25
        last_valid = result[~np.isnan(result)][-1]
        assert last_valid > 0.0

    def test_inner_loop_too_short(self) -> None:
        m = _import_original_module("kairi_relative_index")
        prices = np.array([100.0, 101.0, 102.0])
        result = m._njit_kairi_loop(prices, 10)
        assert np.isnan(result).all()

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("kairi_relative_index")
        close = _make_series([100.0, 101.0, 102.0])
        osc, sig = m.kairi_relative_index(close, length=14)
        assert osc.isna().all()
        assert sig.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("kairi_relative_index")
        close = _make_series(np.linspace(100, 110, 100))
        osc, sig = m.kairi_relative_index(close, length=14, signal_length=3)
        assert osc.name == "KRI_14"
        assert sig.name == "KRI_SIGNAL_14"


# ---------------------------------------------------------------------------
# DeMarker
# ---------------------------------------------------------------------------


class TestDeMarkerDeepCoverage:
    def test_inner_loop_directly(self) -> None:
        m = _import_original_module("demarker")
        high = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 5)
        low = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108] * 5)
        result = m._njit_demarker_loop(high, low, 14)
        # The DeMarker is in [0, 100]
        valid = result[~np.isnan(result)]
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_inner_loop_short_input(self) -> None:
        m = _import_original_module("demarker")
        # n < length + 1
        high = np.array([101.0, 102.0])
        low = np.array([99.0, 100.0])
        result = m._njit_demarker_loop(high, low, 14)
        assert np.isnan(result).all()

    def test_outer_function_invalid_length(self) -> None:
        m = _import_original_module("demarker")
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match="length must be >= 1"):
            m.demarker(df["high"], df["low"], length=0)

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("demarker")
        df = _make_ohlcv(5)
        result = m.demarker(df["high"], df["low"])
        assert isinstance(result, pd.Series)
        assert result.isna().all()

    def test_outer_function_named_output(self) -> None:
        m = _import_original_module("demarker")
        df = _make_ohlcv(100)
        result = m.demarker(df["high"], df["low"], length=14)
        assert result.name == "DEMARKER_14"


# ---------------------------------------------------------------------------
# Quantum Flow
# ---------------------------------------------------------------------------


class TestQuantumFlowDeepCoverage:
    def test_simple_wavelet_transform_short(self) -> None:
        m = _import_original_module("quantum_flow")
        data = np.array([1.0, 2.0, 3.0])
        result = m._simple_wavelet_transform(data, scale=4)
        # n < scale => all NaN
        assert np.isnan(result).all()

    def test_simple_wavelet_transform_min_scale(self) -> None:
        m = _import_original_module("quantum_flow")
        data = np.arange(20, dtype=float)
        # scale < 2 => all NaN
        result = m._simple_wavelet_transform(data, scale=1)
        assert np.isnan(result).all()

    def test_simple_wavelet_transform_normal(self) -> None:
        m = _import_original_module("quantum_flow")
        data = np.arange(30, dtype=float)
        result = m._simple_wavelet_transform(data, scale=4)
        # Indices < scale - 1 = 3 are NaN, rest finite
        assert np.isnan(result[:3]).all()
        assert np.isfinite(result[3:]).all()

    def test_outer_function_insufficient_data(self) -> None:
        m = _import_original_module("quantum_flow")
        df = _make_ohlcv(3)
        flow, sig = m.quantum_flow(df["close"], df["high"], df["low"], df["volume"])
        assert isinstance(flow, pd.Series)
        assert isinstance(sig, pd.Series)
        assert flow.isna().all()
        assert sig.isna().all()

    def test_outer_function_returns_named(self) -> None:
        m = _import_original_module("quantum_flow")
        df = _make_ohlcv(100)
        flow, sig = m.quantum_flow(df["close"], df["high"], df["low"], df["volume"])
        assert flow.name == "QUANTUM_FLOW"
        assert sig.name == "QUANTUM_FLOW_SIGNAL"


# ---------------------------------------------------------------------------
# Window Helpers
# ---------------------------------------------------------------------------


class TestWindowHelpersDeepCoverage:
    def test_window_sum(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert m._window_sum(arr, 0, 5) == 15.0
        assert m._window_sum(arr, 1, 4) == 9.0
        assert m._window_sum(arr, 0, 0) == 0.0

    def test_window_mean(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([2.0, 4.0, 6.0, 8.0])
        assert m._window_mean(arr, 0, 4) == 5.0
        # Zero-length window
        assert m._window_mean(arr, 2, 2) == 0.0

    def test_window_mean_and_std(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = m._window_mean_and_std(arr, 0, 5)
        assert mean == 3.0
        assert std > 1.4
        # Zero-length window
        mean0, std0 = m._window_mean_and_std(arr, 2, 2)
        assert mean0 == 0.0
        assert std0 == 0.0

    def test_window_mean_and_std_finite(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        mean, std, count = m._window_mean_and_std_finite(arr, 0, 5)
        assert count == 4
        assert mean == pytest.approx(3.25, rel=1e-6)
        # No finite values
        arr_nan = np.array([np.nan, np.nan, np.nan])
        mean, std, count = m._window_mean_and_std_finite(arr_nan, 0, 3)
        assert count == 0
        assert mean == 0.0
        assert std == 0.0

    def test_window_min_max(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        min_val, max_val = m._window_min_max(arr, 0, 8)
        assert min_val == 1.0
        assert max_val == 9.0

    def test_window_range(self) -> None:
        m = _import_original_module("_window_helpers")
        arr = np.array([1.0, 3.0, 5.0, 2.0])
        # range / scale
        assert m._window_range(arr, 0, 4, scale=2.0) == 2.0  # (5-1)/2
        assert m._window_range(arr, 0, 4, scale=1.0) == 4.0
