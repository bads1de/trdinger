"""Unified low-level indicator behavior tests."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.overlap import OverlapIndicators
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import (
    VolatilityIndicators,
)


class TestOverlapIndicatorsLogic:
    @pytest.mark.parametrize("method_name", ["sma", "ema"])
    def test_simple_moving_averages_stay_in_range(self, sample_df, method_name):
        close = sample_df["close"]
        result = getattr(OverlapIndicators, method_name)(close, length=10)

        assert isinstance(result, pd.Series)
        valid = result.dropna()
        if not valid.empty:
            assert valid.min() >= close.min()
            assert valid.max() <= close.max()

    def test_wma_validation(self, sample_df):
        close = sample_df["close"]

        assert isinstance(
            OverlapIndicators.wma(data=None, close=close, length=20), pd.Series
        )
        with pytest.raises(ValueError):
            OverlapIndicators.wma(None, None)

    def test_complex_moving_averages(self, sample_df):
        close = sample_df["close"]
        volume = sample_df["volume"]

        assert isinstance(OverlapIndicators.zlma(close, 10), pd.Series)
        with patch("pandas_ta_classic.zlma", return_value=None):
            fallback = OverlapIndicators.zlma(close, 10)
            assert np.isnan(fallback).all()

        assert isinstance(OverlapIndicators.alma(close), pd.Series)
        with pytest.raises(ValueError):
            OverlapIndicators.alma(close, sigma=0)

        assert isinstance(OverlapIndicators.vwma(close, volume), pd.Series)

    @pytest.mark.parametrize("method_name", ["linreg", "linregslope"])
    def test_linear_regression_helpers(self, sample_df, method_name):
        result = getattr(OverlapIndicators, method_name)(sample_df["close"])
        assert isinstance(result, pd.Series)

    def test_supertrend_shape(self, sample_df):
        trend, up, down = OverlapIndicators.supertrend(
            sample_df["high"], sample_df["low"], sample_df["close"], period=7, multiplier=3.0
        )

        assert all(isinstance(series, pd.Series) for series in (trend, up, down))
        assert len(trend) == len(up) == len(down) == len(sample_df)


class TestTrendIndicatorsLogic:
    def test_sar_calculation(self, sample_df):
        result = TrendIndicators.sar(sample_df["high"], sample_df["low"])

        assert isinstance(result, pd.Series)
        valid = result.dropna()
        if not valid.empty:
            assert valid.min() >= sample_df["low"].min()
            assert valid.max() <= sample_df["high"].max()

    def test_amat_and_dpo(self, sample_df):
        close = sample_df["close"]

        assert isinstance(TrendIndicators.amat(close), pd.Series)
        assert isinstance(TrendIndicators.dpo(close), pd.Series)

    def test_adx_and_vortex(self, sample_df):
        high = sample_df["high"]
        low = sample_df["low"]
        close = sample_df["close"]

        adx, dmp, dmn = TrendIndicators.adx(high, low, close, length=14)
        assert all(isinstance(series, pd.Series) for series in (adx, dmp, dmn))
        assert len(adx) == len(dmp) == len(dmn) == len(sample_df)
        assert adx.dropna().between(0, 100).all()

        with pytest.raises(ValueError, match="positive"):
            TrendIndicators.vortex(high, low, close, length=14, drift=0)

        vortex = TrendIndicators.vortex(high, low, close)
        assert isinstance(vortex, tuple)
        assert len(vortex) == 2
        assert all(isinstance(series, pd.Series) for series in vortex)


class TestMomentumIndicatorsLogic:
    def test_rsi_properties(self, sample_df):
        result = MomentumIndicators.rsi(sample_df["close"], period=14)

        assert isinstance(result, pd.Series)
        assert result.dropna().between(0, 100).all()
        assert result.iloc[:13].isna().all()

    def test_stoch_outputs(self, sample_df):
        k, d = MomentumIndicators.stoch(
            sample_df["high"], sample_df["low"], sample_df["close"], k=14, d=3
        )

        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert k.dropna().between(0, 100).all()
        assert d.dropna().between(0, 100).all()

    def test_bias_returns_series(self, sample_df):
        result = MomentumIndicators.bias(sample_df["close"])

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)

    def test_error_on_invalid_params(self, sample_df):
        with pytest.raises(ValueError):
            MomentumIndicators.rsi(sample_df["close"], period=-5)


class TestVolatilityIndicatorsLogic:
    def test_atr_calculation(self, sample_df):
        result = VolatilityIndicators.atr(
            sample_df["high"], sample_df["low"], sample_df["close"], length=14
        )

        assert isinstance(result, pd.Series)
        assert result.dropna().gt(0).all()
        assert result.iloc[:13].isna().all()

    def test_bbands_logic(self, sample_df):
        upper, middle, lower = VolatilityIndicators.bbands(
            sample_df["close"], length=20, std=2.0
        )
        valid = ~(upper.isna() | middle.isna() | lower.isna())

        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="All series must have the same length"):
            VolatilityIndicators.atr(
                pd.Series([1, 2]),
                pd.Series([1, 2, 3]),
                pd.Series([1, 2]),
            )
