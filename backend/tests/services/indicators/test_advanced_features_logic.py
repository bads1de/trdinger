import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.advanced_features import AdvancedFeatures
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators

def test_frac_diff_ffd(sample_df):
    close = sample_df["close"]
    # Apply log as recommended
    log_close = np.log(close)
    
    # Test with small window for speed
    diff_series = AdvancedFeatures.frac_diff_ffd(log_close, d=0.4, window=50)
    
    assert isinstance(diff_series, pd.Series)
    assert len(diff_series) == len(close)
    # Initial values should be NaN due to window
    # The implementation uses a rolling window, so first `width` elements are NaN
    # but my implementation fills with NaN.
    assert diff_series.isna().sum() > 0

def test_liquidation_cascade_score(sample_df):
    score = AdvancedFeatures.liquidation_cascade_score(
        sample_df["close"],
        sample_df["open_interest"],
        sample_df["volume"]
    )
    assert isinstance(score, pd.Series)
    assert len(score) == len(sample_df["close"])

def test_squeeze_probability(sample_df):
    prob = AdvancedFeatures.squeeze_probability(
        sample_df["close"],
        sample_df["funding_rate"],
        sample_df["open_interest"],
        sample_df["low"]
    )
    assert isinstance(prob, pd.Series)
    assert len(prob) == len(sample_df["close"])

def test_trend_quality(sample_df):
    tq = AdvancedFeatures.trend_quality(
        sample_df["close"],
        sample_df["open_interest"]
    )
    assert isinstance(tq, pd.Series)

def test_parkinson_volatility(sample_df):
    vol = VolatilityIndicators.parkinson(
        sample_df["high"],
        sample_df["low"]
    )
    assert isinstance(vol, pd.Series)
    assert not vol.isna().all()

def test_garman_klass_volatility(sample_df):
    vol = VolatilityIndicators.garman_klass(
        sample_df["open"],
        sample_df["high"],
        sample_df["low"],
        sample_df["close"]
    )
    assert isinstance(vol, pd.Series)
    assert not vol.isna().all()

def test_vwap_z_score(sample_df):
    z = VolumeIndicators.vwap_z_score(
        sample_df["high"],
        sample_df["low"],
        sample_df["close"],
        sample_df["volume"]
    )
    assert isinstance(z, pd.Series)

def test_rvol(sample_df):
    rvol = VolumeIndicators.rvol(sample_df["volume"])
    assert isinstance(rvol, pd.Series)
    # Should work with DatetimeIndex
    assert not rvol.isna().all()

def test_absorption_score(sample_df):
    score = VolumeIndicators.absorption_score(
        sample_df["high"],
        sample_df["low"],
        sample_df["volume"]
    )
    assert isinstance(score, pd.Series)

def test_sample_entropy(sample_df):
    entropy = AdvancedFeatures.sample_entropy(
        sample_df["close"],
        window=20,
        m=2,
        r=0.2
    )
    assert isinstance(entropy, pd.Series)
    assert len(entropy) == len(sample_df["close"])
    # 最初のwindow期間はNaN（または0）
    assert entropy.iloc[:19].sum() == 0

def test_fractal_dimension(sample_df):
    fd = AdvancedFeatures.fractal_dimension(
        sample_df["close"],
        window=20
    )
    assert isinstance(fd, pd.Series)
    assert len(fd) == len(sample_df["close"])
    # 1.0 to 2.0 (平面の埋め尽くし) の範囲にクリップされていることを確認
    valid_vals = fd.iloc[20:]
    assert (valid_vals >= 1.0).all(), f"FD below 1.0: {valid_vals.min()}"
    assert (valid_vals <= 2.0).all(), f"FD above 2.0: {valid_vals.max()}"


def test_vpin_approximation(sample_df):
    vpin = AdvancedFeatures.vpin_approximation(
        sample_df["close"],
        sample_df["volume"],
        window=20
    )
    assert isinstance(vpin, pd.Series)
    assert len(vpin) == len(sample_df["close"])
    # 0.0 to 1.0 の範囲
    assert (vpin >= 0.0).all()
    assert (vpin <= 1.0).all()


def test_regime_quadrant(sample_df):
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0], index=sample_df.index[:5])
    open_interest = pd.Series([10.0, 11.0, 10.0, 11.0, 10.0], index=sample_df.index[:5])

    regime = AdvancedFeatures.regime_quadrant(close, open_interest)

    assert isinstance(regime, pd.Series)
    assert regime.iloc[1:].tolist() == [0, 1, 2, 3]


def test_whale_divergence_fill_value():
    positions = pd.Series([2.0, 0.0])
    accounts = pd.Series([1.0, 0.0])

    divergence = AdvancedFeatures.whale_divergence(positions, accounts)

    assert isinstance(divergence, pd.Series)
    assert divergence.iloc[0] == 2.0
    assert divergence.iloc[1] == 1.0




