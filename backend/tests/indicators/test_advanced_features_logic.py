import pytest
import pandas as pd
import numpy as np
from app.services.indicators.technical_indicators.advanced_features import AdvancedFeatures
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators

@pytest.fixture
def sample_data():
    # Create synthetic data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
    close = pd.Series(np.linspace(100, 110, 100) + np.random.normal(0, 1, 100), index=dates)
    high = close + 1
    low = close - 1
    open_ = close.shift(1).fillna(100)
    volume = pd.Series(np.random.randint(100, 1000, 100), index=dates)
    open_interest = pd.Series(np.linspace(1000, 2000, 100) + np.random.normal(0, 10, 100), index=dates)
    funding_rate = pd.Series(np.random.normal(0.0001, 0.00005, 100), index=dates)
    
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "open_interest": open_interest,
        "funding_rate": funding_rate
    }

def test_frac_diff_ffd(sample_data):
    close = sample_data["close"]
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

def test_liquidation_cascade_score(sample_data):
    score = AdvancedFeatures.liquidation_cascade_score(
        sample_data["close"],
        sample_data["open_interest"],
        sample_data["volume"]
    )
    assert isinstance(score, pd.Series)
    assert len(score) == len(sample_data["close"])

def test_squeeze_probability(sample_data):
    prob = AdvancedFeatures.squeeze_probability(
        sample_data["close"],
        sample_data["funding_rate"],
        sample_data["open_interest"],
        sample_data["low"]
    )
    assert isinstance(prob, pd.Series)
    assert len(prob) == len(sample_data["close"])

def test_trend_quality(sample_data):
    tq = AdvancedFeatures.trend_quality(
        sample_data["close"],
        sample_data["open_interest"]
    )
    assert isinstance(tq, pd.Series)

def test_parkinson_volatility(sample_data):
    vol = VolatilityIndicators.parkinson(
        sample_data["high"],
        sample_data["low"]
    )
    assert isinstance(vol, pd.Series)
    assert not vol.isna().all()

def test_garman_klass_volatility(sample_data):
    vol = VolatilityIndicators.garman_klass(
        sample_data["open"],
        sample_data["high"],
        sample_data["low"],
        sample_data["close"]
    )
    assert isinstance(vol, pd.Series)
    assert not vol.isna().all()

def test_vwap_z_score(sample_data):
    z = VolumeIndicators.vwap_z_score(
        sample_data["high"],
        sample_data["low"],
        sample_data["close"],
        sample_data["volume"]
    )
    assert isinstance(z, pd.Series)

def test_rvol(sample_data):
    rvol = VolumeIndicators.rvol(sample_data["volume"])
    assert isinstance(rvol, pd.Series)
    # Should work with DatetimeIndex
    assert not rvol.isna().all()

def test_absorption_score(sample_data):
    score = VolumeIndicators.absorption_score(
        sample_data["high"],
        sample_data["low"],
        sample_data["volume"]
    )
    assert isinstance(score, pd.Series)

def test_sample_entropy(sample_data):
    entropy = AdvancedFeatures.sample_entropy(
        sample_data["close"],
        window=20,
        m=2,
        r=0.2
    )
    assert isinstance(entropy, pd.Series)
    assert len(entropy) == len(sample_data["close"])
    # 最初のwindow期間はNaN（または0）
    assert entropy.iloc[:19].sum() == 0

def test_fractal_dimension(sample_data):
    fd = AdvancedFeatures.fractal_dimension(
        sample_data["close"],
        window=20
    )
    assert isinstance(fd, pd.Series)
    assert len(fd) == len(sample_data["close"])
    # 1.0 to 2.0 (平面の埋め尽くし) の範囲にクリップされていることを確認
    valid_vals = fd.iloc[20:]
    assert (valid_vals >= 1.0).all(), f"FD below 1.0: {valid_vals.min()}"
    assert (valid_vals <= 2.0).all(), f"FD above 2.0: {valid_vals.max()}"


def test_vpin_approximation(sample_data):
    vpin = AdvancedFeatures.vpin_approximation(
        sample_data["close"],
        sample_data["volume"],
        window=20
    )
    assert isinstance(vpin, pd.Series)
    assert len(vpin) == len(sample_data["close"])
    # 0.0 to 1.0 の範囲
    assert (vpin >= 0.0).all()
    assert (vpin <= 1.0).all()




