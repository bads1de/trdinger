import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.advanced_features import (
    AdvancedFeatures,
)
from app.services.indicators.technical_indicators.pandas_ta import VolumeIndicators


def test_liquidation_cascade_score(sample_df):
    score = AdvancedFeatures.liquidation_cascade_score(
        sample_df["close"], sample_df["open_interest"], sample_df["volume"]
    )
    assert isinstance(score, pd.Series)
    assert len(score) == len(sample_df["close"])


def test_squeeze_probability(sample_df):
    prob = AdvancedFeatures.squeeze_probability(
        sample_df["close"],
        sample_df["funding_rate"],
        sample_df["open_interest"],
        sample_df["low"],
    )
    assert isinstance(prob, pd.Series)
    assert len(prob) == len(sample_df["close"])


def test_trend_quality(sample_df):
    tq = AdvancedFeatures.trend_quality(sample_df["close"], sample_df["open_interest"])
    assert isinstance(tq, pd.Series)


def test_vwap_z_score(sample_df):
    z = VolumeIndicators.vwap_z_score(
        sample_df["high"], sample_df["low"], sample_df["close"], sample_df["volume"]
    )
    assert isinstance(z, pd.Series)


def test_rvol(sample_df):
    rvol = VolumeIndicators.rvol(sample_df["volume"])
    assert isinstance(rvol, pd.Series)
    # Should work with DatetimeIndex
    assert not rvol.isna().all()


def test_sample_entropy(sample_df):
    entropy = AdvancedFeatures.sample_entropy(sample_df["close"], window=20, m=2, r=0.2)
    assert isinstance(entropy, pd.Series)
    assert len(entropy) == len(sample_df["close"])
    # 最初のwindow期間はNaN（または0）
    assert entropy.iloc[:19].sum() == 0


def test_fractal_dimension(sample_df):
    fd = AdvancedFeatures.fractal_dimension(sample_df["close"], window=20)
    assert isinstance(fd, pd.Series)
    assert len(fd) == len(sample_df["close"])
    # 1.0 to 2.0 (平面の埋め尽くし) の範囲にクリップされていることを確認
    valid_vals = fd.iloc[20:]
    assert (valid_vals >= 1.0).all(), f"FD below 1.0: {valid_vals.min()}"
    assert (valid_vals <= 2.0).all(), f"FD above 2.0: {valid_vals.max()}"


def test_vpin_approximation(sample_df):
    vpin = AdvancedFeatures.vpin_approximation(
        sample_df["close"], sample_df["volume"], window=20
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


def test_long_short_ratio_zscore():
    # ノイズ入りの変動する比率系列を入力
    rng = np.random.default_rng(7)
    values = 1.0 + rng.normal(0, 0.05, 100)
    series = pd.Series(values, dtype=float)
    lsr = AdvancedFeatures.long_short_ratio_zscore(series, window=20)
    assert isinstance(lsr, pd.Series)
    assert len(lsr) == len(series)
    # 最初のwindow期間は0（NaNを0埋め）
    assert lsr.iloc[:19].sum() == 0
    # 一定値の入力ではz-scoreが0になる
    constant = pd.Series([1.0] * 50)
    flat = AdvancedFeatures.long_short_ratio_zscore(constant, window=20)
    assert (flat.iloc[19:] == 0).all()


def test_long_short_ratio_zscore_detects_deviation():
    # 前半1.0（均衡）、後半1.5（ロング過熱）でz-scoreが正になる
    # 窓内stdが0になるのを避けるため後半はわずかなノイズを加える
    rng = np.random.default_rng(42)
    values = [1.0] * 30 + (1.5 + rng.normal(0, 0.01, 30)).tolist()
    series = pd.Series(values, dtype=float)
    z = AdvancedFeatures.long_short_ratio_zscore(series, window=20)
    assert isinstance(z, pd.Series)
    assert z.iloc[-1] > 0


def test_oi_roc_returns_rate_of_change():
    """OIの変化率が期間差分で正しく算出されること"""
    values = [100.0] * 10 + [110.0] * 10
    series = pd.Series(values, dtype=float)
    roc = AdvancedFeatures.oi_roc(series, period=10)
    assert isinstance(roc, pd.Series)
    assert len(roc) == len(series)
    # 10バー後の増加率は+10%
    assert roc.iloc[10] == pytest.approx(0.10, abs=1e-6)


def test_funding_rate_level_returns_raw_value():
    """FRの水準がそのまま返ること"""
    series = pd.Series([0.0001, -0.0002, 0.0], dtype=float)
    level = AdvancedFeatures.funding_rate_level(series)
    assert isinstance(level, pd.Series)
    assert level.iloc[0] == pytest.approx(0.0001, abs=1e-9)
    assert level.iloc[1] == pytest.approx(-0.0002, abs=1e-9)


def test_long_short_ratio_level_returns_raw_value():
    """LSRの水準がそのまま返ること（欠損は1.0で埋める）"""
    series = pd.Series([1.1, float("nan"), 0.9], dtype=float)
    level = AdvancedFeatures.long_short_ratio_level(series)
    assert isinstance(level, pd.Series)
    assert level.iloc[0] == pytest.approx(1.1, abs=1e-6)
    assert level.iloc[2] == pytest.approx(0.9, abs=1e-6)


def test_lsr_roc_returns_rate_of_change():
    """LSRの変化率が期間差分で正しく算出されること"""
    values = [1.0] * 10 + [1.2] * 10
    series = pd.Series(values, dtype=float)
    roc = AdvancedFeatures.lsr_roc(series, period=10)
    assert isinstance(roc, pd.Series)
    assert len(roc) == len(series)
    # 10バー後の増加率は+20%
    assert roc.iloc[10] == pytest.approx(0.20, abs=1e-6)
