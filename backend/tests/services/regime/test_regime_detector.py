"""レジーム検出器のテスト."""

import pandas as pd
import pytest

from app.services.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    RegimeDetector,
    classify_return,
    forward_return_pp,
)


def _daily_closes(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class TestClassifyReturn:
    def test_boundaries(self):
        assert classify_return(20.1) == REGIME_BULL
        assert classify_return(-20.1) == REGIME_BEAR
        assert classify_return(20.0) == REGIME_SIDEWAYS
        assert classify_return(-20.0) == REGIME_SIDEWAYS
        assert classify_return(0.0) == REGIME_SIDEWAYS


class TestRegimeDetector:
    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError):
            RegimeDetector(lookback_days=1)

    def test_requires_datetime_index(self):
        detector = RegimeDetector(lookback_days=5)
        with pytest.raises(TypeError):
            detector.detect_series(pd.Series([1.0, 2.0]))

    def test_flat_series_is_sideways(self):
        closes = _daily_closes([100.0] * 60)
        labels = RegimeDetector(lookback_days=30).detect_series(closes)
        known = labels.dropna()
        assert len(known) > 0
        assert set(known) == {REGIME_SIDEWAYS}

    def test_rally_detected_as_bull(self):
        # 30日で +50% 上昇する系列
        values = [100.0 * (1.01**i) for i in range(60)]
        labels = RegimeDetector(lookback_days=30).detect_series(_daily_closes(values))
        assert labels.dropna().iloc[-1] == REGIME_BULL

    def test_crash_detected_as_bear(self):
        values = [100.0 * (0.98**i) for i in range(60)]
        labels = RegimeDetector(lookback_days=30).detect_series(_daily_closes(values))
        assert labels.dropna().iloc[-1] == REGIME_BEAR

    def test_warmup_region_is_na(self):
        closes = _daily_closes([100.0] * 100)
        detector = RegimeDetector(lookback_days=30)
        labels = detector.detect_series(closes)
        # 最初の min_history_ratio 分は未判定
        assert pd.isna(labels.iloc[0])
        assert pd.isna(labels.iloc[10])
        assert not pd.isna(labels.iloc[-1])

    def test_walk_forward_no_lookahead(self):
        """label_at(t) は t 以降のデータを参照しない。"""
        values = [100.0 * (1.005**i) for i in range(120)]
        closes = _daily_closes(values)
        detector = RegimeDetector(lookback_days=60)
        full_labels = detector.detect_series(closes)
        truncated_labels = detector.detect_series(closes.iloc[:80])
        asof = closes.index[79]
        assert detector.label_at(closes, asof) == str(full_labels.loc[asof])
        assert str(truncated_labels.loc[asof]) == str(full_labels.loc[asof])

    def test_monthly_labels_use_only_prior_data(self):
        """月初ラベルは前日以前のみで計算される（当月データ変更の影響を受けない）。"""
        values = [100.0] * 90
        closes = _daily_closes(values, start="2024-01-01")
        detector = RegimeDetector(lookback_days=60)
        ms = pd.Timestamp("2024-03-01")
        base = detector.monthly_labels(closes, [ms])[ms]
        # 当月以降を急騰させても月初ラベルは不変
        tampered = closes.copy()
        tampered.loc[tampered.index >= ms] = 200.0
        after = detector.monthly_labels(tampered, [ms])[ms]
        assert base == after

    def test_label_at_insufficient_history_returns_none(self):
        closes = _daily_closes([100.0] * 5)
        assert (
            RegimeDetector(lookback_days=30).label_at(closes, closes.index[-1]) is None
        )


class TestForwardReturn:
    def test_simple_range(self):
        closes = _daily_closes([100.0, 110.0, 120.0])
        ret = forward_return_pp(
            closes, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")
        )
        assert ret == pytest.approx(20.0)

    def test_missing_data_returns_none(self):
        closes = _daily_closes([100.0])
        assert (
            forward_return_pp(
                closes, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")
            )
            is None
        )
