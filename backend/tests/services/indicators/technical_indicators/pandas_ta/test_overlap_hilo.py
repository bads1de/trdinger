"""
OverlapIndicators.hilo (Gann HiLo) のテスト

pandas_ta の hilo は、close が high/low MA のチャネル内に留まると初期 NaN を
引き継ぎ続け、データによっては全期間 NaN を返す。
本テストは、その NaN 伝播を防ぐフォールバックと、正常時の pandas_ta との
一致性を検証する。
"""

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest

from app.services.indicators.technical_indicators.pandas_ta.overlap import (
    OverlapIndicators,
)


def _channel_bound_data(length: int = 300, seed: int = 42) -> pd.DataFrame:
    """close が high/low MA のチャネル内に留まり続けるデータを生成"""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=length, freq="h")
    close = pd.Series(
        np.linspace(10000.0, 10050.0, length) + rng.normal(0.0, 1.0, length),
        index=index,
    )
    return pd.DataFrame(
        {
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        }
    )


def _breakout_data(length: int = 500, seed: int = 0) -> pd.DataFrame:
    """ブレイクアウトが発生するボラタイルなデータを生成"""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=length, freq="h")
    close = pd.Series(
        np.linspace(10000.0, 15000.0, length) + rng.normal(0.0, 200.0, length),
        index=index,
    )
    return pd.DataFrame(
        {
            "High": close + rng.uniform(100.0, 500.0, length),
            "Low": close - rng.uniform(100.0, 500.0, length),
            "Close": close,
        }
    )


class TestHiloIndicator:
    """Gann HiLo の回帰テスト"""

    def test_hilo_channel_bound_data_returns_finite(self):
        """チャネル内データ（pandas_ta が全 NaN になるケース）でも有限値を返す"""
        df = _channel_bound_data()

        # 前提確認: pandas_ta 単体では全 NaN になる
        raw = ta.hilo(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            high_length=13,
            low_length=13,
            mamode="sma",
        )
        assert raw is not None
        assert raw.isna().all().all()

        hilo, hilo_long, hilo_short = OverlapIndicators.hilo(
            high=df["High"], low=df["Low"], close=df["Close"]
        )

        assert len(hilo) == len(df)
        assert np.isfinite(hilo.to_numpy()).sum() > 0
        assert np.isfinite(hilo_long.to_numpy()).sum() > 0
        assert np.isfinite(hilo_short.to_numpy()).sum() > 0
        assert hilo.index.equals(df.index)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_hilo_matches_pandas_ta_when_breakout_occurs(self, seed: int):
        """ブレイクアウト発生時は pandas_ta と完全に一致する"""
        df = _breakout_data(seed=seed)

        result = OverlapIndicators.hilo(
            high=df["High"], low=df["Low"], close=df["Close"]
        )

        raw = ta.hilo(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            high_length=13,
            low_length=13,
            mamode="sma",
        )
        assert raw is not None
        expected = (
            raw["HILO_13_13"],
            raw["HILOl_13_13"],
            raw["HILOs_13_13"],
        )

        for actual, expect in zip(result, expected, strict=True):
            pd.testing.assert_series_equal(
                actual,
                expect,
                check_names=False,
                check_index_type=False,
            )

    def test_hilo_hold_state_keeps_previous_value(self):
        """ホールド状態では前回値を引き継ぐ（NaN 伝播は起きない）"""
        df = _channel_bound_data(length=200, seed=7)

        hilo, _hilo_long, _hilo_short = OverlapIndicators.hilo(
            high=df["High"], low=df["Low"], close=df["Close"]
        )

        hilo_values = hilo.to_numpy(dtype=float)
        finite = hilo_values[np.isfinite(hilo_values)]
        # 初回有限値以降は途中で NaN が出てはいけない
        first_finite = np.argmax(np.isfinite(hilo_values))
        assert np.isfinite(hilo_values[first_finite:]).all()
        assert len(finite) > 0

    def test_hilo_offset_shifts_result(self):
        """offset 指定時は結果がシフトされる"""
        df = _channel_bound_data(length=200, seed=8)

        hilo, _hilo_long, _hilo_short = OverlapIndicators.hilo(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            offset=1,
        )

        unshifted, _long, _short = OverlapIndicators.hilo(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
        )
        pd.testing.assert_series_equal(
            hilo, unshifted.shift(1), check_names=False, check_index_type=False
        )

    def test_hilo_return_shape(self):
        """タプルで3系列を返し、長さとインデックスが入力と一致する"""
        df = _breakout_data(seed=3)
        result = OverlapIndicators.hilo(
            high=df["High"], low=df["Low"], close=df["Close"]
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        for series in result:
            assert isinstance(series, pd.Series)
            assert len(series) == len(df)
            assert series.index.equals(df.index)
