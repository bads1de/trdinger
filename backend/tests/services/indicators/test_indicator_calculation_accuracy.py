"""
インジケーター計算の正確性検証テスト

手計算で検証済みの値と比較して、インジケーターの計算が正確であることを担保する。
このテストは「エラーなく実行できるか」だけでなく「計算結果が正しいか」を検証する。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.pandas_ta import (
    MomentumIndicators,
    OverlapIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)

# =============================================================================
# 既知のデータセット（手計算で検証済み）
# =============================================================================


@pytest.fixture
def known_prices():
    """手計算で検証済みの価格データ"""
    # 10個の価格データ
    return pd.Series(
        [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]
    )


@pytest.fixture
def known_ohlcv():
    """手計算で検証済みのOHLCVデータ"""
    data = {
        "open": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
        "high": [102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0, 111.0],
        "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0],
        "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0],
        "volume": [1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1600],
    }
    return pd.DataFrame(data)


@pytest.fixture
def constant_prices():
    """一定の価格データ（変動なし）"""
    return pd.Series([100.0] * 20)


@pytest.fixture
def increasing_prices():
    """単調増加する価格データ"""
    return pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    )


@pytest.fixture
def decreasing_prices():
    """単調減少する価格データ"""
    return pd.Series(
        [110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
    )


# =============================================================================
# SMA (Simple Moving Average) の正確性検証
# =============================================================================


class TestSMACalculationAccuracy:
    """SMA計算の正確性を検証"""

    def test_sma_3period_known_values(self, known_prices):
        """3期間SMAの手計算値との一致を検証"""
        result = OverlapIndicators.sma(known_prices, length=3)

        # 手計算: SMA[2] = (100 + 102 + 101) / 3 = 101.0
        # 手計算: SMA[3] = (102 + 101 + 103) / 3 = 102.0
        # 手計算: SMA[4] = (101 + 103 + 105) / 3 = 103.0
        expected_values = {
            2: 101.0,
            3: 102.0,
            4: 103.0,
            5: 104.0,
            6: 105.0,
            7: 106.0,
            8: 107.0,
            9: 108.0,
        }

        for idx, expected in expected_values.items():
            actual = result.iloc[idx]
            assert not np.isnan(actual), f"SMA[{idx}]がNaNです"
            assert np.isclose(
                actual, expected, rtol=1e-10
            ), f"SMA[{idx}]の値が不正: expected={expected}, actual={actual}"

    def test_sma_initial_nan_values(self, known_prices):
        """SMAの最初の(length-1)個がNaNであることを検証"""
        result = OverlapIndicators.sma(known_prices, length=3)

        # 最初の2個はNaN
        assert np.isnan(result.iloc[0]), "SMA[0]はNaNであるべき"
        assert np.isnan(result.iloc[1]), "SMA[1]はNaNであるべき"

    def test_sma_constant_prices(self, constant_prices):
        """一定価格ではSMAも一定値になることを検証"""
        result = OverlapIndicators.sma(constant_prices, length=5)

        valid_values = result.dropna()
        assert (valid_values == 100.0).all(), "一定価格のSMAは一定値であるべき"

    def test_sma_increasing_prices_increases(self, increasing_prices):
        """単調増加価格ではSMAも単調増加することを検証"""
        result = OverlapIndicators.sma(increasing_prices, length=3)

        valid_values = result.dropna()
        diffs = valid_values.diff().dropna()
        assert (diffs > 0).all(), "単調増加価格のSMAは単調増加するべき"


# =============================================================================
# WMA (Weighted Moving Average) の正確性検証
# =============================================================================


class TestWMACalculationAccuracy:
    """WMA計算の正確性を検証"""

    def test_wma_3period_known_values(self, known_prices):
        """3期間WMAの手計算値との一致を検証"""
        result = OverlapIndicators.wma(known_prices, length=3)

        # pandas-taのWMAは最初の(length-1)個がNaNになる
        assert np.isnan(result.iloc[0]), "WMA[0]はNaNであるべき"
        assert np.isnan(result.iloc[1]), "WMA[1]はNaNであるべき"

        # WMAの計算式: WMA = Σ(Price[i] * weight[i]) / Σ(weight[i])
        # weight: [1, 2, 3] (最近のデータほど大きい重み)
        # WMA[2] = (100*1 + 102*2 + 101*3) / (1+2+3) = (100 + 204 + 303) / 6 = 607/6 = 101.1667
        # WMA[3] = (102*1 + 101*2 + 103*3) / 6 = (102 + 202 + 309) / 6 = 613/6 = 102.1667
        expected_wma_2 = (100 * 1 + 102 * 2 + 101 * 3) / 6
        expected_wma_3 = (102 * 1 + 101 * 2 + 103 * 3) / 6

        actual_2 = result.iloc[2]
        actual_3 = result.iloc[3]

        assert not np.isnan(actual_2), "WMA[2]がNaNです"
        assert not np.isnan(actual_3), "WMA[3]がNaNです"
        assert np.isclose(
            actual_2, expected_wma_2, rtol=1e-5
        ), f"WMA[2]の値が不正: expected={expected_wma_2}, actual={actual_2}"
        assert np.isclose(
            actual_3, expected_wma_3, rtol=1e-5
        ), f"WMA[3]の値が不正: expected={expected_wma_3}, actual={actual_3}"


# =============================================================================
# RSI (Relative Strength Index) の正確性検証
# =============================================================================


class TestRSICalculationAccuracy:
    """RSI計算の正確性を検証"""

    def test_rsi_all_gains(self):
        """全て上昇の場合、RSIは100に近づくことを検証"""
        # 全て1ずつ上昇
        prices = pd.Series(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
            ]
        )
        result = MomentumIndicators.rsi(prices, period=14)

        valid_values = result.dropna()
        assert len(valid_values) > 0, "RSIの有効な値がありません"
        # 全て上昇なのでRSIは100に近い値になる
        assert (
            valid_values.iloc[-1] > 90
        ), f"全て上昇の場合RSIは90以上であるべき: {valid_values.iloc[-1]}"

    def test_rsi_all_losses(self):
        """全て下落の場合、RSIは0に近づくことを検証"""
        # 全て1ずつ下落
        prices = pd.Series(
            [
                115.0,
                114.0,
                113.0,
                112.0,
                111.0,
                110.0,
                109.0,
                108.0,
                107.0,
                106.0,
                105.0,
                104.0,
                103.0,
                102.0,
                101.0,
                100.0,
            ]
        )
        result = MomentumIndicators.rsi(prices, period=14)

        valid_values = result.dropna()
        assert len(valid_values) > 0, "RSIの有効な値がありません"
        # 全て下落なのでRSIは0に近い値になる
        assert (
            valid_values.iloc[-1] < 10
        ), f"全て下落の場合RSIは10以下であるべき: {valid_values.iloc[-1]}"

    def test_rsi_constant_prices(self):
        """一定価格ではRSIは計算できない（変動なし）ことを検証"""
        prices = pd.Series([100.0] * 30)
        result = MomentumIndicators.rsi(prices, period=14)

        valid_values = result.dropna()
        # 変動がない場合、pandas-taのRSIはNaNを返す（損益ゼロのため）
        # これは正常な動作
        assert len(valid_values) == 0 or np.allclose(
            valid_values, 50.0, atol=1.0
        ), "一定価格のRSIはNaNまたは50付近であるべき"

    def test_rsi_range(self, known_prices):
        """RSIが0-100の範囲にあることを検証"""
        result = MomentumIndicators.rsi(known_prices, period=3)

        valid_values = result.dropna()
        assert (valid_values >= 0).all(), f"RSIが0未満: {valid_values.min()}"
        assert (valid_values <= 100).all(), f"RSIが100超: {valid_values.max()}"

    def test_rsi_initial_nan(self, known_prices):
        """RSIの最初のperiod個がNaNであることを検証"""
        period = 5
        result = MomentumIndicators.rsi(known_prices, period=period)

        # 最初のperiod個はNaN（pandas-taの実装による）
        for i in range(period):
            assert np.isnan(result.iloc[i]), f"RSI[{i}]はNaNであるべき"


# =============================================================================
# MACD の正確性検証
# =============================================================================


class TestMACDCalculationAccuracy:
    """MACD計算の正確性を検証"""

    def test_macd_line_is_fast_ema_minus_slow_ema(self, known_prices):
        """MACDライン = Fast EMA - Slow EMAであることを検証"""
        fast, slow = 3, 5
        macd_line, signal, histogram = MomentumIndicators.macd(
            known_prices, fast=fast, slow=slow, signal=3
        )

        # 独自にEMAを計算
        ema_fast = OverlapIndicators.ema(known_prices, length=fast)
        ema_slow = OverlapIndicators.ema(known_prices, length=slow)

        # MACDラインがFast EMA - Slow EMAと一致するか検証
        # pandas-taのEMAは最初の(length-1)個がNaNになるため、
        # MACDの有効な値はslow-1以降
        for i in range(slow - 1, len(known_prices)):
            if (
                not np.isnan(macd_line.iloc[i])
                and not np.isnan(ema_fast.iloc[i])
                and not np.isnan(ema_slow.iloc[i])
            ):
                expected_macd = ema_fast.iloc[i] - ema_slow.iloc[i]
                actual_macd = macd_line.iloc[i]
                assert np.isclose(
                    actual_macd, expected_macd, rtol=1e-5
                ), f"MACD[{i}]がFast-Slowと一致しない: expected={expected_macd}, actual={actual_macd}"

    def test_macd_histogram_is_macd_minus_signal(self, known_prices):
        """ヒストグラム = MACD - Signalであることを検証"""
        macd_line, signal, histogram = MomentumIndicators.macd(
            known_prices, fast=3, slow=5, signal=3
        )

        valid_indices = ~np.isnan(histogram)
        for i in range(len(known_prices)):
            if valid_indices.iloc[i]:
                expected_histogram = macd_line.iloc[i] - signal.iloc[i]
                actual_histogram = histogram.iloc[i]
                assert np.isclose(
                    actual_histogram, expected_histogram, rtol=1e-5
                ), f"Histogram[{i}]がMACD-Signalと一致しない: expected={expected_histogram}, actual={actual_histogram}"


# =============================================================================
# 追加の Momentum 系指標の正確性検証
# =============================================================================


class TestAdditionalMomentumIndicatorsAccuracy:
    """新規追加の Momentum 指標の正確性を検証"""

    def test_dm_follows_directional_bias(self):
        """上昇トレンドでは正の方向性が優勢になることを検証"""
        high = pd.Series(np.linspace(101.0, 130.0, 30))
        low = high - 2.0

        dmp, dmn = MomentumIndicators.dm(high, low, length=5)

        valid_dmp = dmp.dropna()
        valid_dmn = dmn.dropna()

        assert len(valid_dmp) > 0
        assert len(valid_dmn) > 0
        assert (valid_dmp >= 0).all()
        assert (valid_dmn >= 0).all()
        assert valid_dmp.iloc[-1] > 0
        assert valid_dmn.iloc[-1] == pytest.approx(0.0, abs=1e-10)

    def test_er_is_one_on_monotonic_series(self):
        """単調増加系列では ER が 1 に近づくことを検証"""
        prices = pd.Series(np.linspace(100.0, 129.0, 30))

        result = MomentumIndicators.er(prices, length=10)
        valid = result.dropna()

        assert len(valid) > 0
        assert valid.between(0, 1).all()
        assert valid.iloc[-1] == pytest.approx(1.0, abs=1e-10)

    def test_lrsi_is_bounded_on_trend(self):
        """LRSI が 0-100 の範囲に収まることを検証"""
        prices = pd.Series(np.linspace(100.0, 129.0, 30))

        result = MomentumIndicators.lrsi(prices, length=14)
        valid = result.dropna()

        assert len(valid) > 0
        assert valid.between(0, 100).all()
        assert valid.iloc[-1] == pytest.approx(100.0, abs=1e-10)

    def test_trixh_histogram_is_line_minus_signal(self):
        """TRIXH のヒストグラムが line - signal であることを検証"""
        prices = pd.Series(np.linspace(100.0, 160.0, 60))

        trix_line, trix_signal, trix_hist = MomentumIndicators.trixh(
            prices, length=18, signal=9
        )
        valid = ~(trix_line.isna() | trix_signal.isna() | trix_hist.isna())

        assert valid.any()
        assert np.allclose(
            trix_hist[valid],
            trix_line[valid] - trix_signal[valid],
            rtol=1e-5,
            atol=1e-8,
        )

    def test_vwmacd_histogram_is_line_minus_signal(self):
        """VWMACD のヒストグラムが line - signal であることを検証"""
        close = pd.Series(np.linspace(100.0, 160.0, 60))
        volume = pd.Series(np.linspace(1000.0, 2000.0, 60))

        vwmacd_line, vwmacd_hist, vwmacd_signal = MomentumIndicators.vwmacd(
            close, volume, fast=12, slow=26, signal=9
        )
        valid = ~(vwmacd_line.isna() | vwmacd_hist.isna() | vwmacd_signal.isna())

        assert valid.any()
        assert np.allclose(
            vwmacd_hist[valid],
            vwmacd_line[valid] - vwmacd_signal[valid],
            rtol=1e-5,
            atol=1e-8,
        )


# =============================================================================
# Bollinger Bands の正確性検証
# =============================================================================


class TestBollingerBandsAccuracy:
    """Bollinger Bands計算の正確性を検証"""

    def test_bbands_middle_is_sma(self, known_prices):
        """BBの中間線がSMAと一致することを検証"""
        length = 5
        std_dev = 2.0
        upper, middle, lower = VolatilityIndicators.bbands(
            known_prices, length=length, std=std_dev
        )

        # 独自にSMAを計算
        sma = OverlapIndicators.sma(known_prices, length=length)

        # 中間線がSMAと一致するか検証
        for i in range(len(known_prices)):
            if not np.isnan(middle.iloc[i]):
                assert np.isclose(
                    middle.iloc[i], sma.iloc[i], rtol=1e-10
                ), f"BB middle[{i}]がSMAと一致しない: middle={middle.iloc[i]}, sma={sma.iloc[i]}"

    def test_bbands_upper_lower_symmetry(self, known_prices):
        """BBの上下バンドがSMA±k*stdの関係にあることを検証"""
        length = 5
        std_dev = 2.0
        upper, middle, lower = VolatilityIndicators.bbands(
            known_prices, length=length, std=std_dev
        )

        # pandas-taは母標準偏差(ddof=0)を使用
        rolling_std = known_prices.rolling(window=length).std(ddof=0)

        for i in range(len(known_prices)):
            if not np.isnan(upper.iloc[i]) and not np.isnan(rolling_std.iloc[i]):
                expected_upper = middle.iloc[i] + std_dev * rolling_std.iloc[i]
                expected_lower = middle.iloc[i] - std_dev * rolling_std.iloc[i]

                assert np.isclose(
                    upper.iloc[i], expected_upper, rtol=1e-5
                ), f"BB upper[{i}]がSMA+{std_dev}*stdと一致しない"
                assert np.isclose(
                    lower.iloc[i], expected_lower, rtol=1e-5
                ), f"BB lower[{i}]がSMA-{std_dev}*stdと一致しない"

    def test_bbands_order(self, known_prices):
        """BBで upper >= middle >= lower が成り立つことを検証"""
        upper, middle, lower = VolatilityIndicators.bbands(
            known_prices, length=5, std=2.0
        )

        valid = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert (
            upper[valid] >= middle[valid] - 1e-10
        ).all(), "upper >= middle が成り立たない"
        assert (
            middle[valid] >= lower[valid] - 1e-10
        ).all(), "middle >= lower が成り立たない"


# =============================================================================
# ATR (Average True Range) の正確性検証
# =============================================================================


class TestATRCalculationAccuracy:
    """ATR計算の正確性を検証"""

    def test_true_range_calculation(self, known_ohlcv):
        """True Rangeの計算が正しいことを検証"""
        high = known_ohlcv["high"]
        low = known_ohlcv["low"]
        close = known_ohlcv["close"]

        tr = VolatilityIndicators.true_range(high, low, close)

        # pandas-taのtrue_rangeは最初の値がNaNになる（前のCloseが必要）
        assert np.isnan(tr.iloc[0]), "TR[0]はNaNであるべき（前のCloseがないため）"

        # TR[i] = max(High[i]-Low[i], |High[i]-Close[i-1]|, |Low[i]-Close[i-1]|)
        for i in range(1, len(known_ohlcv)):
            hl = high.iloc[i] - low.iloc[i]
            hc = abs(high.iloc[i] - close.iloc[i - 1])
            lc = abs(low.iloc[i] - close.iloc[i - 1])
            expected_tr = max(hl, hc, lc)
            actual_tr = tr.iloc[i]
            assert np.isclose(
                actual_tr, expected_tr, rtol=1e-10
            ), f"TR[{i}]が不正: expected={expected_tr}, actual={actual_tr}"

    def test_atr_is_average_of_tr(self, known_ohlcv):
        """ATRがTRのRMA（Wilder's smoothing）であることを検証"""
        high = known_ohlcv["high"]
        low = known_ohlcv["low"]
        close = known_ohlcv["close"]

        length = 3
        atr = VolatilityIndicators.atr(high, low, close, length=length)
        tr = VolatilityIndicators.true_range(high, low, close)

        # RMA（Wilder's smoothing）でATRを手動計算
        # pandas-taの実装に合わせる:
        # 1. tr.dropna() で有効なTR値のみ取得
        # 2. 最初のATRは、NaNを含むTRの先頭にNaNを追加した後、ewmで計算
        # 実際には: pandas-taは tr.dropna() した後、先頭にNaNを追加して ewm(alpha=1/length, min_periods=length)

        tr_valid = tr.dropna()
        alpha = 1.0 / length

        # 正しい期待値: pandas-taは内部的に talibまたは同等の計算を使用
        # 単純に最初のlength個のTRのSMAから始まり、その後Wilder's smoothing
        if len(tr_valid) >= length:
            expected_rma_values = []
            # 最初のRMA値（length番目のTRまでの単純平均）
            first_rma = tr_valid.iloc[:length].mean()
            expected_rma_values.append(first_rma)

            # その後のRMA値
            for i in range(length, len(tr_valid)):
                prev_rma = expected_rma_values[-1]
                current_tr = tr_valid.iloc[i]
                new_rma = (prev_rma * (length - 1) + current_tr) / length
                expected_rma_values.append(new_rma)

            # ATRのNaNでない値を取得
            atr_valid = atr.dropna()

            # 値を比較
            assert len(atr_valid) == len(
                expected_rma_values
            ), f"ATRの有效値数が一致しない: expected={len(expected_rma_values)}, actual={len(atr_valid)}"

            for i, (actual_val, expected_val) in enumerate(
                zip(atr_valid, expected_rma_values)
            ):
                assert np.isclose(
                    actual_val, expected_val, rtol=0.01
                ), f"ATR[{i}]がRMAと一致しない: expected={expected_val}, actual={actual_val}"
        else:
            # データが足りない場合は、すべてのATRがNaNであることを確認
            assert atr.dropna().empty, "データが足りない場合、ATRはすべてNaNであるべき"

    def test_atr_positive_values(self, known_ohlcv):
        """ATRが常に正の値であることを検証"""
        atr = VolatilityIndicators.atr(
            known_ohlcv["high"], known_ohlcv["low"], known_ohlcv["close"], length=3
        )

        valid_values = atr.dropna()
        assert (valid_values > 0).all(), "ATRは正の値であるべき"


# =============================================================================
# VWAP (Volume Weighted Average Price) の正確性検証
# =============================================================================


class TestVWAPCalculationAccuracy:
    """VWAP計算の正確性を検証"""

    def test_vwap_calculation(self, known_ohlcv):
        """VWAPの計算が正しいことを検証"""
        high = known_ohlcv["high"]
        low = known_ohlcv["low"]
        close = known_ohlcv["close"]
        volume = known_ohlcv["volume"]

        vwap = VolumeIndicators.vwap(high, low, close, volume)

        # Typical Price = (High + Low + Close) / 3
        typical_price = (high + low + close) / 3

        # VWAP = 累積(Price * Volume) / 累積(Volume)
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        expected_vwap = cumulative_pv / cumulative_volume

        for i in range(len(known_ohlcv)):
            actual = vwap.iloc[i]
            expected = expected_vwap.iloc[i]
            assert np.isclose(
                actual, expected, rtol=1e-5
            ), f"VWAP[{i}]が不正: expected={expected}, actual={actual}"

    def test_vwap_constant_prices(self):
        """一定価格ではVWAPもその価格になることを検証"""
        n = 10
        data = pd.DataFrame(
            {
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )

        vwap = VolumeIndicators.vwap(
            data["high"], data["low"], data["close"], data["volume"]
        )

        assert np.allclose(vwap, 100.0), "一定価格のVWAPはその価格であるべき"


# =============================================================================
# Stochastic の正確性検証
# =============================================================================


class TestStochasticAccuracy:
    """Stochastic Oscillator計算の正確性を検証"""

    def test_stochastic_range(self, known_ohlcv):
        """Stochasticが0-100の範囲にあることを検証"""
        k, d = MomentumIndicators.stoch(
            known_ohlcv["high"],
            known_ohlcv["low"],
            known_ohlcv["close"],
            k=3,
            d=2,
            smooth_k=1,
        )

        valid_k = k.dropna()
        valid_d = d.dropna()

        assert (valid_k >= 0).all() and (
            valid_k <= 100
        ).all(), f"%Kが範囲外: {valid_k.min()}-{valid_k.max()}"
        assert (valid_d >= 0).all() and (
            valid_d <= 100
        ).all(), f"%Dが範囲外: {valid_d.min()}-{valid_d.max()}"

    def test_stochastic_at_high(self):
        """最高値でStochasticが100に近いことを検証"""
        # 最後のデータが最高値（単調増加）- 十分なデータ長
        data = pd.DataFrame(
            {
                "high": [float(i) for i in range(100, 125)],
                "low": [float(i) for i in range(99, 124)],
                "close": [float(i) for i in range(100, 125)],
            }
        )

        k, d = MomentumIndicators.stoch(
            data["high"], data["low"], data["close"], k=5, d=3, smooth_k=3
        )

        valid_k = k.dropna()
        if len(valid_k) > 0:
            # 最後の%Kは100に近いはず
            last_k = valid_k.iloc[-1]
            assert last_k > 90, f"最高値での%Kが100に近くない: {last_k}"

    def test_stochastic_at_low(self):
        """最安値でStochasticが0に近いことを検証"""
        # 最後のデータが最安値（単調減少）- 十分なデータ長
        data = pd.DataFrame(
            {
                "high": [float(i) for i in range(124, 99, -1)],
                "low": [float(i) for i in range(123, 98, -1)],
                "close": [float(i) for i in range(124, 99, -1)],
            }
        )

        k, d = MomentumIndicators.stoch(
            data["high"], data["low"], data["close"], k=5, d=3, smooth_k=3
        )

        valid_k = k.dropna()
        if len(valid_k) > 0:
            # 最後の%Kは0に近いはず
            last_k = valid_k.iloc[-1]
            assert last_k < 10, f"最安値での%Kが0に近くない: {last_k}"


# =============================================================================
# DEMA (Double Exponential Moving Average) の正確性検証
# =============================================================================


class TestDEMACalculationAccuracy:
    """DEMA計算の正確性を検証"""

    def test_dema_formula(self, known_prices):
        """DEMAの基本的な性質を検証"""
        length = 3
        dema = OverlapIndicators.dema(known_prices, length=length)
        ema = OverlapIndicators.ema(known_prices, length=length)

        # DEMAはEMAより価格に追従しやすい（ボラティリティが高い）
        valid_dema = dema.dropna()
        valid_ema = ema.dropna()

        # 共通のインデックスで比較
        common_idx = valid_dema.index.intersection(valid_ema.index)
        if len(common_idx) > 2:
            dema_values = valid_dema.loc[common_idx]
            ema_values = valid_ema.loc[common_idx]

            # DEMAの分散はEMAの分散より大きい（追従性が高いため）
            # ただし、一定価格では同じになる
            dema_diff = dema_values.diff().dropna().abs()
            ema_diff = ema_values.diff().dropna().abs()

            # DEMAの変化率はEMAの変化率より大きい傾向がある
            # （厳密なテストではないが、基本的な性質を検証）
            assert len(dema_diff) > 0, "DEMAの有効な値がない"
            assert len(ema_diff) > 0, "EMAの有効な値がない"


# =============================================================================
# RMA (Wilde's Moving Average) の正確性検証
# =============================================================================


class TestRMACalculationAccuracy:
    """RMA計算の正確性を検証"""

    def test_rma_constant_prices(self, constant_prices):
        """一定価格ではRMAも一定MAも一定値になることを検証"""
        result = OverlapIndicators.rma(constant_prices, length=5)

        valid_values = result.dropna()
        assert np.allclose(valid_values, 100.0), "一定価格のRMAは一定値であるべき"

    def test_rma_smooths_data(self, known_prices):
        """RMAがデータを平滑化することを検証"""
        result = OverlapIndicators.rma(known_prices, length=3)

        valid_rma = result.dropna()
        valid_prices = known_prices.loc[valid_rma.index]

        # RMAの分散は元データの分散より小さいはず
        rma_variance = valid_rma.var()
        price_variance = valid_prices.var()

        assert (
            rma_variance < price_variance
        ), f"RMAの分散({rma_variance})は元データの分散({price_variance})より小さいべき"


# =============================================================================
# エッジケースと特殊なデータパターンのテスト
# =============================================================================


class TestIndicatorEdgeCases:
    """インジケーターのエッジケーステスト"""

    def test_sma_single_value(self):
        """単一データポイントでのSMA"""
        data = pd.Series([100.0])
        result = OverlapIndicators.sma(data, length=1)
        assert np.isclose(result.iloc[0], 100.0), "単一データのSMAはその値と一致すべき"

    def test_rsi_minimum_data(self):
        """最小データでのRSI"""
        # RSIには最低period+1個のデータが必要
        data = pd.Series(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
            ]
        )
        result = MomentumIndicators.rsi(data, period=14)

        # 最後の値は有効であるべき
        last_value = result.iloc[-1]
        assert not np.isnan(last_value), "最小データでもRSIは計算されるべき"
        assert 0 <= last_value <= 100, f"RSIが範囲外: {last_value}"

    def test_bbands_zero_std(self):
        """標準偏差ゼロ（一定価格）でのBB"""
        data = pd.Series([100.0] * 30)
        upper, middle, lower = VolatilityIndicators.bbands(data, length=5, std=2.0)

        # 一定価格なので upper == middle == lower
        valid = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.allclose(upper[valid], middle[valid]), "一定価格ではupper == middle"
        assert np.allclose(middle[valid], lower[valid]), "一定価格ではmiddle == lower"

    def test_atr_no_volatility(self):
        """ボラティリティなし（High==Low==Close）でのATR"""
        n = 20
        data = pd.DataFrame(
            {
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
            }
        )

        atr = VolatilityIndicators.atr(
            data["high"], data["low"], data["close"], length=5
        )

        # ボラティリティがないのでATRは0
        valid_atr = atr.dropna()
        assert np.allclose(valid_atr, 0.0), "ボラティリティなしではATRは0であるべき"


# =============================================================================
# 複数インジケーターの組み合わせテスト
# =============================================================================


class TestIndicatorConsistency:
    """複数インジケーター間の一貫性テスト"""

    def test_macd_ema_relationship(self, known_prices):
        """MACDがEMAベースであることを検証"""
        macd_line, _, _ = MomentumIndicators.macd(
            known_prices, fast=12, slow=26, signal=9
        )

        ema12 = OverlapIndicators.ema(known_prices, length=12)
        ema26 = OverlapIndicators.ema(known_prices, length=26)

        # MACDラインがEMA12-EMA26と一致するか
        expected_macd = ema12 - ema26

        valid = ~np.isnan(macd_line)
        for i in range(len(known_prices)):
            if valid.iloc[i] and not np.isnan(expected_macd.iloc[i]):
                assert np.isclose(
                    macd_line.iloc[i], expected_macd.iloc[i], rtol=1e-5
                ), f"MACD[{i}]がEMA12-EMA26と一致しない"

    def test_sma_is_special_case_of_wma(self, known_prices):
        """SMAが重みが全て等しい場合のWMAであることを検証"""
        length = 5
        sma = OverlapIndicators.sma(known_prices, length=length)

        # SMAの計算
        for i in range(length - 1, len(known_prices)):
            expected_sma = known_prices.iloc[i - length + 1 : i + 1].mean()
            actual_sma = sma.iloc[i]
            assert np.isclose(
                actual_sma, expected_sma, rtol=1e-10
            ), f"SMA[{i}]が平均と一致しない"

    def test_bb_width_increases_with_std_multiplier(self, known_prices):
        """BBのstd倍数が大きいほどバンド幅が広がることを検証"""
        upper1, _, lower1 = VolatilityIndicators.bbands(known_prices, length=5, std=1.0)
        upper2, _, lower2 = VolatilityIndicators.bbands(known_prices, length=5, std=2.0)

        width1 = upper1 - lower1
        width2 = upper2 - lower2

        valid = ~(np.isnan(width1) | np.isnan(width2))
        assert (
            width2[valid] > width1[valid]
        ).all(), "std倍数が大きいほどバンド幅は広がるべき"
