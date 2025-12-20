import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.overlap import OverlapIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators


@pytest.fixture
def sample_data():
    length = 200
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "open": np.random.randn(length).cumsum() + 100,
            "high": np.random.randn(length).cumsum() + 105,
            "low": np.random.randn(length).cumsum() + 95,
            "close": np.random.randn(length).cumsum() + 100,
            "volume": np.random.randint(100, 1000, length),
        },
        index=dates,
    )

    # Ensure High is highest and Low is lowest
    data["high"] = data[["open", "high", "low", "close"]].max(axis=1) + 1.0
    data["low"] = data[["open", "high", "low", "close"]].min(axis=1) - 1.0

    return data


class TestMomentumIndicatorsNew:
    def test_brar(self, sample_data):
        res1, res2 = MomentumIndicators.brar(
            sample_data["open"],
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
        )
        assert isinstance(res1, pd.Series)
        assert isinstance(res2, pd.Series)
        assert len(res1) == len(sample_data)

    def test_cfo(self, sample_data):
        res = MomentumIndicators.cfo(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_eri(self, sample_data):
        res1, res2 = MomentumIndicators.eri(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(res1, pd.Series)

    def test_inertia(self, sample_data):
        res = MomentumIndicators.inertia(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_kdj(self, sample_data):
        k, d, j = MomentumIndicators.kdj(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert isinstance(j, pd.Series)

    def test_rsx(self, sample_data):
        res = MomentumIndicators.rsx(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_rvgi(self, sample_data):
        res1, res2 = MomentumIndicators.rvgi(
            sample_data["open"],
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
        )
        assert isinstance(res1, pd.Series)

    def test_slope(self, sample_data):
        res = MomentumIndicators.slope(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_smi(self, sample_data):
        res1, res2, res3 = MomentumIndicators.smi(sample_data["close"])
        assert isinstance(res1, pd.Series)

    def test_td_seq(self, sample_data):
        # td_seq usually returns a dataframe with multiple columns in newer pandas-ta
        res = MomentumIndicators.td_seq(sample_data["close"], show_all=True)
        # Check if result is valid (Series or DF)
        assert res is not None

    def test_squeeze_pro(self, sample_data):
        res = MomentumIndicators.squeeze_pro(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert res is not None


class TestTrendIndicatorsNew:
    def test_cksp(self, sample_data):
        res1, res2 = TrendIndicators.cksp(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(res1, pd.Series)

    def test_decay(self, sample_data):
        res = TrendIndicators.decay(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_qstick(self, sample_data):
        res = TrendIndicators.qstick(sample_data["open"], sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_ttm_trend(self, sample_data):
        res = TrendIndicators.ttm_trend(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(res, pd.Series)

    def test_decreasing(self, sample_data):
        res = TrendIndicators.decreasing(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_increasing(self, sample_data):
        res = TrendIndicators.increasing(sample_data["close"])
        assert isinstance(res, pd.Series)

    def test_runs(self, sample_data):
        res_long = TrendIndicators.long_run(
            fast=sample_data["close"], slow=sample_data["open"]
        )
        res_short = TrendIndicators.short_run(
            fast=sample_data["close"], slow=sample_data["open"]
        )
        assert isinstance(res_long, pd.Series)
        assert isinstance(res_short, pd.Series)


class TestVolatilityIndicatorsNew:
    def test_aberration(self, sample_data):
        r1, r2, r3, r4 = VolatilityIndicators.aberration(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(r1, pd.Series)

    def test_hwc(self, sample_data):
        r1, r2, r3 = VolatilityIndicators.hwc(sample_data["close"])
        assert isinstance(r1, pd.Series)

    def test_pdist(self, sample_data):
        res = VolatilityIndicators.pdist(
            sample_data["open"],
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
        )
        assert isinstance(res, pd.Series)

    def test_thermo(self, sample_data):
        r1, r2, r3, r4 = VolatilityIndicators.thermo(
            sample_data["high"], sample_data["low"]
        )
        assert isinstance(r1, pd.Series)


class TestOverlapIndicatorsNew:
    def test_hilo(self, sample_data):
        r1, r2, r3 = OverlapIndicators.hilo(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert isinstance(r1, pd.Series)

    def test_avgs(self, sample_data):
        # Test basic avgs added
        assert isinstance(
            OverlapIndicators.hl2(sample_data["high"], sample_data["low"]), pd.Series
        )
        assert isinstance(
            OverlapIndicators.hlc3(
                sample_data["high"], sample_data["low"], sample_data["close"]
            ),
            pd.Series,
        )
        assert isinstance(
            OverlapIndicators.ohlc4(
                sample_data["open"],
                sample_data["high"],
                sample_data["low"],
                sample_data["close"],
            ),
            pd.Series,
        )

    def test_mid(self, sample_data):
        assert isinstance(OverlapIndicators.midpoint(sample_data["close"]), pd.Series)
        assert isinstance(
            OverlapIndicators.midprice(sample_data["high"], sample_data["low"]),
            pd.Series,
        )

    def test_vidya(self, sample_data):
        assert isinstance(OverlapIndicators.vidya(sample_data["close"]), pd.Series)

    def test_wcp(self, sample_data):
        assert isinstance(
            OverlapIndicators.wcp(
                sample_data["high"], sample_data["low"], sample_data["close"]
            ),
            pd.Series,
        )

    def test_mcgd(self, sample_data):
        assert isinstance(OverlapIndicators.mcgd(sample_data["close"]), pd.Series)

    def test_jma(self, sample_data):
        assert isinstance(OverlapIndicators.jma(sample_data["close"]), pd.Series)

    def test_other_mas(self, sample_data):
        assert isinstance(OverlapIndicators.fwma(sample_data["close"]), pd.Series)
        assert isinstance(OverlapIndicators.pwma(sample_data["close"]), pd.Series)
        assert isinstance(OverlapIndicators.sinwma(sample_data["close"]), pd.Series)
        assert isinstance(OverlapIndicators.ssf(sample_data["close"]), pd.Series)
        assert isinstance(OverlapIndicators.swma(sample_data["close"]), pd.Series)


class TestVolumeIndicatorsNew:
    def test_aobv(self, sample_data):
        try:
            res_tuple = VolumeIndicators.aobv(
                sample_data["close"], sample_data["volume"]
            )
            assert isinstance(res_tuple, tuple)
            # Should have multiple series
            assert len(res_tuple) >= 1
        except Exception:
            pytest.fail("AOBV raise exception")

    def test_pvi(self, sample_data):
        assert isinstance(
            VolumeIndicators.pvi(sample_data["close"], sample_data["volume"]), pd.Series
        )

    def test_pvol(self, sample_data):
        assert isinstance(
            VolumeIndicators.pvol(sample_data["close"], sample_data["volume"]),
            pd.Series,
        )

    def test_pvr(self, sample_data):
        assert isinstance(
            VolumeIndicators.pvr(sample_data["close"], sample_data["volume"]), pd.Series
        )
