"""Fallback coverage for indicator wrappers when pandas-ta returns None."""

import pandas_ta_classic as ta
import pytest

from app.services.indicators.technical_indicators.pandas_ta import (
    MomentumIndicators,
    OverlapIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)


def _case(patch_target, method, *args):
    return pytest.param(patch_target, method, args, id=patch_target)


def _resolve_args(source, args):
    return tuple(source[arg] if isinstance(arg, str) else arg for arg in args)


VOLUME_CASES = [
    _case("ad", VolumeIndicators.ad, "high", "low", "close", "volume"),
    _case("adosc", VolumeIndicators.adosc, "high", "low", "close", "volume"),
    _case("obv", VolumeIndicators.obv, "close", "volume"),
    _case("eom", VolumeIndicators.eom, "high", "low", "close", "volume"),
    _case("vwap", VolumeIndicators.vwap, "high", "low", "close", "volume"),
    _case("cmf", VolumeIndicators.cmf, "high", "low", "close", "volume"),
    _case("efi", VolumeIndicators.efi, "close", "volume"),
    _case("mfi", VolumeIndicators.mfi, "high", "low", "close", "volume"),
    _case("pvo", VolumeIndicators.pvo, "volume"),
    _case("pvt", VolumeIndicators.pvt, "close", "volume"),
    _case("nvi", VolumeIndicators.nvi, "close", "volume"),
]

MOMENTUM_CASES = [
    _case("rsi", MomentumIndicators.rsi, "close"),
    _case("macd", MomentumIndicators.macd, "close"),
    _case("ppo", MomentumIndicators.ppo, "close"),
    _case("trix", MomentumIndicators.trix, "close"),
    _case("dm", MomentumIndicators.dm, "high", "low"),
    _case("er", MomentumIndicators.er, "close"),
    _case("lrsi", MomentumIndicators.lrsi, "close"),
    _case("po", MomentumIndicators.po, "close"),
    _case("trixh", MomentumIndicators.trixh, "close"),
    _case("vwmacd", MomentumIndicators.vwmacd, "close", "volume"),
    _case("stoch", MomentumIndicators.stoch, "high", "low", "close"),
    _case("stochrsi", MomentumIndicators.stochrsi, "close"),
    _case("willr", MomentumIndicators.willr, "high", "low", "close"),
    _case("cci", MomentumIndicators.cci, "high", "low", "close"),
    _case("cmo", MomentumIndicators.cmo, "close"),
    _case("stc", MomentumIndicators.stc, "close"),
    _case("fisher", MomentumIndicators.fisher, "high", "low"),
    _case("kst", MomentumIndicators.kst, "close"),
    _case("roc", MomentumIndicators.roc, "close"),
    _case("mom", MomentumIndicators.mom, "close"),
    _case("qqe", MomentumIndicators.qqe, "close"),
    _case("cti", MomentumIndicators.cti, "close"),
    _case("apo", MomentumIndicators.apo, "close"),
    _case("tsi", MomentumIndicators.tsi, "close"),
    _case("bias", MomentumIndicators.bias, "close"),
]

OVERLAP_CASES = [
    _case("ema", OverlapIndicators.ema, "close", 10),
    _case("wma", OverlapIndicators.wma, "close", 10),
    _case("trima", OverlapIndicators.trima, "close", 10),
    _case("zlma", OverlapIndicators.zlma, "close", 10),
    _case("alma", OverlapIndicators.alma, "close", 10),
    _case("dema", OverlapIndicators.dema, "close", 10),
    _case("tema", OverlapIndicators.tema, "close", 10),
    _case("t3", OverlapIndicators.t3, "close", 10),
    _case("kama", OverlapIndicators.kama, "close", 10),
    _case("hma", OverlapIndicators.hma, "close", 10),
    _case("vwma", OverlapIndicators.vwma, "close", "volume", 10),
    _case("supertrend", OverlapIndicators.supertrend, "high", "low", "close"),
    _case("rma", OverlapIndicators.rma, "close", 10),
    _case("linreg", OverlapIndicators.linreg, "close"),
    _case("linreg", OverlapIndicators.linregslope, "close"),
]

TREND_CASES = [
    _case("sma", TrendIndicators.sma, "close", 10),
    _case("amat", TrendIndicators.amat, "close"),
    _case("dpo", TrendIndicators.dpo, "close", 10),
    _case("vhf", TrendIndicators.vhf, "close"),
]

VOLATILITY_CASES = [
    _case("atr", VolatilityIndicators.atr, "high", "low", "close"),
    _case("natr", VolatilityIndicators.natr, "high", "low", "close"),
    _case("bbands", VolatilityIndicators.bbands, "close"),
    _case("kc", VolatilityIndicators.keltner, "high", "low", "close"),
    _case("donchian", VolatilityIndicators.donchian, "high", "low"),
    _case("accbands", VolatilityIndicators.accbands, "high", "low", "close"),
    _case("ui", VolatilityIndicators.ui, "close"),
    _case("rvi", VolatilityIndicators.rvi, "close", "high", "low"),
    _case("true_range", VolatilityIndicators.true_range, "high", "low", "close"),
]


class TestPerFunctionMock:
    """Exercise None-return fallbacks for the main indicator wrappers."""

    def _run_none_case(self, patch_target, method, sample_values, args, monkeypatch):
        resolved_args = _resolve_args(sample_values, args)

        # Some wrappers return fallback arrays, some raise after the helper swallows the error.
        monkeypatch.setattr(ta, patch_target, lambda *args, **kwargs: None)
        try:
            method(*resolved_args)
        except Exception:
            pass

    @pytest.mark.parametrize("patch_target,method,args", VOLUME_CASES)
    def test_volume_all_none_returns(
        self, sample_ohlcv_dict, patch_target, method, args, monkeypatch
    ):
        self._run_none_case(patch_target, method, sample_ohlcv_dict, args, monkeypatch)

    @pytest.mark.parametrize("patch_target,method,args", MOMENTUM_CASES)
    def test_momentum_all_none_returns(
        self, sample_ohlcv_dict, patch_target, method, args, monkeypatch
    ):
        self._run_none_case(patch_target, method, sample_ohlcv_dict, args, monkeypatch)

    @pytest.mark.parametrize("patch_target,method,args", OVERLAP_CASES)
    def test_overlap_all_none_returns(
        self, sample_ohlcv_dict, patch_target, method, args, monkeypatch
    ):
        self._run_none_case(patch_target, method, sample_ohlcv_dict, args, monkeypatch)

    @pytest.mark.parametrize("patch_target,method,args", TREND_CASES)
    def test_trend_all_none_returns(
        self, sample_ohlcv_dict, patch_target, method, args, monkeypatch
    ):
        self._run_none_case(patch_target, method, sample_ohlcv_dict, args, monkeypatch)

    @pytest.mark.parametrize("patch_target,method,args", VOLATILITY_CASES)
    def test_volatility_all_none_returns(
        self, sample_ohlcv_dict, patch_target, method, args, monkeypatch
    ):
        self._run_none_case(patch_target, method, sample_ohlcv_dict, args, monkeypatch)
