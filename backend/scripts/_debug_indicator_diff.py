"""アプリの再実装インジケーターと pandas_ta_classic 参照実装の差分検証スクリプト

使い方: uv run python scripts/_debug_indicator_diff.py

アプリの再実装(Momentum/Overlap/Trend/Volatility/Volume)と参照ライブラリを
同一データ・同一パラメータで比較し、計算ズレを検出する。
パラメータ対応を揃えた状態で 116/123 が完全一致(残りは NaN 位置の差や
意図的なカスタム実装: sar / vwap)。
"""

import numpy as np
import pandas as pd
import pandas_ta_classic as pta

from app.services.indicators.technical_indicators.pandas_ta import (
    MomentumIndicators,
    OverlapIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)

# ---------------------------------------------------------------- データ生成
rng = np.random.default_rng(42)
n = 400
close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), name="close")
high = pd.Series(close + rng.random(n) * 2, name="high")
low = pd.Series(close - rng.random(n) * 2, name="low")
open_ = pd.Series(
    (close.shift(1).fillna(close.iloc[0]) + close) / 2, name="open"
)
volume = pd.Series(rng.integers(100, 10000, n).astype(float), name="volume")


def to_columns(result):
    """Series/DataFrame/tuple/dict/ndarray を (name, Series) のリストに再帰正規化"""
    if result is None:
        return []
    if isinstance(result, pd.Series):
        return [(str(result.name) if result.name is not None else "out", result)]
    if isinstance(result, pd.DataFrame):
        return [(str(c), result[c]) for c in result.columns]
    if isinstance(result, dict):
        out = []
        for k, v in result.items():
            for sub_name, s in to_columns(v):
                out.append((f"{k}.{sub_name}" if sub_name != "out" else str(k), s))
        return out
    if isinstance(result, (tuple, list)):
        out = []
        for i, s in enumerate(result):
            for sub_name, ser in to_columns(s):
                out.append((f"out{i}.{sub_name}" if sub_name != "out" else f"out{i}", ser))
        return out
    if isinstance(result, np.ndarray):
        if result.ndim == 1:
            return [("out", pd.Series(result))]
        return [(f"out{i}", pd.Series(result[:, i])) for i in range(result.shape[1])]
    return []


def _series_array(s):
    return np.asarray(s, dtype=float)


def compare_pair(name, col_name, a, r, tol=1e-6):
    """1列ペアを比較し、問題リストを返す"""
    problems = []
    a_arr = _series_array(a)
    r_arr = _series_array(r)
    if len(a_arr) != len(r_arr):
        return [f"  - {col_name}: 長さ不一致 app={len(a_arr)}, ref={len(r_arr)}"]
    a_nan = np.isnan(a_arr)
    r_nan = np.isnan(r_arr)
    nan_mismatch = int((a_nan != r_nan).sum())
    both = ~a_nan & ~r_nan
    max_diff = 0.0
    if both.sum() > 0:
        d = np.abs(a_arr[both] - r_arr[both])
        rel = d / (np.abs(r_arr[both]) + 1e-12)
        max_diff = float(d.max())
        bad = (d > tol) & (rel > tol)
        if bad.any():
            idx = np.where(bad)[0][:3]
            problems.append(
                f"  - {col_name}: {int(bad.sum())}点が不一致 max_abs_diff={d.max():.6g}"
                f" (例: {[(int(j), float(a_arr[both][j]), float(r_arr[both][j])) for j in idx[:2]]})"
            )
    if nan_mismatch > 2:
        problems.append(
            f"  - {col_name}: NaN位置が{nan_mismatch}点不一致"
            f" (app NaN={int(a_nan.sum())}, ref NaN={int(r_nan.sum())})"
        )
    return problems, max_diff


def compare(name, app_result, ref_result, tol=1e-6):
    """アプリ実装と参照実装を比較する (列順の違いは相関でマッチング)"""
    app_cols = to_columns(app_result)
    ref_cols = to_columns(ref_result)

    if not app_cols or not ref_cols:
        print(f"[SKIP] {name}: 結果が空 (app={len(app_cols)}, ref={len(ref_cols)})")
        return None

    problems = []
    max_diff = 0.0

    # --- 列マッチング (相関ベース・貪欲) ---
    if len(app_cols) == 1 and len(ref_cols) == 1:
        pairs = [(app_cols[0][0], app_cols[0][1], ref_cols[0][1])]
    else:
        remaining = list(ref_cols)
        pairs = []
        for name_a, a in app_cols:
            if not remaining:
                problems.append(
                    f"  - {name_a}: 対応する参照列がない"
                    f" (app列数={len(app_cols)} > ref列数={len(ref_cols)})"
                )
                continue
            best_idx = 0
            best_corr = -2.0
            for i, (_, r) in enumerate(remaining):
                a_arr = _series_array(a)
                r_arr = _series_array(r)
                if len(a_arr) != len(r_arr):
                    corr = -1.0
                else:
                    both = ~np.isnan(a_arr) & ~np.isnan(r_arr)
                    if both.sum() < 5:
                        corr = -1.0
                    else:
                        corr = float(np.corrcoef(a_arr[both], r_arr[both])[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_idx = i
            r_name, r = remaining.pop(best_idx)
            pairs.append((f"{name_a}~{r_name}", a, r))

    for col_name, a, r in pairs:
        col_problems, col_max = compare_pair(name, col_name, a, r, tol=tol)
        problems.extend(col_problems)
        max_diff = max(max_diff, col_max)

    if problems:
        print(f"[DIFF] {name} (max_abs_diff={max_diff:.6g}):")
        for p in problems[:4]:
            print(p)
        return False
    print(f"[OK]   {name} (列数={len(pairs)}, max_abs_diff={max_diff:.3g})")
    return True


def run_all(cases):
    ok = 0
    diff = 0
    skip = 0
    for case in cases:
        name, fn = case
        try:
            app_result, ref_result = fn()
        except Exception as e:
            print(f"[ERR] {name}: {type(e).__name__}: {e}")
            skip += 1
            continue
        try:
            result = compare(name, app_result, ref_result)
        except Exception as e:

            print(f"[CRASH] {name}: {type(e).__name__}: {e}")
            skip += 1
            continue
        if result is True:
            ok += 1
        elif result is False:
            diff += 1
        else:
            skip += 1
    print(f"\n=== 集計: OK={ok}, DIFF={diff}, SKIP={skip} / 計{len(cases)} ===")


# ================================================================ ケース定義
# momentum
def m(name, fn):
    return (f"Momentum.{name}", fn)


# vwap は DatetimeIndex が必要

dt_index = pd.date_range("2024-01-01", periods=n, freq="h")
close_dt = close.set_axis(dt_index)
high_dt = high.set_axis(dt_index)
low_dt = low.set_axis(dt_index)
volume_dt = volume.set_axis(dt_index)


cases = [
    m("rsi", lambda: (MomentumIndicators.rsi(close, period=14), pta.momentum.rsi(close, length=14))),
    m("macd", lambda: (MomentumIndicators.macd(close, fast=12, slow=26, signal=9), pta.momentum.macd(close, fast=12, slow=26, signal=9))),
    m("stoch", lambda: (MomentumIndicators.stoch(high, low, close, k=14, d=3, smooth_k=3), pta.momentum.stoch(high, low, close, k=14, d=3, smooth_k=3))),
    m("stochrsi", lambda: (MomentumIndicators.stochrsi(close, rsi_length=14, stoch_length=14, k=3, d=3), pta.momentum.stochrsi(close, length=14, k=3, d=3))),
    m("cci", lambda: (MomentumIndicators.cci(high, low, close, length=20), pta.momentum.cci(high, low, close, length=20))),
    m("roc", lambda: (MomentumIndicators.roc(close, period=10), pta.momentum.roc(close, length=10))),
    m("mom", lambda: (MomentumIndicators.mom(close, length=10), pta.momentum.mom(close, length=10))),
    m("willr", lambda: (MomentumIndicators.willr(high, low, close, length=14), pta.momentum.willr(high, low, close, length=14))),
    m("cg", lambda: (MomentumIndicators.cg(close, length=10), pta.momentum.cg(close, length=10))),
    m("cmo", lambda: (MomentumIndicators.cmo(close, length=14), pta.momentum.cmo(close, length=14))),
    m("er", lambda: (MomentumIndicators.er(close, length=10), pta.momentum.er(close, length=10))),
    m("efficiency_ratio", lambda: (MomentumIndicators.efficiency_ratio(close, length=10), pta.momentum.er(close, length=10))),
    m("tsi", lambda: (MomentumIndicators.tsi(close, slow=25, fast=13), pta.momentum.tsi(close, slow=25, fast=13))),
    m("trix", lambda: (MomentumIndicators.trix(close, length=15), pta.momentum.trix(close, length=15))),
    m("uo", lambda: (MomentumIndicators.uo(high, low, close, fast=7, medium=14, slow=28), pta.momentum.uo(high, low, close, fast=7, medium=14, slow=28))),
    m("ao", lambda: (MomentumIndicators.ao(high, low, fast=5, slow=34), pta.momentum.ao(high, low, fast=5, slow=34))),
    m("apo", lambda: (MomentumIndicators.apo(close, fast=12, slow=26), pta.momentum.apo(close, fast=12, slow=26))),
    m("ppo", lambda: (MomentumIndicators.ppo(close, fast=12, slow=26), pta.momentum.ppo(close, fast=12, slow=26))),
    m("bias", lambda: (MomentumIndicators.bias(close, length=26), pta.momentum.bias(close, length=26))),
    m("bop", lambda: (MomentumIndicators.bop(open_, high, low, close), pta.momentum.bop(open_, high, low, close))),
    m("cfo", lambda: (MomentumIndicators.cfo(close, length=9), pta.momentum.cfo(close, length=9))),
    m("coppock", lambda: (MomentumIndicators.coppock(close, length=11, fast=14, slow=10), pta.momentum.coppock(close, length=11, fast=14, slow=10))),
    m("cti", lambda: (MomentumIndicators.cti(close, length=20), pta.momentum.cti(close, length=20))),
    m("dm", lambda: (MomentumIndicators.dm(high, low, length=14), pta.momentum.dm(high, low, length=14))),
    m("eri", lambda: (MomentumIndicators.eri(high, low, close, length=13), pta.momentum.eri(high, low, close, length=13))),
    m("fisher", lambda: (MomentumIndicators.fisher(high, low, length=9), pta.momentum.fisher(high, low, length=9))),
    m("inertia", lambda: (MomentumIndicators.inertia(close, length=14), pta.momentum.inertia(close, length=14))),
    m("kdj", lambda: (MomentumIndicators.kdj(high, low, close, length=9), pta.momentum.kdj(high, low, close, length=9))),
    m("kst", lambda: (MomentumIndicators.kst(close), pta.momentum.kst(close))),
    m("lrsi", lambda: (MomentumIndicators.lrsi(close, length=14), pta.momentum.lrsi(close, length=14))),
    m("pgo", lambda: (MomentumIndicators.pgo(high, low, close, length=14), pta.momentum.pgo(high, low, close, length=14))),
    m("psl", lambda: (MomentumIndicators.psl(close, open_=None, length=12), pta.momentum.psl(close, open_=None, length=12))),
    m("qqe", lambda: (MomentumIndicators.qqe(close, length=14), pta.momentum.qqe(close, length=14))),
    m("rsx", lambda: (MomentumIndicators.rsx(close, length=14), pta.momentum.rsx(close, length=14))),
    m("rvgi", lambda: (MomentumIndicators.rvgi(open_, high, low, close, length=14), pta.momentum.rvgi(open_, high, low, close, length=14))),
    m("slope", lambda: (MomentumIndicators.slope(close, length=14), pta.momentum.slope(close, length=14))),
    m("smi", lambda: (MomentumIndicators.smi(close, fast=5, slow=20, signal=5), pta.momentum.smi(close, fast=5, slow=20, signal=5))),
    m("squeeze", lambda: (
        MomentumIndicators.squeeze(high, low, close),
        pta.momentum.squeeze(high, low, close),
    )),
    m("squeeze_pro", lambda: (
        MomentumIndicators.squeeze_pro(high, low, close),
        pta.momentum.squeeze_pro(high, low, close),
    )),
    m("stc", lambda: (MomentumIndicators.stc(close, fast=23, slow=50), pta.momentum.stc(close, fast=23, slow=50, tclength=10, factor=0.3))),
    m("td_seq", lambda: (MomentumIndicators.td_seq(close, show_all=False), pta.momentum.td_seq(close))),
    m("trixh", lambda: (MomentumIndicators.trixh(close, length=18, signal=9), pta.momentum.trixh(close, length=18, signal=9))),
    m("vwmacd", lambda: (MomentumIndicators.vwmacd(close, volume, fast=12, slow=26, signal=9), pta.momentum.vwmacd(close, volume, fast=12, slow=26, signal=9))),
]

# overlap
def o(name, fn):
    return (f"Overlap.{name}", fn)


cases += [
    o("sma", lambda: (OverlapIndicators.sma(close, length=20), pta.overlap.sma(close, length=20))),
    o("ema", lambda: (OverlapIndicators.ema(close, length=20), pta.overlap.ema(close, length=20))),
    o("wma", lambda: (OverlapIndicators.wma(close, length=20), pta.overlap.wma(close, length=20))),
    o("dema", lambda: (OverlapIndicators.dema(close, length=20), pta.overlap.dema(close, length=20))),
    o("tema", lambda: (OverlapIndicators.tema(close, length=20), pta.overlap.tema(close, length=20))),
    o("trima", lambda: (OverlapIndicators.trima(close, length=20), pta.overlap.trima(close, length=20))),
    o("hma", lambda: (OverlapIndicators.hma(close, length=20), pta.overlap.hma(close, length=20))),
    o("rma", lambda: (OverlapIndicators.rma(close, length=20), pta.overlap.rma(close, length=20))),
    o("vwma", lambda: (OverlapIndicators.vwma(close, volume, length=20), pta.overlap.vwma(close, volume, length=20))),
    o("linreg", lambda: (OverlapIndicators.linreg(close, length=20), pta.overlap.linreg(close, length=20))),
    o("linregslope", lambda: (OverlapIndicators.linregslope(close, length=20), pta.overlap.linregslope(close, length=20))),
    o("midpoint", lambda: (OverlapIndicators.midpoint(close, length=20), pta.overlap.midpoint(close, length=20))),
    o("midprice", lambda: (OverlapIndicators.midprice(high, low, length=20), pta.overlap.midprice(high, low, length=20))),
    o("zlma", lambda: (OverlapIndicators.zlma(close, length=20), pta.overlap.zlma(close, length=20))),
    o("kama", lambda: (OverlapIndicators.kama(close, length=20), pta.overlap.kama(close, length=20))),
    o("alma", lambda: (OverlapIndicators.alma(close, length=10, offset=0.85, sigma=6), pta.overlap.alma(close, length=10, offset=0.85, sigma=6))),
    o("fwma", lambda: (OverlapIndicators.fwma(close, length=20), pta.overlap.fwma(close, length=20))),
    o("pwma", lambda: (OverlapIndicators.pwma(close, length=20), pta.overlap.pwma(close, length=20))),
    o("swma", lambda: (OverlapIndicators.swma(close, length=10), pta.overlap.swma(close, length=10))),
    o("sinwma", lambda: (OverlapIndicators.sinwma(close, length=20), pta.overlap.sinwma(close, length=20))),
    o("t3", lambda: (OverlapIndicators.t3(close, length=20), pta.overlap.t3(close, length=20))),
    o("vidya", lambda: (OverlapIndicators.vidya(close, length=20), pta.overlap.vidya(close, length=20))),
    o("mcgd", lambda: (OverlapIndicators.mcgd(close, length=20), pta.overlap.mcgd(close, length=20))),
    o("jma", lambda: (OverlapIndicators.jma(close, length=20), pta.overlap.jma(close, length=20, phase=50))),
    o("ssf", lambda: (OverlapIndicators.ssf(close, length=20), pta.overlap.ssf(close, length=20))),
    o("hilo", lambda: (OverlapIndicators.hilo(high, low, close, high_length=13, low_length=21), pta.overlap.hilo(high, low, close, high_length=13, low_length=21))),
    o("ichimoku", lambda: (
        OverlapIndicators.ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_span_b_period=52),
        pta.overlap.ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52),
    )),
    o("jma", lambda: (OverlapIndicators.jma(close, length=7, phase=50), pta.overlap.jma(close, length=7, phase=50))),
    o("supertrend", lambda: (OverlapIndicators.supertrend(high, low, close, period=10, multiplier=3.0), pta.overlap.supertrend(high, low, close, length=10, multiplier=3.0))),
    o("hl2", lambda: (OverlapIndicators.hl2(high, low), pta.overlap.hl2(high, low))),
    o("hlc3", lambda: (OverlapIndicators.hlc3(high, low, close), pta.overlap.hlc3(high, low, close))),
    o("ohlc4", lambda: (OverlapIndicators.ohlc4(open_, high, low, close), pta.overlap.ohlc4(open_, high, low, close))),
    o("wcp", lambda: (OverlapIndicators.wcp(high, low, close), pta.overlap.wcp(high, low, close))),
]

# trend
def t(name, fn):
    return (f"Trend.{name}", fn)


cases += [
    t("adx", lambda: (TrendIndicators.adx(high, low, close, length=14), pta.trend.adx(high, low, close, length=14))),
    t("aroon", lambda: (TrendIndicators.aroon(high, low, length=14), pta.trend.aroon(high, low, length=14))),
    t("chop", lambda: (TrendIndicators.chop(high, low, close, length=14), pta.trend.chop(high, low, close, length=14))),
    t("dpo", lambda: (TrendIndicators.dpo(close, length=14), pta.trend.dpo(close, length=14, centered=False))),
    t("vortex", lambda: (TrendIndicators.vortex(high, low, close, length=14), pta.trend.vortex(high, low, close, length=14))),
    t("vhf", lambda: (TrendIndicators.vhf(close, length=28), pta.trend.vhf(close, length=28))),
    t("qstick", lambda: (TrendIndicators.qstick(open_, close, length=10), pta.trend.qstick(open_, close, length=10))),
    t("increasing", lambda: (TrendIndicators.increasing(close, length=5), pta.trend.increasing(close, length=5))),
    t("decreasing", lambda: (TrendIndicators.decreasing(close, length=5), pta.trend.decreasing(close, length=5))),
    t("cksp", lambda: (TrendIndicators.cksp(high, low, close, p=10, x=1.0, q=9), pta.trend.cksp(high, low, close, p=10, x=1.0, q=9))),
    t("ttm_trend", lambda: (TrendIndicators.ttm_trend(high, low, close, length=20), pta.trend.ttm_trend(high, low, close, length=20))),
    t("sar(psar)", lambda: (TrendIndicators.sar(high, low, af=0.02, max_af=0.2), pta.trend.psar(high, low, af=0.02, max_af=0.2))),
    t("amat", lambda: (TrendIndicators.amat(close, fast=8, slow=21), pta.trend.amat(close, fast=8, slow=21))),
    t("decay", lambda: (TrendIndicators.decay(close, length=10), pta.trend.decay(close, length=10))),
    t("long_run", lambda: (TrendIndicators.long_run(close, close, length=2), pta.trend.long_run(close, close, length=2))),
    t("short_run", lambda: (TrendIndicators.short_run(close, close, length=2), pta.trend.short_run(close, close, length=2))),
]

# volatility
def v(name, fn):
    return (f"Volatility.{name}", fn)


cases += [
    v("atr", lambda: (VolatilityIndicators.atr(high, low, close, length=14), pta.volatility.atr(high, low, close, length=14))),
    v("natr", lambda: (VolatilityIndicators.natr(high, low, close, length=14), pta.volatility.natr(high, low, close, length=14))),
    v("true_range", lambda: (VolatilityIndicators.true_range(high, low, close), pta.volatility.true_range(high, low, close))),
    v("bbands", lambda: (VolatilityIndicators.bbands(close, length=20, std=2.0), pta.volatility.bbands(close, length=20, std=2.0))),
    v("keltner(kc)", lambda: (VolatilityIndicators.keltner(high, low, close, period=20, scalar=2.0), pta.volatility.kc(high, low, close, length=20, scalar=2.0, mamode="sma"))),
    v("donchian", lambda: (VolatilityIndicators.donchian(high, low, length=20), pta.volatility.donchian(high, low, lower_length=20, upper_length=20))),
    v("massi", lambda: (VolatilityIndicators.massi(high, low, fast=9, slow=25), pta.volatility.massi(high, low, fast=9, slow=25))),
    v("ui", lambda: (VolatilityIndicators.ui(close, period=14), pta.volatility.ui(close, length=14))),
    v("accbands", lambda: (VolatilityIndicators.accbands(high, low, close, period=20), pta.volatility.accbands(high, low, close, length=20))),
    v("pdist", lambda: (VolatilityIndicators.pdist(open_, high, low, close), pta.volatility.pdist(open_, high, low, close))),
    v("rvi", lambda: (VolatilityIndicators.rvi(high, low, close, length=14), pta.volatility.rvi(high, low, close, length=14))),
    v("thermo", lambda: (VolatilityIndicators.thermo(high, low, length=20), pta.volatility.thermo(high, low, length=20, long=2, short=2))),
    v("aberration", lambda: (VolatilityIndicators.aberration(high, low, close, length=20), pta.volatility.aberration(high, low, close, length=20))),
]

# volume
def vo(name, fn):
    return (f"Volume.{name}", fn)


cases += [
    vo("ad", lambda: (VolumeIndicators.ad(high, low, close, volume), pta.volume.ad(high, low, close, volume))),
    vo("adosc", lambda: (VolumeIndicators.adosc(high, low, close, volume, fast=3, slow=10), pta.volume.adosc(high, low, close, volume, fast=3, slow=10))),
    vo("aobv", lambda: (
        VolumeIndicators.aobv(close, volume, fast=4, slow=12),
        pta.volume.aobv(close, volume, fast=4, slow=12, max_lookback=2, min_lookback=5),
    )),
    vo("cmf", lambda: (VolumeIndicators.cmf(high, low, close, volume, length=20), pta.volume.cmf(high, low, close, volume, length=20))),
    vo("efi", lambda: (VolumeIndicators.efi(close, volume, period=13), pta.volume.efi(close, volume, length=13))),
    vo("eom", lambda: (VolumeIndicators.eom(high, low, close, volume, length=14), pta.volume.eom(high, low, close, volume, length=14))),
    vo("kvo", lambda: (VolumeIndicators.kvo(high, low, close, volume, fast=34, slow=55), pta.volume.kvo(high, low, close, volume, fast=34, slow=55))),
    vo("mfi", lambda: (VolumeIndicators.mfi(high, low, close, volume, length=14), pta.volume.mfi(high, low, close, volume, length=14))),
    vo("nvi", lambda: (VolumeIndicators.nvi(close, volume), pta.volume.nvi(close, volume))),
    vo("obv", lambda: (VolumeIndicators.obv(close, volume), pta.volume.obv(close, volume))),
    vo("pvi", lambda: (VolumeIndicators.pvi(close, volume), pta.volume.pvi(close, volume, length=13))),
    vo("pvo", lambda: (VolumeIndicators.pvo(volume, fast=12, slow=26), pta.momentum.pvo(volume, fast=12, slow=26))),
    vo("pvol", lambda: (VolumeIndicators.pvol(close, volume), pta.volume.pvol(close, volume))),
    vo("pvr", lambda: (VolumeIndicators.pvr(close, volume), pta.volume.pvr(close, volume))),
    vo("pvt", lambda: (VolumeIndicators.pvt(close, volume), pta.volume.pvt(close, volume))),
    vo("vfi", lambda: (VolumeIndicators.vfi(close, volume, length=130), pta.volume.vfi(close, volume, length=130))),
    vo("vp", lambda: (VolumeIndicators.vp(close, volume, bins=10), pta.volume.vp(close, volume, width=10))),
]

cases += [
    ("Volume.vwap", lambda: (
        VolumeIndicators.vwap(high_dt, low_dt, close_dt, volume_dt, period=10),
        pta.overlap.vwap(high_dt, low_dt, close_dt, volume_dt),
    )),
]

run_all(cases)
