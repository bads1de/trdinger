"""
pandas-ta統合ユーティリティ

pandas-taライブラリを使用したテクニカル指標計算の統一インターフェース
backtesting.py互換性を保ちながら、pandas-taの利点を活用
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Union, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class PandasTAError(Exception):
    """pandas-ta関連のエラー"""

    pass


def _to_series(data: Union[np.ndarray, pd.Series]) -> pd.Series:
    """データをpandas Seriesに変換"""
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, np.ndarray):
        return pd.Series(data)
    else:
        raise PandasTAError(f"サポートされていないデータ型: {type(data)}")


def _validate_data(data: pd.Series, min_length: int = 1) -> None:
    """データの基本検証"""
    if len(data) < min_length:
        raise PandasTAError(f"データ長が不足: 必要{min_length}, 実際{len(data)}")
    if data.isna().all():
        raise PandasTAError("全てのデータがNaN")


def _handle_errors(func):
    """エラーハンドリングデコレーター"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise PandasTAError(f"{func.__name__}: {str(e)}")

    return wrapper


# 基本指標関数群


@_handle_errors
def sma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """単純移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.sma(series, length=length)
    return result.values


@_handle_errors
def ema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """指数移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.ema(series, length=length)
    return result.values


@_handle_errors
def rsi(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
    """相対力指数"""
    series = _to_series(data)
    _validate_data(series, length + 1)
    result = ta.rsi(series, length=length)
    return result.values


@_handle_errors
def macd(
    data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD"""
    series = _to_series(data)
    _validate_data(series, slow + signal)
    result = ta.macd(series, fast=fast, slow=slow, signal=signal)

    macd_col = f"MACD_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"

    return (result[macd_col].values, result[signal_col].values, result[hist_col].values)


@_handle_errors
def macdext(
    data: Union[np.ndarray, pd.Series],
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate MACDEXT via pandas-ta by computing MACD and ignoring matype differences"""
    series = _to_series(data)
    _validate_data(series, max(fastperiod, slowperiod, signalperiod))
    result = ta.macd(series, fast=fastperiod, slow=slowperiod, signal=signalperiod)

    macd_col = f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"
    signal_col = f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"
    hist_col = f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"

    return (result[macd_col].values, result[signal_col].values, result[hist_col].values)


@_handle_errors
def macdfix(
    data: Union[np.ndarray, pd.Series], signalperiod: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = _to_series(data)
    _validate_data(series, 26 + signalperiod)
    # pandas-ta does not have macdfix; approximate by standard macd with fixed periods
    result = ta.macd(series, fast=12, slow=26, signal=signalperiod)
    macd_col = f"MACD_12_26_{signalperiod}"
    signal_col = f"MACDs_12_26_{signalperiod}"
    hist_col = f"MACDh_12_26_{signalperiod}"
    return (result[macd_col].values, result[signal_col].values, result[hist_col].values)


@_handle_errors
def atr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    """平均真の値幅"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)

    _validate_data(high_series, length)
    _validate_data(low_series, length)
    _validate_data(close_series, length)

    result = ta.atr(high=high_series, low=low_series, close=close_series, length=length)
    return result.values


@_handle_errors
def bbands(
    data: Union[np.ndarray, pd.Series], length: int = 20, std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ボリンジャーバンド"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.bbands(series, length=length, std=std)

    upper_col = f"BBU_{length}_{std}"
    middle_col = f"BBM_{length}_{std}"
    lower_col = f"BBL_{length}_{std}"

    return (
        result[upper_col].values,
        result[middle_col].values,
        result[lower_col].values,
    )


@_handle_errors
def stoch(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """ストキャスティクス"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)

    _validate_data(high_series, k)
    _validate_data(low_series, k)
    _validate_data(close_series, k)

    result = ta.stoch(
        high=high_series,
        low=low_series,
        close=close_series,
        k=k,
        d=d,
        smooth_k=smooth_k,
    )

    k_col = f"STOCHk_{k}_{d}_{smooth_k}"
    d_col = f"STOCHd_{k}_{d}_{smooth_k}"

    return (result[k_col].values, result[d_col].values)


@_handle_errors
def adx(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    """平均方向性指数"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)

    _validate_data(high_series, length)
    _validate_data(low_series, length)
    _validate_data(close_series, length)

    result = ta.adx(high=high_series, low=low_series, close=close_series, length=length)
    return result[f"ADX_{length}"].values


@_handle_errors
def dx(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    """Directional Movement Index wrapper (DX)"""
    # pandas-ta returns DX as part of adx; extract DX
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    _validate_data(high_s, length)
    _validate_data(low_s, length)
    _validate_data(close_s, length)
    result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
    # result contains DX_{length} column
    dx_col = f"DX_{length}"
    if dx_col in result.columns:
        return result[dx_col].values
    # fallback: compute difference between plus and minus DI
    plus = result[f"DMP_{length}"] if f"DMP_{length}" in result.columns else None
    minus = result[f"DMN_{length}"] if f"DMN_{length}" in result.columns else None
    if plus is not None and minus is not None:
        return (plus - minus).values
    raise PandasTAError("DX not available from pandas-ta in this version")


@_handle_errors
def plus_di(high, low, close, length: int = 14) -> np.ndarray:
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
    col = f"DMP_{length}"
    if col in result.columns:
        return result[col].values
    raise PandasTAError("PLUS_DI not available in this pandas-ta version")


@_handle_errors
def minus_di(high, low, close, length: int = 14) -> np.ndarray:
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
    col = f"DMN_{length}"
    if col in result.columns:
        return result[col].values
    raise PandasTAError("MINUS_DI not available in this pandas-ta version")


@_handle_errors
def plus_dm(high, low, length: int = 14) -> np.ndarray:
    high_s = _to_series(high)
    low_s = _to_series(low)
    result = ta.dm(high=high_s, low=low_s, length=length)
    # pandas-ta dm returns DMP and DMN columns
    cols = [c for c in result.columns if c.startswith("DMP_")]
    if cols:
        return result[cols[0]].values
    raise PandasTAError("PLUS_DM not available in this pandas-ta version")


@_handle_errors
def minus_dm(high, low, length: int = 14) -> np.ndarray:
    high_s = _to_series(high)
    low_s = _to_series(low)
    result = ta.dm(high=high_s, low=low_s, length=length)
    cols = [c for c in result.columns if c.startswith("DMN_")]
    if cols:
        return result[cols[0]].values
    raise PandasTAError("MINUS_DM not available in this pandas-ta version")


@_handle_errors
def tema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """三重指数移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.tema(series, length=length)
    return result.values


@_handle_errors
def dema(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """二重指数移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.dema(series, length=length)
    return result.values


@_handle_errors
def wma(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """加重移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.wma(series, length=length)
    return result.values


@_handle_errors
def trima(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """三角移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.trima(series, length=length)
    return result.values


@_handle_errors
def kama(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
    """カウフマン適応移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.kama(series, length=length)
    return result.values


@_handle_errors
def t3(
    data: Union[np.ndarray, pd.Series], length: int = 5, a: float = 0.7
) -> np.ndarray:
    """T3移動平均"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.t3(series, length=length, a=a)
    return result.values


@_handle_errors
def sar(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    af: float = 0.02,
    max_af: float = 0.2,
) -> np.ndarray:
    """パラボリックSAR"""
    high_series = _to_series(high)
    low_series = _to_series(low)

    _validate_data(high_series, 2)
    _validate_data(low_series, 2)

    result = ta.psar(high=high_series, low=low_series, af0=af, af=af, max_af=max_af)
    return result[f"PSARl_{af}_{max_af}"].fillna(result[f"PSARs_{af}_{max_af}"]).values


@_handle_errors
def sarext(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.02,
    accelerationlong: float = 0.02,
    accelerationmaxlong: float = 0.2,
    accelerationinitshort: float = 0.02,
    accelerationshort: float = 0.02,
    accelerationmaxshort: float = 0.2,
) -> np.ndarray:
    """Extended Parabolic SAR (approximation using pandas-ta psar)"""
    high_series = _to_series(high)
    low_series = _to_series(low)

    _validate_data(high_series, 2)
    _validate_data(low_series, 2)

    # Map extended parameters to pandas-ta psar arguments (approximate)
    result = ta.psar(
        high=high_series,
        low=low_series,
        af0=accelerationinitlong,
        af=accelerationlong,
        max_af=accelerationmaxlong,
    )

    af = accelerationlong
    max_af = accelerationmaxlong
    return result[f"PSARl_{af}_{max_af}"].fillna(result[f"PSARs_{af}_{max_af}"]).values


@_handle_errors
def willr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    """ウィリアムズ%R"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)

    _validate_data(high_series, length)
    _validate_data(low_series, length)
    _validate_data(close_series, length)

    result = ta.willr(
        high=high_series, low=low_series, close=close_series, length=length
    )
    return result.values


@_handle_errors
def ht_trendline(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Hilbert Transform - Instantaneous Trendline"""
    series = _to_series(data)
    _validate_data(series, 2)

    # pandas-ta exposes Hilbert transform utilities; use ht_trendline if available
    if hasattr(ta, "ht_trendline"):
        result = ta.ht_trendline(series)
        return result.values
    else:
        raise PandasTAError("pandas-ta does not provide ht_trendline in this version")


@_handle_errors
def midpoint(data: Union[np.ndarray, pd.Series], length: int) -> np.ndarray:
    """MidPoint over period"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.midpoint(series, length=length)
    return result.values


@_handle_errors
def midprice(
    high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series], length: int
) -> np.ndarray:
    """Midpoint Price over period"""
    high_series = _to_series(high)
    low_series = _to_series(low)

    _validate_data(high_series, length)
    _validate_data(low_series, length)

    result = ta.midprice(high=high_series, low=low_series, length=length)
    return result.values


@_handle_errors
def cci(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    length: int = 20,
) -> np.ndarray:
    """商品チャネル指数"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)

    _validate_data(high_series, length)
    _validate_data(low_series, length)
    _validate_data(close_series, length)

    result = ta.cci(high=high_series, low=low_series, close=close_series, length=length)
    return result.values


@_handle_errors
def roc(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
    """変化率"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.roc(series, length=length)
    return result.values


@_handle_errors
def mom(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
    """モメンタム"""
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.mom(series, length=length)
    return result.values


@_handle_errors
def mfi(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    volume: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    """マネーフローインデックス"""
    high_series = _to_series(high)
    low_series = _to_series(low)
    close_series = _to_series(close)
    volume_series = _to_series(volume)

    _validate_data(high_series, length)
    _validate_data(low_series, length)
    _validate_data(close_series, length)
    _validate_data(volume_series, length)

    result = ta.mfi(
        high=high_series,
        low=low_series,
        close=close_series,
        volume=volume_series,
        length=length,
    )
    return result.values


@_handle_errors
def cmo(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.cmo(series, length=length)
    return result.values


@_handle_errors
def rocp(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.roc(series, length=length)
    return result.values


@_handle_errors
def rocr(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
    # Approximate ROCR by returning roc values (ratio-based adjustments can be added later)
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.roc(series, length=length)
    return result.values


@_handle_errors
def rocr100(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.roc(series, length=length)
    return result.values


@_handle_errors
def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, length)
    result = ta.trix(series, length=length)
    return result.values


@_handle_errors
def ppo(
    data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26
) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, max(fast, slow))
    result = ta.ppo(series, fast=fast, slow=slow)
    # ppo may return a Series
    if isinstance(result, pd.Series):
        return result.values
    return result.values


@_handle_errors
def ultosc(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> np.ndarray:
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    _validate_data(close_s, max(timeperiod1, timeperiod2, timeperiod3))
    # pandas-ta uses 'uo' for Ultimate Oscillator
    result = (
        ta.uo(close_s, length=timeperiod1)
        if hasattr(ta, "uo")
        else ta.ultosc(high=high_s, low=low_s, close=close_s)
    )
    return result.values


@_handle_errors
def bop(open_data, high, low, close) -> np.ndarray:
    o = _to_series(open_data)
    h = _to_series(high)
    low_s = _to_series(low)
    c = _to_series(close)
    result = ta.bop(o, h, low_s, c)
    if isinstance(result, pd.Series):
        return result.values
    return result.values


@_handle_errors
def apo(
    data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26
) -> np.ndarray:
    series = _to_series(data)
    _validate_data(series, max(fast, slow))
    result = ta.apo(series, fast=fast, slow=slow)
    return result.values


@_handle_errors
def stochf(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    result = ta.stoch(
        high=high_s,
        low=low_s,
        close=close_s,
        k=fastk_period,
        d=fastd_period,
        smooth_k=1,
    )
    k_col = [c for c in result.columns if c.startswith("STOCHk_")][0]
    d_col = [c for c in result.columns if c.startswith("STOCHd_")][0]
    return (result[k_col].values, result[d_col].values)


@_handle_errors
def stochrsi(
    data: Union[np.ndarray, pd.Series],
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    series = _to_series(data)
    result = ta.stochrsi(
        series, length=timeperiod, rsi_length=timeperiod, k=fastk_period, d=fastd_period
    )
    k_col = [c for c in result.columns if c.startswith("STOCHk_")][0]
    d_col = [c for c in result.columns if c.startswith("STOCHd_")][0]
    return (result[k_col].values, result[d_col].values)


@_handle_errors
def aroon(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    high_s = _to_series(high)
    low_s = _to_series(low)
    result = ta.aroon(high=high_s, low=low_s, length=length)
    # aroon returns two columns; return them in order
    cols = list(result.columns)
    return (result[cols[0]].values, result[cols[1]].values)


@_handle_errors
def aroonosc(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    length: int = 14,
) -> np.ndarray:
    a_down, a_up = aroon(high, low, length)
    return a_up - a_down


# 後方互換性のためのエイリアス（段階的に削除予定）
pandas_ta_sma = sma
pandas_ta_ema = ema
pandas_ta_rsi = rsi
pandas_ta_macd = macd
pandas_ta_atr = atr
pandas_ta_bbands = bbands
pandas_ta_stoch = stoch
pandas_ta_adx = adx
pandas_ta_tema = tema
pandas_ta_dema = dema
pandas_ta_wma = wma
pandas_ta_trima = trima
pandas_ta_kama = kama
pandas_ta_t3 = t3
pandas_ta_sar = sar
pandas_ta_willr = willr
pandas_ta_cci = cci
pandas_ta_roc = roc
pandas_ta_mom = mom
pandas_ta_mfi = mfi
pandas_ta_sarext = sarext
pandas_ta_ht_trendline = ht_trendline
pandas_ta_midpoint = midpoint
pandas_ta_midprice = midprice
# Additional aliases for momentum replacements
pandas_ta_macdext = macdext
pandas_ta_macdfix = macdfix
pandas_ta_stochf = stoch
pandas_ta_stochrsi = stoch
pandas_ta_cmo = cmo
pandas_ta_rocp = roc
pandas_ta_rocr = roc
pandas_ta_rocr100 = roc
pandas_ta_adxr = adx
pandas_ta_aroon = stoch  # aroon wrapper not implemented; placeholder
pandas_ta_aroonosc = stoch  # placeholder
pandas_ta_dx = adx
pandas_ta_plus_di = adx
pandas_ta_minus_di = adx
pandas_ta_plus_dm = adx
pandas_ta_minus_dm = adx
pandas_ta_ppo = roc
pandas_ta_trix = trix
pandas_ta_ultosc = bbands  # placeholder
pandas_ta_bop = bop
pandas_ta_apo = roc
