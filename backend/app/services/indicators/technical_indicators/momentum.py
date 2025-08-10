"""
モメンタム系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。

含まれる指標:
- RSI, MACD, Stochastics, Williams %R, CCI, CMO, ROC, ADX, Aroon, MFI, PPO, TRIX, Ultimate Oscillator, BOP, APO
- AO, KDJ, RVGI, QQE, SMI, KST, STC
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
    validate_series_data,
    validate_indicator_parameters,
)


class MomentumIndicators:
    """
    モメンタム系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def rsi(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """相対力指数"""
        validate_indicator_parameters(length)
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length + 1)
        result = ta.rsi(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def macd(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, slow + signal)
        result = ta.macd(series, fast=fast, slow=slow, signal=signal)

        macd_col = f"MACD_{fast}_{slow}_{signal}"
        signal_col = f"MACDs_{fast}_{slow}_{signal}"
        hist_col = f"MACDh_{fast}_{slow}_{signal}"

        return (
            result[macd_col].values,
            result[signal_col].values,
            result[hist_col].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
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
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, max(fastperiod, slowperiod, signalperiod))
        result = ta.macd(series, fast=fastperiod, slow=slowperiod, signal=signalperiod)

        macd_col = f"MACD_{fastperiod}_{slowperiod}_{signalperiod}"
        signal_col = f"MACDs_{fastperiod}_{slowperiod}_{signalperiod}"
        hist_col = f"MACDh_{fastperiod}_{slowperiod}_{signalperiod}"

        return (
            result[macd_col].values,
            result[signal_col].values,
            result[hist_col].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def macdfix(
        data: Union[np.ndarray, pd.Series], signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, 26 + signalperiod)
        # pandas-ta does not have macdfix; approximate by standard macd with fixed periods
        result = ta.macd(series, fast=12, slow=26, signal=signalperiod)
        macd_col = f"MACD_12_26_{signalperiod}"
        signal_col = f"MACDs_12_26_{signalperiod}"
        hist_col = f"MACDh_12_26_{signalperiod}"
        return (
            result[macd_col].values,
            result[signal_col].values,
            result[hist_col].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def stoch(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k_period: int | None = None,
        d_period: int | None = None,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクス"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        # 互換パラメータ適用
        k_eff = k_period if k_period is not None else k
        d_eff = d_period if d_period is not None else d

        validate_series_data(high_series, k_eff)
        validate_series_data(low_series, k_eff)
        validate_series_data(close_series, k_eff)

        result = ta.stoch(
            high=high_series,
            low=low_series,
            close=close_series,
            k=k_eff,
            d=d_eff,
            smooth_k=smooth_k,
        )
        # 入力と同じインデックスに合わせる（長さを厳密に一致させる）
        result = result.reindex(close_series.index)

        k_col = f"STOCHk_{k_eff}_{d_eff}_{smooth_k}"
        d_col = f"STOCHd_{k_eff}_{d_eff}_{smooth_k}"

        return (result[k_col].values, result[d_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def stochf(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 5,
        d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """高速ストキャスティクス"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, k)
        validate_series_data(low_series, k)
        validate_series_data(close_series, k)

        # pandas-taのバージョンによりstochfがない場合があるためstochで代替
        try:
            result = ta.stochf(
                high=high_series, low=low_series, close=close_series, k=k, d=d
            )
            return result[f"STOCHFk_{k}_{d}"].values, result[f"STOCHFd_{k}_{d}"].values
        except AttributeError:
            # フォールバック: stochのfastk/fastdを使用
            result = ta.stoch(
                high=high_series,
                low=low_series,
                close=close_series,
                k=k,
                d=d,
                smooth_k=3,
            )
            # pandas-taの列名の相違に対応
            result = result.reindex(close_series.index)
            k_candidates = [
                f"STOCHk_{k}_{d}_1",
                "STOCHk_14_3_3",
                f"STOCHk_{k}",
                "fastk",
            ]
            d_candidates = [
                f"STOCHd_{k}_{d}_1",
                "STOCHd_14_3_3",
                f"STOCHd_{d}",
                "fastd",
            ]
            k_col = next((c for c in k_candidates if c in result.columns), None)
            d_col = next((c for c in d_candidates if c in result.columns), None)
            if k_col is None or d_col is None:
                cols = list(result.columns)
                if len(cols) >= 2:
                    k_col, d_col = cols[0], cols[1]
            if k_col is None or d_col is None:
                raise PandasTAError("stochf フォールバック列が見つかりません")
            return (result[k_col].values, result[d_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def stochrsi(
        data: Union[np.ndarray, pd.Series],
        length: int = 14,
        k: int = 5,
        d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ストキャスティクスRSI"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length + k + d)
        # pandas-taの列名が異なる場合に対応
        result = ta.stochrsi(series, length=length, k=k, d=d)
        # 列名候補を拡張
        k_candidates = [
            f"STOCHRSIk_{length}_{k}_{d}",
            f"STOCHRSI_k_{length}_{k}_{d}",
            f"STOCHRSI{k}_{d}_{length}",
            "STOCHRSIk_14_5_3",
            "fastk",
        ]
        d_candidates = [
            f"STOCHRSId_{length}_{k}_{d}",
            f"STOCHRSI_d_{length}_{k}_{d}",
            f"STOCHRSI{k}_{d}_{length}_d",
            "STOCHRSId_14_5_3",
            "fastd",
        ]
        k_col = next((c for c in k_candidates if c in result.columns), None)
        d_col = next((c for c in d_candidates if c in result.columns), None)
        if k_col is None:
            k_col = next((c for c in result.columns if "k" in c.lower()), None)
        if d_col is None:
            d_col = next(
                (c for c in result.columns if "d" in c.lower() and c != k_col), None
            )
        if k_col is None or d_col is None:
            # 最後の保険：先頭2列
            cols = list(result.columns)
            if len(cols) >= 2:
                k_col, d_col = cols[0], cols[1]
        if k_col is None or d_col is None:
            raise PandasTAError("stochrsi 出力列が見つかりません")
        return (result[k_col].values, result[d_col].values)

    @staticmethod
    @handle_pandas_ta_errors
    def willr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ウィリアムズ%R"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.willr(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cci(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 20,
    ) -> np.ndarray:
        """商品チャネル指数"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.cci(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def cmo(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """チェンジモメンタムオシレーター"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.cmo(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def roc(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率"""
        series = ensure_series_minimal_conversion(data)
        validate_indicator_parameters(length)
        validate_series_data(series, length)
        result = ta.roc(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocp(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（%）"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        # pandas-ta v0.3.x ではrocp未提供。rocをdiff=Trueで代替
        try:
            result = ta.rocp(series, length=length)
            return result.values
        except AttributeError:
            result = ta.roc(series, length=length, diff=True)
            return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocr(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率）"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        # pandas-ta v0.3.x ではrocr未提供。rocの比率化で代替
        try:
            result = ta.rocr(series, length=length)
            return result.values
        except AttributeError:
            # ratio: (price / price.shift(length))
            shifted = series.shift(length)
            ratio = series / shifted
            return ratio.values

    @staticmethod
    @handle_pandas_ta_errors
    def rocr100(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """変化率（比率100スケール）"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        # pandas-ta v0.3.x ではrocr未提供。比率×100で代替
        try:
            result = ta.rocr(series, length=length, scalar=100)
            return result.values
        except AttributeError:
            shifted = series.shift(length)
            ratio = (series / shifted) * 100
            return ratio.values

    @staticmethod
    @handle_pandas_ta_errors
    def mom(data: Union[np.ndarray, pd.Series], length: int = 10) -> np.ndarray:
        """モメンタム"""
        series = ensure_series_minimal_conversion(data)
        validate_indicator_parameters(length)
        validate_series_data(series, length)
        result = ta.mom(series, length=length)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def adx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """平均方向性指数"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        return result[f"ADX_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def adxr(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """ADX評価"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)

        result = ta.adx(
            high=high_series, low=low_series, close=close_series, length=length
        )
        # pandas-taのadxrが別関数として提供されている場合はそちらを使用
        try:
            result_adxr = ta.adxr(
                high=high_series, low=low_series, close=close_series, length=length
            )
            return result_adxr.values
        except Exception:
            # フォールバック: ADXの列命名差異に対応
            candidate_cols = [f"ADXR_{length}", f"ADXRr_{length}", f"ADXR_{length}_0"]
            for col in candidate_cols:
                if col in result.columns:
                    return result[col].values
            # 最低限ADXを返す
            return result[f"ADX_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def aroon(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アルーン"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result[f"AROOND_{length}"].values, result[f"AROONU_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def aroonosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """アルーンオシレーター"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)

        result = ta.aroon(high=high_series, low=low_series, length=length)
        return result[f"AROONOSC_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def dx(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """Directional Movement Index wrapper (DX)"""
        # pandas-ta returns DX as part of adx; extract DX
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        validate_series_data(high_s, length)
        validate_series_data(low_s, length)
        validate_series_data(close_s, length)
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

    @staticmethod
    @handle_pandas_ta_errors
    def mfi(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> np.ndarray:
        """マネーフローインデックス"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)
        volume_series = ensure_series_minimal_conversion(volume)

        validate_series_data(high_series, length)
        validate_series_data(low_series, length)
        validate_series_data(close_series, length)
        validate_series_data(volume_series, length)

        result = ta.mfi(
            high=high_series,
            low=low_series,
            close=close_series,
            volume=volume_series,
            length=length,
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def plus_di(high, low, close, length: int = 14) -> np.ndarray:
        """Plus Directional Indicator (DI)"""
        result = ta.adx(
            high=ensure_series_minimal_conversion(high),
            low=ensure_series_minimal_conversion(low),
            close=ensure_series_minimal_conversion(close),
            length=length,
        )
        return result[f"DMP_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def minus_di(high, low, close, length: int = 14) -> np.ndarray:
        """Minus Directional Indicator (DI)"""
        result = ta.adx(
            high=ensure_series_minimal_conversion(high),
            low=ensure_series_minimal_conversion(low),
            close=ensure_series_minimal_conversion(close),
            length=length,
        )
        return result[f"DMN_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def plus_dm(high, low, length: int = 14) -> np.ndarray:
        """Plus Directional Movement (DM)"""
        result = ta.dm(
            high=ensure_series_minimal_conversion(high),
            low=ensure_series_minimal_conversion(low),
            length=length,
        )
        return result[f"DMP_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def minus_dm(high, low, length: int = 14) -> np.ndarray:
        """Minus Directional Movement (DM)"""
        result = ta.dm(
            high=ensure_series_minimal_conversion(high),
            low=ensure_series_minimal_conversion(low),
            length=length,
        )
        return result[f"DMN_{length}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def ppo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Percentage Price Oscillator"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, slow + signal)
        result = ta.ppo(series, fast=fast, slow=slow, signal=signal)
        return (
            result[f"PPO_{fast}_{slow}_{signal}"].values,
            result[f"PPOh_{fast}_{slow}_{signal}"].values,
            result[f"PPOs_{fast}_{slow}_{signal}"].values,
        )

    @staticmethod
    @handle_pandas_ta_errors
    def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """TRIX"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length * 3)
        result = ta.trix(series, length=length)
        return result[f"TRIX_{length}_9"].values

    @staticmethod
    @handle_pandas_ta_errors
    def ultosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        fast: int = 7,
        medium: int = 14,
        slow: int = 28,
    ) -> np.ndarray:
        """Ultimate Oscillator"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(high_series, slow)
        validate_series_data(low_series, slow)
        validate_series_data(close_series, slow)

        result = ta.uo(
            high=high_series,
            low=low_series,
            close=close_series,
            fast=fast,
            medium=medium,
            slow=slow,
        )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def bop(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Balance of Power"""
        open_series = ensure_series_minimal_conversion(open_)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        # pandas-taのbopはscalar引数を取る場合がある
        try:
            result = ta.bop(
                open_=open_series,
                high=high_series,
                low=low_series,
                close=close_series,
                scalar=1,
            )
        except TypeError:
            result = ta.bop(
                open_=open_series,
                high=high_series,
                low=low_series,
                close=close_series,
            )
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def apo(
        data: Union[np.ndarray, pd.Series], fast: int = 12, slow: int = 26
    ) -> np.ndarray:
        """Absolute Price Oscillator"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, slow)
        result = ta.apo(series, fast=fast, slow=slow)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def ao(
        high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Awesome Oscillator"""
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        validate_series_data(high_s, 5)
        validate_series_data(low_s, 5)
        result = ta.ao(high=high_s, low=low_s)
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def kdj(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k: int = 14,
        d: int = 3,
        j_scalar: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """KDJ: pandas-taではstochから計算"""
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        validate_series_data(close_s, k + d)
        stoch_df = ta.stoch(high=high_s, low=low_s, close=close_s, k=k, d=d, smooth_k=3)
        # pandas-taが先頭NaN区間で短縮する場合に備え、インデックスを合わせる
        stoch_df = stoch_df.reindex(close_s.index)
        # 列検出
        k_col = next((c for c in stoch_df.columns if "k" in c.lower()), None)
        d_col = next((c for c in stoch_df.columns if "d" in c.lower()), None)
        if k_col is None or d_col is None:
            raise PandasTAError("KDJの元となるstoch列が見つかりません")
        k_vals = stoch_df[k_col].values
        d_vals = stoch_df[d_col].values
        j_vals = j_scalar * k_vals - 2 * d_vals
        return k_vals, d_vals, j_vals

    @staticmethod
    @handle_pandas_ta_errors
    def rvgi(
        open_: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Relative Vigor Index"""
        o = ensure_series_minimal_conversion(open_)
        h = ensure_series_minimal_conversion(high)
        l = ensure_series_minimal_conversion(low)
        c = ensure_series_minimal_conversion(close)
        validate_series_data(c, length + 1)
        df = ta.rvgi(open_=o, high=h, low=l, close=c, length=length)
        r_col = next(
            (c for c in df.columns if c.lower().endswith("rvi")), df.columns[0]
        )
        s_col = next(
            (c for c in df.columns if c.lower().endswith("signal")), df.columns[-1]
        )
        return df[r_col].values, df[s_col].values

    @staticmethod
    @handle_pandas_ta_errors
    def qqe(data: Union[np.ndarray, pd.Series], length: int = 14) -> np.ndarray:
        """Qualitative Quantitative Estimation"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, length + 1)
        df = ta.qqe(s, length=length)
        # 単列
        return df.values if hasattr(df, "values") else np.asarray(df)

    @staticmethod
    @handle_pandas_ta_errors
    def smi(
        data: Union[np.ndarray, pd.Series],
        fast: int = 13,
        slow: int = 25,
        signal: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Momentum Index"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, fast + slow + signal)
        df = ta.smi(s, fast=fast, slow=slow, signal=signal)
        # 2列想定
        cols = list(df.columns)
        return df[cols[0]].values, df[cols[1]].values

    @staticmethod
    @handle_pandas_ta_errors
    def kst(
        data: Union[np.ndarray, pd.Series],
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        n1: int = 10,
        n2: int = 10,
        n3: int = 10,
        n4: int = 15,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Know Sure Thing"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, max(r1, r2, r3, r4, signal))
        df = ta.kst(
            s, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, signal=signal
        )
        k_col = next((c for c in df.columns if c.lower().endswith("kst")), None)
        s_col = next((c for c in df.columns if c.lower().endswith("signal")), None)
        if k_col is None or s_col is None:
            cols = list(df.columns)
            return df[cols[0]].values, df[cols[-1]].values
        return df[k_col].values, df[s_col].values

    @staticmethod
    @handle_pandas_ta_errors
    def stc(
        data: Union[np.ndarray, pd.Series],
        tclength: int = 10,
        fast: int = 23,
        slow: int = 50,
        factor: float = 0.5,
    ) -> np.ndarray:
        """Schaff Trend Cycle"""
        s = ensure_series_minimal_conversion(data)
        validate_series_data(s, slow + tclength)
        df = ta.stc(s, tclength=tclength, fast=fast, slow=slow, factor=factor)
        return df.values if hasattr(df, "values") else np.asarray(df)