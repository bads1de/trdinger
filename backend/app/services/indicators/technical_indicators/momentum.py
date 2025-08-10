"""
モメンタム系テクニカル指標（pandas-ta移行版）

このモジュールはpandas-taライブラリを使用し、
backtesting.pyとの完全な互換性を提供します。
numpy配列ベースのインターフェースを維持しています。
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
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMP_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("PLUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def minus_di(high, low, close, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        close_s = ensure_series_minimal_conversion(close)
        result = ta.adx(high=high_s, low=low_s, close=close_s, length=length)
        col = f"DMN_{length}"
        if col in result.columns:
            return result[col].values
        raise PandasTAError("MINUS_DI not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def plus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        # pandas-ta dm returns DMP and DMN columns
        cols = [c for c in result.columns if c.startswith("DMP_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("PLUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def minus_dm(high, low, length: int = 14) -> np.ndarray:
        high_s = ensure_series_minimal_conversion(high)
        low_s = ensure_series_minimal_conversion(low)
        result = ta.dm(high=high_s, low=low_s, length=length)
        cols = [c for c in result.columns if c.startswith("DMN_")]
        if cols:
            return result[cols[0]].values
        raise PandasTAError("MINUS_DM not available in this pandas-ta version")

    @staticmethod
    @handle_pandas_ta_errors
    def ppo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> np.ndarray:
        """パーセンテージ価格オシレーター"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, slow)
        result = ta.ppo(series, fast=fast, slow=slow, signal=signal)
        return result[f"PPO_{fast}_{slow}_{signal}"].values

    @staticmethod
    @handle_pandas_ta_errors
    def trix(data: Union[np.ndarray, pd.Series], length: int = 30) -> np.ndarray:
        """TRIX"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, length)
        result = ta.trix(series, length=length)
        return result[f"TRIX_{length}_9"].values

    @staticmethod
    @handle_pandas_ta_errors
    def ultosc(
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        period1: int | None = None,
        period2: int | None = None,
        period3: int | None = None,
        timeperiod1: int = 7,
        timeperiod2: int = 14,
        timeperiod3: int = 28,
    ) -> np.ndarray:
        """アルティメットオシレーター"""
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        # パラメータ名互換（period1/2/3 または timeperiod1/2/3）
        s = period1 if period1 is not None else timeperiod1
        m = period2 if period2 is not None else timeperiod2
        l = period3 if period3 is not None else timeperiod3

        # 最大期間でバリデーション
        max_len = max(s, m, l)
        validate_series_data(high_series, max_len)
        validate_series_data(low_series, max_len)
        validate_series_data(close_series, max_len)

        # pandas-ta に ultosc が無い場合は手動実装
        if hasattr(ta, "ultosc"):
            result = ta.ultosc(
                high=high_series,
                low=low_series,
                close=close_series,
                s=s,
                m=m,
                l=l,
            )
            return result.values
        else:
            pc = close_series.shift(1)
            min_l_pc = pd.concat([low_series, pc], axis=1).min(axis=1)
            max_h_pc = pd.concat([high_series, pc], axis=1).max(axis=1)
            bp = close_series - min_l_pc
            tr = max_h_pc - min_l_pc

            sum_bp_s = bp.rolling(window=s).sum()
            sum_tr_s = tr.rolling(window=s).sum()
            sum_bp_m = bp.rolling(window=m).sum()
            sum_tr_m = tr.rolling(window=m).sum()
            sum_bp_l = bp.rolling(window=l).sum()
            sum_tr_l = tr.rolling(window=l).sum()

            avg_s = sum_bp_s / sum_tr_s
            avg_m = sum_bp_m / sum_tr_m
            avg_l = sum_bp_l / sum_tr_l

            uo = 100.0 * (4 * avg_s + 2 * avg_m + 1 * avg_l) / 7.0
            return uo.to_numpy()

    @staticmethod
    @handle_pandas_ta_errors
    def bop(
        open_data: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """バランスオブパワー"""
        open_series = ensure_series_minimal_conversion(open_data)
        high_series = ensure_series_minimal_conversion(high)
        low_series = ensure_series_minimal_conversion(low)
        close_series = ensure_series_minimal_conversion(close)

        validate_series_data(open_series, 1)
        validate_series_data(high_series, 1)
        validate_series_data(low_series, 1)
        validate_series_data(close_series, 1)

        # pandas-ta v0.3.14+ は引数名 open_ を使用
        if hasattr(ta, "bop"):
            try:
                result = ta.bop(
                    open_=open_series,
                    high=high_series,
                    low=low_series,
                    close=close_series,
                )
            except TypeError:
                # バージョン差異で open 引数名の場合
                result = ta.bop(
                    open=open_series,
                    high=high_series,
                    low=low_series,
                    close=close_series,
                )
        else:
            raise PandasTAError("pandas-ta に b op が存在しません")
        return result.values

    @staticmethod
    @handle_pandas_ta_errors
    def apo(
        data: Union[np.ndarray, pd.Series],
        fast: int = 12,
        slow: int = 26,
    ) -> np.ndarray:
        """アブソリュートプライスオシレーター"""
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, slow)
        result = ta.apo(series, fast=fast, slow=slow)
        return result.values
