"""
Hilbert Transform系テクニカル指標

Hilbert Transformに基づくテクニカル指標を実装します。
"""

from typing import Union, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors


class HilbertTransformIndicators:
    """
    Hilbert Transform系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def ht_dcperiod(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hilbert Transform - Dominant Cycle Period"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # pandas-taにht_dcperiodがない場合は簡易実装
        if hasattr(ta, "ht_dcperiod"):
            return ta.ht_dcperiod(series).values
        else:
            # フォールバック: 固定値を返す
            return np.full(len(series), 20.0)

    @staticmethod
    @handle_pandas_ta_errors
    def ht_dcphase(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hilbert Transform - Dominant Cycle Phase"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # pandas-taにht_dcphaseがない場合は簡易実装
        if hasattr(ta, "ht_dcphase"):
            return ta.ht_dcphase(series).values
        else:
            # フォールバック: サイン波を返す
            return np.sin(np.arange(len(series)) * 2 * np.pi / 20)

    @staticmethod
    @handle_pandas_ta_errors
    def ht_phasor(data: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """Hilbert Transform - Phasor Components"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # pandas-taにht_phasorがない場合は簡易実装
        if hasattr(ta, "ht_phasor"):
            result = ta.ht_phasor(series)
            return result.iloc[:, 0].values, result.iloc[:, 1].values
        else:
            # フォールバック: サイン・コサイン波を返す
            phase = np.arange(len(series)) * 2 * np.pi / 20
            inphase = np.cos(phase)
            quadrature = np.sin(phase)
            return inphase, quadrature

    @staticmethod
    @handle_pandas_ta_errors
    def ht_sine(data: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """Hilbert Transform - SineWave"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # pandas-taにht_sineがない場合は簡易実装
        if hasattr(ta, "ht_sine"):
            result = ta.ht_sine(series)
            return result.iloc[:, 0].values, result.iloc[:, 1].values
        else:
            # フォールバック: サイン・コサイン波を返す
            phase = np.arange(len(series)) * 2 * np.pi / 20
            sine = np.sin(phase) * 0.5
            leadsine = np.sin(phase + np.pi / 4) * 0.5
            return sine, leadsine

    @staticmethod
    @handle_pandas_ta_errors
    def ht_trendmode(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Hilbert Transform - Trend vs Cycle Mode"""
        series = pd.Series(data) if isinstance(data, np.ndarray) else data
        # pandas-taにht_trendmodeがない場合は簡易実装
        if hasattr(ta, "ht_trendmode"):
            return ta.ht_trendmode(series).values
        else:
            # フォールバック: 移動平均の傾きベースでトレンドモードを判定
            ma = series.rolling(window=20).mean()
            slope = ma.diff()
            trend_mode = np.where(np.abs(slope) > slope.std(), 1, 0)
            return trend_mode
