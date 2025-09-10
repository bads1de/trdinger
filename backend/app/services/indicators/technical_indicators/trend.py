"""
トレンド系テクニカル指標

登録してあるテクニカルの一覧:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
- T3 (Tillson's T3 Moving Average)
- KAMA (Kaufman's Adaptive Moving Average)
- SAR (Parabolic SAR)
"""

from typing import Tuple, Optional
import warnings
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

from ..utils import handle_pandas_ta_errors
from ..config.indicator_config import IndicatorResultType

# PandasのSeries位置アクセス警告を抑制 (pandas-taとの互換性のため)
warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated",
    category=FutureWarning,
)


class TrendIndicators:
    """
    トレンド系指標クラス
    """

    @staticmethod
    @handle_pandas_ta_errors
    def sma(data: pd.Series, period: int) -> pd.Series:
        """単純移動平均（軽量エラーハンドリング付き）"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if period <= 0:
            raise ValueError(f"period must be positive: {period}")

        # 基本的な入力検証
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)
            # raise ValueError("データが空です")

        # データチェック: NaNが多すぎる場合
        if data.isna().sum() > len(data) * 0.5:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # データ長チェック
        if len(data) < period:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        if period == 1:
            return data

        result = data.rolling(window=period).mean()

        # 結果検証（重要な異常ケースのみ）
        if result.isna().all():
            return pd.Series(np.full(len(data), np.nan), index=data.index)
            # raise ValueError("計算結果が全てNaNです")

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def ema(data: pd.Series, length: int) -> pd.Series:
        """指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"period must be positive: {length}")

        # データチェック
        if len(data) == 0:
            return pd.Series(np.full(0, np.nan), index=data.index)
        if data.isna().sum() > len(data) * 0.5:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # 第一優先: pandas-ta
        try:
            result = ta.ema(data, window=length)
            if result is None:
                pass
            elif not isinstance(result, pd.Series):
                pass
            else:
                # 全てNaNではなく、有用な値が含まれている場合のみ使用
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length - 1
                if start_idx < len(result):
                    valid_from_start = result.iloc[start_idx:]
                    all_nan_from_start = valid_from_start.isna().all()
                    if not all_nan_from_start:
                        return result
                    else:
                        pass
                else:
                    pass
        except Exception as e:
            pass

        # フォールバック実装: numpyベース -> pandas.Seriesに変更
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA計算用変数
        alpha = 2.0 / (length + 1)

        ema_values = np.full(len(data), np.nan, dtype=float)
        if len(data) >= length:
            initial_sma = data.iloc[:length].mean()
            ema_values[length - 1] = initial_sma # 初期値はSMA
        else:
            pass

        # ループでEMA計算
        for i in range(length, len(data)):
            current_price = data.iloc[i]
            prev_ema = ema_values[i - 1]
            new_ema = alpha * current_price + (1 - alpha) * prev_ema
            ema_values[i] = new_ema

            # NaNチェック
            if np.isnan(new_ema):
                pass

        result_series = pd.Series(ema_values, index=data.index)
        return result_series

    @staticmethod
    @handle_pandas_ta_errors
    def wma(
        data: pd.Series = None,
        length: int = 14,
        close: pd.Series = None,
    ) -> pd.Series:
        """加重移動平均"""
        # dataが提供されない場合はcloseを使用
        if data is None and close is not None:
            data = close
        elif data is None:
            raise ValueError("Either 'data' or 'close' must be provided")

        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        return ta.wma(data, window=length)

    @staticmethod
    @handle_pandas_ta_errors
    def dema(data: pd.Series, length: int) -> pd.Series:
        """二重指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.dema(data, window=length)
            if result is not None:
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 2 - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算 (DEMA = 2*EMA1 - EMA2)
        if len(data) < length * 2:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA1, EMA2の計算
        ema1 = TrendIndicators.ema(data, length)
        ema2 = TrendIndicators.ema(ema1, length)

        # DEMA = 2*EMA1 - EMA2
        return 2 * ema1 - ema2

    @staticmethod
    @handle_pandas_ta_errors
    def tema(data: pd.Series, length: int) -> pd.Series:
        """三重指数移動平均"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        # 第一優先: pandas-ta
        try:
            result = ta.tema(data, window=length)
            if result is not None:
                # 最初のlength位置以降に有効な値がある場合
                start_idx = length * 3 - 1
                if (
                    start_idx < len(result)
                    and not np.isnan(result.iloc[start_idx:]).all()
                ):
                    return result
        except Exception:
            pass

        # フォールバック実装: 多段EMA計算
        if len(data) < length * 3:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # EMA1, EMA2, EMA3の計算
        ema1 = TrendIndicators.ema(data, length)
        ema2 = TrendIndicators.ema(ema1, length)
        ema3 = TrendIndicators.ema(ema2, length)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    @handle_pandas_ta_errors
    def t3(data: pd.Series, length: int = 5, a: float = 0.7) -> pd.Series:
        """T3移動平均 with enhanced fallback"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        if not (0.0 <= a <= 1.0):
            raise ValueError(f"a must be between 0.0 and 1.0: {a}")

        # 第一優先: pandas-ta
        try:
            result = ta.t3(data, window=length, a=a)
            if result is not None and not result.isna().all():
                return result
        except Exception:
            pass

        # 強化フォールバック実装: T3 = a^3*EMA1 + 3*a^2*(1-a)*EMA2 + 3*a*(1-a)^2*EMA3 + (1-a)^3*EMA4
        if len(data) < length * 4:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        try:
            # 多段EMA計算
            ema1 = TrendIndicators.ema(data, length)
            ema2 = TrendIndicators.ema(ema1, length)
            ema3 = TrendIndicators.ema(ema2, length)
            ema4 = TrendIndicators.ema(ema3, length)

            # T3計算
            t3_result = (
                a * a * ema1
                + 3 * a * a * (1 - a) * ema2
                + 3 * a * (1 - a) * (1 - a) * ema3
                + (1 - a) * (1 - a) * (1 - a) * ema4
            )

            return t3_result

        except Exception:
            # 最終フォールバック: EMAベース近似
            return TrendIndicators.ema(data, length)

    @staticmethod
    @handle_pandas_ta_errors
    def kama(data: pd.Series, length: int = 30) -> pd.Series:
        """カウフマン適応移動平均"""
        # 第一優先: pandas-ta
        try:
            result = ta.kama(data, window=length)
            if result is not None:
                return result
        except Exception:
            pass

        # フォールバック: EMAで代替
        return TrendIndicators.ema(data, length)

    @staticmethod
    @handle_pandas_ta_errors
    def sar(
        high: pd.Series,
        low: pd.Series,
        af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """パラボリックSAR"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.psar(high=high, low=low, af0=af, af=af, max_af=max_af)
        # PSARl と PSARs を結合
        psar_long = result[f"PSARl_{af}_{max_af}"]
        psar_short = result[f"PSARs_{af}_{max_af}"]
        return psar_long.fillna(psar_short)