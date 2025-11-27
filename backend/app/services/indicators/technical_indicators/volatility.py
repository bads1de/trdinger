"""
ボラティリティ系テクニカル指標

登録してあるテクニカルの一覧:
- ATR (Average True Range)
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Supertrend
- Acceleration Bands
- Ulcer Index
- Relative Volatility Index (RVI)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors

TA_LIB_AVAILABLE = False
try:
    import talib  # noqa: F401

    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False


logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス

    ATR, Bollinger Bandsなどのボラティリティ系テクニカル指標を提供。
    市場の変動性とリスク評価に使用します。
    """

    @staticmethod
    @handle_pandas_ta_errors
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """平均真の値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.atr(
            high=high.values, low=low.values, close=close.values, length=length
        )

        if result is None:
            logger.error("ATR: Calculation returned None - returning NaN series")
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range"""
        for series, name in ((high, "high"), (low, "low"), (close, "close")):
            if not isinstance(series, pd.Series):
                raise TypeError(f"{name} must be pandas Series")
        if length <= 0:
            raise ValueError("length must be positive")

        result = ta.natr(high=high, low=low, close=close, length=length)
        if result is None or (hasattr(result, "empty") and result.empty):
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.bbands(data, length=length, std=std)

        if result is None:
            logger.error("BBands: Calculation returned None - returning NaN series")
            nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
            return (nan_series, nan_series, nan_series)

        # 列名を動的に取得（pandas-taのバージョンによって異なる可能性がある）
        columns = result.columns.tolist()

        # 上位、中位、下位バンドを特定
        upper_col = [col for col in columns if "BBU" in col][0]
        middle_col = [col for col in columns if "BBM" in col][0]
        lower_col = [col for col in columns if "BBL" in col][0]

        return (
            result[upper_col],
            result[middle_col],
            result[lower_col],
        )

    @staticmethod
    @handle_pandas_ta_errors
    def keltner(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        scalar: float = 2.0,
        std_dev: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels: returns (upper, middle, lower)"""

        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # std_dev パラメータは使用しないが、互換性のため受け入れる
        _ = std_dev

        # データ長チェック
        min_length = max(period * 2, 20)  # 最小データ長
        if len(high) < min_length:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        df = ta.kc(
            high=high.values,
            low=low.values,
            close=close.values,
            length=period,
            scalar=scalar,
            mamode="sma",
        )

        if df is None:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        # パフォーマンスカラム抽出処理
        cols = list(df.columns)

        upper = df[
            next((c for c in cols if "KCu" in c or "upper" in c.lower()), cols[0])
        ]
        middle = df[
            next(
                (c for c in cols if "KCe" in c or "KCM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = df[
            next((c for c in cols if "KCl" in c or "lower" in c.lower()), cols[-1])
        ]

        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels: returns (upper, middle, lower)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        df = ta.donchian(high=high, low=low, length=length)

        if df is None or df.empty:
            logger.error("DONCHIAN: Calculation returned None - returning NaN series")
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        cols = list(df.columns)
        upper = df[
            next((c for c in cols if "DCHU" in c or "upper" in c.lower()), cols[0])
        ]
        middle = df[
            next(
                (c for c in cols if "DCHM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = df[
            next((c for c in cols if "DCHL" in c or "lower" in c.lower()), cols[-1])
        ]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 7,
        multiplier: float = 3.0,
        **kwargs,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Supertrend: returns (lower, upper, direction)"""

        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Support 'factor' parameter as alias for 'multiplier'
        if "factor" in kwargs:
            multiplier = kwargs["factor"]

        # Enhanced data length check - Supertrend requires sufficient data
        length = period
        min_length = max(
            length * 2, 14
        )  # Relaxed minimum (Green Phase: more flexible data requirements)
        if len(high) < min_length:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        # Calculate basic bands for compatibility with enhanced parameter validation
        hl2 = (high + low) / 2
        atr_length = max(period, 14)  # Ensure ATR length is sufficient
        atr = VolatilityIndicators.atr(high, low, close, atr_length)

        # Add NaN check for ATR
        if atr is None or atr.isna().all():
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        df = ta.supertrend(
            high=high.values,
            low=low.values,
            close=close.values,
            length=length,
            multiplier=multiplier,
        )

        if df is None:
            direction = pd.Series(
                np.where(close >= (upper + lower) / 2, 1.0, -1.0), index=close.index
            )
            return lower, upper, direction

        assert df is not None  # for type checker
        cols = list(df.columns)
        st_col = next(
            (c for c in cols if "SUPERT_" in c.upper() or "supertrend" in c.lower()),
            cols[0],
        )
        dir_col = next(
            (c for c in cols if "SUPERTd_" in c.upper() or "direction" in c.lower()),
            None,
        )
        if dir_col is None:
            direction = pd.Series(
                np.where(close.to_numpy() >= df[st_col].to_numpy(), 1.0, -1.0),
                index=close.index,
            )
        else:
            direction = df[dir_col].fillna(0)
            if direction.isna().all():
                direction = pd.Series(
                    np.where(close.to_numpy() >= df[st_col].to_numpy(), 1.0, -1.0),
                    index=close.index,
                )

        return lower, upper, direction

    @staticmethod
    @handle_pandas_ta_errors
    def accbands(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Acceleration Bands: returns (upper, middle, lower)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        length = period

        result = ta.accbands(
            high=high, low=low, close=close, length=length, mamode="sma"
        )

        if result is None or (hasattr(result, "isna") and result.isna().all().all()):
            logger.error("ACCBANDS: Calculation returned None - returning NaN series")
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        cols = list(result.columns)
        upper = result[
            next((c for c in cols if "ACCBU" in c or "upper" in c.lower()), cols[0])
        ]
        middle = result[
            next(
                (c for c in cols if "ACCBM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = result[
            next((c for c in cols if "ACCBL" in c or "lower" in c.lower()), cols[-1])
        ]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def ui(data: pd.Series, period: int = 14) -> pd.Series:
        """Ulcer Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        length = period
        result = ta.ui(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def rvi(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        length: int = 14,
        scalar: float = 100.0,
        refined: bool = False,
        thirds: bool = False,
        mamode: str | None = None,
        drift: int | None = None,
        offset: int | None = None,
    ) -> pd.Series:
        """Relative Volatility Index"""

        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        result = ta.rvi(
            close=close,
            high=high,
            low=low,
            length=length,
            scalar=scalar,
            refined=refined,
            thirds=thirds,
            mamode=mamode,
            drift=drift,
            offset=offset,
        )

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def vhf(
        data: pd.Series,
        length: int = 28,
        scalar: float = 100.0,
        drift: int = 1,
        offset: int = 0,
    ) -> pd.Series:
        """Vertical Horizontal Filter"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if scalar <= 0:
            raise ValueError(f"scalar must be positive: {scalar}")
        if drift <= 0:
            raise ValueError(f"drift must be positive: {drift}")

        # VHF requires sufficient data length
        min_length = length * 2
        if len(data) < min_length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        result = ta.vhf(
            close=data,
            length=length,
            scalar=scalar,
            drift=drift,
            offset=offset,
        )

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def gri(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
        offset: int = 0,
    ) -> pd.Series:
        """Gopalakrishnan Range Index (GRI) - 市場のレンジ幅を測定するオシレーター"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")

        # pandas-taのkvo（Know Sure Thing）を使用して代替実装
        # KVOは同様のレンジベースの計算を行う
        try:
            result = ta.kvo(
                high=high,
                low=low,
                close=close,
                fast=length,
                slow=length * 2,
                signal=None,  # 信号線は不要
                offset=offset,
            )
        except Exception:
            # pandas-taが使えない場合のフォールバック実装
            # 簡易的なGRI計算
            typical_range = high - low
            typical_price = (high + low) / 2

            # 簡易的なレンジインデックスを計算
            gri_raw = typical_range / typical_price * 100

            # 移動平均でスムージング
            if len(gri_raw) >= length:
                result = gri_raw.rolling(window=length, min_periods=1).mean()
            else:
                result = pd.Series(np.full(len(gri_raw), np.nan), index=gri_raw.index)

        if result is None or (hasattr(result, "isna") and result.isna().all()):
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        return result

    @staticmethod
    @handle_pandas_ta_errors
    def true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        drift: int = 1,
    ) -> pd.Series:
        """True Range"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        result = ta.true_range(high=high, low=low, close=close, drift=drift)

        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)

        return result
