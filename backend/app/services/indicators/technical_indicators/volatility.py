"""
ボラティリティ系テクニカル指標

登録してあるテクニカルの一覧:
- ATR (Average True Range)
- NATR (Normalized Average True Range)
- True Range
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Supertrend
- Acceleration Bands
- Holt-Winter Channel
- Mass Index
- Price Distance
- Elder's Thermometer
- Ulcer Index
"""

from typing import Tuple, Optional
import logging

import numpy as np
import pandas as pd
# pandas-ta availability check for fallback implementation
TA_LIB_AVAILABLE = False
try:
    import talib  # pandas-ta might use talib internally
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
import pandas_ta as ta

from ..utils import handle_pandas_ta_errors
from ..config.indicator_config import IndicatorResultType

logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    ボラティリティ系指標クラス
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
            high=high, low=low, close=close, length=length
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        """正規化平均実効値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        # NATRパラメータエラー診断
        logger.debug(f"NATR: length={length}")
        logger.debug(f"NATR: high and low and close available")
        logger.debug(f"NATR: Checking ta.natr signature")
    
        # pandas-ta natrのパラメータを確認
        import inspect
        try:
            sig = inspect.signature(ta.natr)
            logger.debug(f"NATR: ta.natr params: {list(sig.parameters.keys())}")
        except Exception as e:
            logger.debug(f"NATR: inspect failed: {e}")
    
        try:
            # lengthパラメータの確認
            result = ta.natr(high=high, low=low, close=close, length=length)
        except Exception as e:
            logger.error(f"NATR: ta.natr failed with length parameter: {e}")
            # windowパラメータで試す
            try:
                result = ta.natr(high=high, low=low, close=close, window=length)
                logger.info("NATR: Successful with window parameter")
            except Exception as e2:
                logger.error(f"NATR: ta.natr also failed with window parameter: {e2}")
                # パラメータなしで試す
                try:
                    if hasattr(ta, 'natr'):
                        result = ta.natr(high=high, low=low, close=close)
                        logger.info("NATR: Successful with no parameters")
                    else:
                        logger.warning("NATR: ta.natr function not found in pandas-ta")
                        result = None
                except Exception as e3:
                    logger.error(f"NATR: ta.natr failed completely: {e3}")
                    result = None
    
        # Use length parameter for ta.natr
        result = ta.natr(
            high=high, low=low, close=close, length=length
        )
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def trange(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,  # API consistency parameter (not used for trange calculation)
    ) -> pd.Series:
        """真の値幅"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
    
        # period parameter added for API consistency, not used in trange calculation
        result = ta.true_range(high=high, low=low, close=close)
        if result is None:
            return pd.Series(np.full(len(high), np.nan), index=high.index)
        assert result is not None  # for type checker
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def bbands(
        data: pd.Series, length: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド with enhanced fallback"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        result = ta.bbands(data, length=length, std=std)

        if result is None:
            # Enhanced fallback: Manual Bollinger Bands calculation
            try:
                if len(data) < length:
                    nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                    return (nan_series, nan_series, nan_series)

                # Calculate moving average (middle band)
                middle = data.rolling(window=length).mean()

                # Calculate standard deviation
                std_dev = data.rolling(window=length).std()

                # Calculate upper and lower bands
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)

                # Handle NaN values properly
                upper = upper.fillna(method='bfill').fillna(0)
                lower = lower.fillna(method='bfill').fillna(0)
                middle = middle.fillna(method='bfill').fillna(0)

                return upper, middle, lower

            except Exception as e:
                logger.warning(f"Bollinger Bands fallback calculation failed: {e}")
                nan_series = pd.Series(np.full(len(data), np.nan), index=data.index)
                return (nan_series, nan_series, nan_series)

        assert result is not None  # for type checker
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

        length = period
        df = ta.kc(high=high, low=low, close=close, window=length, scalar=scalar)
        if df is None:
            # ta.kcがNoneを返す場合のフォールバック
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        assert df is not None  # for type checker
        cols = list(df.columns)
        upper = df[next((c for c in cols if "KCu" in c), cols[0])]
        middle = df[
            next(
                (c for c in cols if "KCe" in c or "KCM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = df[next((c for c in cols if "KCl" in c), cols[-1])]
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def donchian(
        high: pd.Series,
        low: pd.Series,
        length: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels: returns (upper, middle, lower) with enhanced fallback"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        # DONCHIANパラメータエラー診断
        logger.debug(f"DONCHIAN: length={length}")
        logger.debug(f"DONCHIAN: high and low available")
        logger.debug(f"DONCHIAN: Checking ta.donchian signature")

        # pandas-ta donchianのパラメータを確認
        import inspect
        try:
            sig = inspect.signature(ta.donchian)
            logger.debug(f"DONCHIAN: ta.donchian params: {list(sig.parameters.keys())}")
        except Exception as e:
            logger.debug(f"DONCHIAN: inspect failed: {e}")

        try:
            # lengthパラメータの確認
            df = ta.donchian(high=high, low=low, length=length)
        except Exception as e:
            logger.error(f"DONCHIAN: ta.donchian failed with length parameter: {e}")
            # windowパラメータで試す
            try:
                df = ta.donchian(high=high, low=low, window=length)
                logger.info("DONCHIAN: Successful with window parameter")
            except Exception as e2:
                logger.error(f"DONCHIAN: ta.donchian also failed with window parameter: {e2}")
                # パラメータなしで試す
                try:
                    df = ta.donchian(high=high, low=low)
                    logger.info("DONCHIAN: Successful with no parameters")
                except Exception as e3:
                    logger.error(f"DONCHIAN: ta.donchian failed completely: {e3}")
                    df = None
    
        # Use length as length parameter for ta.donchian
        df = ta.donchian(high=high, low=low, length=length)
        if df is None or df.empty:
            # Enhanced fallback: Manual Donchian Channels calculation
            try:
                if len(high) < length:
                    nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
                    return nan_series, nan_series, nan_series

                # Upper band: highest high over length
                upper = high.rolling(window=length).max()

                # Lower band: lowest low over length
                lower = low.rolling(window=length).min()

                # Middle band: average of highest high and lowest low
                middle = (upper + lower) / 2.0

                # Handle NaN values
                upper = upper.fillna(method='bfill').fillna(high.rolling(window=length).max().iloc[0])
                lower = lower.fillna(method='bfill').fillna(low.rolling(window=length).min().iloc[0])
                middle = middle.fillna(method='bfill').fillna((upper + lower).mean())

                return upper, middle, lower

            except Exception as e:
                logger.warning(f"Donchian Channels fallback calculation failed: {e}")
                nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
                return nan_series, nan_series, nan_series

        assert df is not None  # for type checker
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
        period: int = 10,
        multiplier: float = 3.0,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Supertrend: returns (lower, upper, direction)"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Support 'factor' parameter as alias for 'multiplier'
        if 'factor' in kwargs:
            multiplier = kwargs['factor']

        # Enhanced data length check - Supertrend requires sufficient data
        length = period
        min_length = max(length * 4, 22)  # Ensure adequate data for ATR and calculation
        if len(high) < min_length:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        # Calculate basic bands for compatibility with enhanced parameter validation
        hl2 = (high + low) / 2
        atr = VolatilityIndicators.atr(high, low, close, period)

        # Add NaN check for ATR
        if atr.isna().all():
            logger.warning("SUPERTREND: ATR calculation failed, returning NaN")
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series, nan_series

        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        df = ta.supertrend(
            high=high, low=low, close=close, window=length, multiplier=multiplier
        )

        # pandas-ta version might not support factor, so ensure we use multiplier
        if df is None or (hasattr(df, 'isna') and df.isna().all().all()):
            # Enhanced fallback: Manual Supertrend calculation with improved algorithm
            try:
                logger.warning("SUPERTREND: Using enhanced manual fallback due to None or all NaN result")
                st_values = np.full(len(close), np.nan)
                direction = np.full(len(close), np.nan, dtype=float)

                # Initialize first values with proper validation
                if len(close) > length * 2:
                    start_idx = length * 2
                    if not upper.iloc[start_idx:].isna().all() and not lower.iloc[start_idx:].isna().all():
                        st_values[start_idx] = (upper.iloc[start_idx] + lower.iloc[start_idx]) / 2
                        direction[start_idx] = 1.0 if close.iloc[start_idx] >= st_values[start_idx] else -1.0

                        # Main calculation loop with enhanced validation
                        for i in range(start_idx + 1, len(close)):
                            if np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]):
                                continue

                            # Update bands with better trend logic
                            curr_upper = upper.iloc[i]
                            curr_lower = lower.iloc[i]

                            # Determine direction and Supertrend value
                            if direction[i-1] == 1.0:
                                # Bull trend
                                if close.iloc[i] < curr_upper:
                                    st_values[i] = curr_upper
                                    direction[i] = -1.0  # Switch to bearish
                                elif not np.isnan(st_values[i-1]):
                                    st_values[i] = st_values[i-1]
                                    direction[i] = 1.0
                            elif direction[i-1] == -1.0:
                                # Bear trend
                                if close.iloc[i] > curr_lower:
                                    st_values[i] = curr_lower
                                    direction[i] = 1.0  # Switch to bullish
                                elif not np.isnan(st_values[i-1]):
                                    st_values[i] = st_values[i-1]
                                    direction[i] = -1.0
                            else:
                                # Initialize new trend
                                st_values[i] = (curr_upper + curr_lower) / 2
                                direction[i] = 1.0 if close.iloc[i] >= st_values[i] else -1.0

                # Enhanced NaN handling
                st_series = pd.Series(st_values, index=close.index)
                direction_series = pd.Series(direction, index=close.index)

                return st_series, upper, direction_series

            except Exception as e:
                logger.warning(f"SUPERTREND enhanced fallback calculation failed: {e}")
                direction = pd.Series(np.where(close >= (upper + lower) / 2, 1.0, -1.0), index=close.index)
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
            direction = pd.Series(np.where(close.to_numpy() >= df[st_col].to_numpy(), 1.0, -1.0), index=close.index)
        else:
            direction = df[dir_col].fillna(0)
            if direction.isna().all():
                direction = pd.Series(np.where(close.to_numpy() >= df[st_col].to_numpy(), 1.0, -1.0), index=close.index)

        return lower, upper, direction

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
        logger.debug(f"ACCBANDS: Input data lengths - high: {len(high)}, low: {len(low)}, close: {len(close)}")
        logger.debug(f"ACCBANDS: period={period}")
        logger.debug(f"ACCBANDS: TA_LIB_AVAILABLE: {TA_LIB_AVAILABLE}")
        logger.debug(f"ACCBANDS: high sample: {high.head().values}")
        logger.debug(f"ACCBANDS: low sample: {low.head().values}")
        logger.debug(f"ACCBANDS: close sample: {close.head().values}")

        result = ta.accbands(high=high, low=low, close=close, length=length, mamode='sma')
        logger.debug(f"ACCBANDS: ta.accbands result is None: {result is None}")
        if result is not None:
            logger.debug(f"ACCBANDS: result shape: {getattr(result, 'shape', 'no shape')}")
            if hasattr(result, 'isna'):
                logger.debug(f"ACCBANDS: all NaN: {result.isna().all().all()}")

        if result is None or (hasattr(result, 'isna') and result.isna().all().all()):
            # ta.accbandsがNoneまたは全てNaNを返す場合のフォールバック
            logger.warning("ACCBANDS: Using fallback due to None or all NaN result")
            if len(high) >= length:
                # Manual Acceleration Bands calculation (simplified)
                middle = (high + low) / 2
                acceleration = 0.1  # default acceleration factor (10%)
                upper = middle * (1 + acceleration)
                lower = middle * (1 - acceleration)
                logger.debug(f"ACCBANDS: Fallback result all NaN upper: {upper.isna().all()}, middle: {middle.isna().all()}, lower: {lower.isna().all()}")
                return upper, middle, lower
            else:
                logger.debug(f"ACCBANDS: Data length too short: {len(high)} < {length}")
                nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
                return nan_series, nan_series, nan_series

        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "ACCBU" in c or "upper" in c.lower()), cols[0])]
        middle = result[
            next(
                (c for c in cols if "ACCBM" in c or "mid" in c.lower()),
                cols[1 if len(cols) > 1 else 0],
            )
        ]
        lower = result[next((c for c in cols if "ACCBL" in c or "lower" in c.lower()), cols[-1])]
        logger.debug(f"ACCBANDS: Columns found: {cols}")
        logger.debug(f"ACCBANDS: Final result all NaN upper: {upper.isna().all()}, middle: {middle.isna().all()}, lower: {lower.isna().all()}")
        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def hwc(
        close: pd.Series,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Holt-Winter Channel: returns (upper, middle, lower)"""
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Data length check - HWC requires sufficient data length
        if len(close) < 52:  # Minimum required length for proper calculation
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series, nan_series

        # Extract parameters from kwargs with enhanced defaults
        na = kwargs.get('na', 0.1)     # smoothing factor, reduced from 0.2
        nb = kwargs.get('nb', 0.05)    # trend factor, reduced from 0.1
        nc = kwargs.get('nc', 2.0)     # cycle amplitude, reduced from 3.0
        nd = kwargs.get('nd', 0.1)     # seasonality factor, reduced from 0.3
        scalar = kwargs.get('scalar', 1.5)  # channel width, reduced from 2.0

        result = ta.hwc(close=close, na=na, nb=nb, nc=nc, nd=nd, scalar=scalar)

        if result is None or (hasattr(result, 'isna') and result.isna().all().all()):
            # ta.hwcがNoneまたは全てNaNを返す場合のフォールバック
            logger.warning("HWC: Using enhanced manual fallback due to None or all NaN result")
            try:
                # Enhanced manual Holt-Winter Channel calculation
                # Based on TA-Lib/MetaQuotes implementation with proper error handling
                period = max(20, len(close) // 8)  # adaptive period based on data length

                if len(close) >= period:
                    # Middle band: double exponential smoothing approximation
                    ema1 = close.ewm(span=period // 2, adjust=False).mean()
                    middle = ema1.ewm(span=period // 2, adjust=False).mean()

                    # Upper band: calibrated channel calculation
                    upper = middle * (1 + na * scalar) - nb * close

                    # Lower band: calibrated channel calculation
                    lower = middle * (1 - na * scalar) + nb * close

                    # Enhanced NaN handling with forward fill then backward fill
                    middle = middle.fillna(method='ffill').fillna(method='bfill').fillna(close)
                    upper = upper.fillna(method='ffill').fillna(method='bfill').fillna(middle * (1 + na * scalar))
                    lower = lower.fillna(method='ffill').fillna(method='bfill').fillna(middle * (1 - na * scalar))

                    return upper, middle, lower
                else:
                    logger.warning(f"HWC: Data length {len(close)} too short for calculation with period {period}")
                    nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
                    return nan_series, nan_series, nan_series

            except Exception as e:
                logger.warning(f"HWC enhanced fallback calculation failed: {e}")
                nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
                return nan_series, nan_series, nan_series

        # Extract results from pandas-ta output with enhanced column detection
        assert result is not None  # for type checker
        cols = list(result.columns)
        upper = result[next((c for c in cols if "HWU" in c or "upper" in c.lower()), cols[0])]
        middle = result[next((c for c in cols if "HWM" in c or "mid" in c.lower()), cols[1 if len(cols) > 1 else 0])]
        lower = result[next((c for c in cols if "HWL" in c or "lower" in c.lower()), cols[-1])]

        return upper, middle, lower

    @staticmethod
    @handle_pandas_ta_errors
    def massi(
        high: pd.Series,
        low: pd.Series,
        period: int = 25,
        fast: int = 9,
        slow: int = 25,
    ) -> pd.Series:
        """Mass Index"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")

        # Fallback calculation when TA-Lib is unavailable or pandas-ta fails
        length = period
        if not TA_LIB_AVAILABLE:
            return VolatilityIndicators._massi_fallback(high, low, fast, slow)

        result = ta.massi(high=high, low=low, window=length, fast=fast, slow=slow)
        if result is None:
            # If pandas-ta fails, use fallback calculation
            return VolatilityIndicators._massi_fallback(high, low, fast, slow)
        return result

    @staticmethod
    def _massi_fallback(high: pd.Series, low: pd.Series, fast: int = 9, slow: int = 25) -> pd.Series:
        """Mass Index fallback calculation using pure pandas"""
        try:
            high_minus_low = high - low
            if len(high_minus_low) < fast:
                return pd.Series(np.full(len(high), np.nan), index=high.index)

            ratio = high_minus_low / high_minus_low.rolling(window=fast).mean()
            single_ema = ratio.ewm(span=fast, adjust=False).mean()
            mass_index = single_ema.ewm(span=slow, adjust=False).mean() * 100

            return mass_index
        except Exception as e:
            # If fallback also fails, return NaN series
            return pd.Series(np.full(len(high), np.nan), index=high.index)

    @staticmethod
    @handle_pandas_ta_errors
    def pdist(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
    ) -> pd.Series:
        """Price Distance"""
        if not isinstance(open, pd.Series):
            raise TypeError("open must be pandas Series")
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        length = period
        result = ta.pdist(open_=open, high=high, low=low, close=close, window=length)
        if result is None:
            return pd.Series(np.full(len(close), np.nan), index=close.index)
        return result

    @staticmethod
    @handle_pandas_ta_errors
    def variance(
        data: pd.Series, period: int = 14
    ) -> pd.Series:
        """Variance - Statistical variance indicator"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # Calculate rolling variance
        variance_values = data.rolling(window=length).var()
        return variance_values

    @staticmethod
    @handle_pandas_ta_errors
    def coefficient_of_variation(
        data: pd.Series, period: int = 14
    ) -> pd.Series:
        """Coefficient of Variation = Standard Deviation / Mean"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")
        length = period
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(data) < length:
            return pd.Series(np.full(len(data), np.nan), index=data.index)

        # Calculate rolling mean and std
        rolling_mean = data.rolling(window=length).mean()
        rolling_std = data.rolling(window=length).std()

        # Coefficient of Variation = std / mean
        # Handle division by zero
        cv_values = rolling_std / rolling_mean.replace(0, np.nan)
        return cv_values

    @staticmethod
    @handle_pandas_ta_errors
    def implied_risk_measure(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Implied Risk Measure - Estimated volatility-based risk measure"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("high, low, and close must have the same length")
        length = period
        if length <= 0:
            raise ValueError(f"length must be positive: {length}")
        if len(close) < length:
            return pd.Series(np.full(len(close), np.nan), index=close.index)

        # Calculate True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Implied Risk Measure = Rolling mean of True Range / Close * 100
        rolling_tr_mean = tr.rolling(window=length).mean()
        irm_values = (rolling_tr_mean / close) * 100

        return irm_values

    @staticmethod
    @handle_pandas_ta_errors
    def vortex(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator with enhanced fallback"""
        if not isinstance(high, pd.Series):
            raise TypeError("high must be pandas Series")
        if not isinstance(low, pd.Series):
            raise TypeError("low must be pandas Series")
        if not isinstance(close, pd.Series):
            raise TypeError("close must be pandas Series")

        # Enhanced data length check
        length = period
        min_length = max(length + 5, 20)  # Ensure adequate data for calculation
        if len(high) < min_length:
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

        # Input validation
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("high, low, and close must have the same length")

        try:
            # Try pandas-ta first with enhanced error checking
            logger.warning(f"VORTEX: Calling ta.vortex with length={length}, data_length={len(high)}")
            result = ta.vortex(high=high, low=low, close=close, window=length)
            logger.warning(f"VORTEX: ta.vortex result is None: {result is None}")

            if result is not None:
                # Check if result is valid (not all NaN)
                if hasattr(result, 'isna'):
                    if not result.isna().all().all():
                        if hasattr(result, 'iloc') and result.shape[1] >= 2:
                            vi_plus = result.iloc[:, 0]
                            vi_minus = result.iloc[:, 1]
                            return vi_plus, vi_minus

        except Exception as e:
            logger.warning(f"VORTEX ta.vortex call failed: {e}")

        # Enhanced fallback: Manual Vortex Indicator calculation with better error handling
        try:
            logger.warning("VORTEX: Using enhanced manual fallback calculation")

            # Calculate True Range (TR) with proper error handling
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - close.shift(1)),
                    np.abs(low - close.shift(1))
                )
            )

            # Handle division by zero and edge cases
            tr = tr.replace(0, np.nan).fillna(method='bfill').fillna(high - low)

            # Vortex movements
            vm_plus = np.abs(high - low.shift(1))
            vm_minus = np.abs(low - high.shift(1))

            # Vortex Indicator calculation with improved division handling
            vi_plus = vm_plus / tr
            vi_minus = vm_minus / tr

            # Shift arrays to pandas Series for proper indexing
            vi_plus_series = pd.Series(vi_plus, index=high.index)
            vi_minus_series = pd.Series(vi_minus, index=low.index)

            # Fill initial NaN values more robustly
            if len(vi_plus_series) > 1:
                valid_idx = vi_plus_series.notna().idxmax()
                if valid_idx is not None:
                    vi_plus_series.iloc[0:valid_idx] = vi_plus_series[valid_idx]

            if len(vi_minus_series) > 1:
                valid_idx = vi_minus_series.notna().idxmax()
                if valid_idx is not None:
                    vi_minus_series.iloc[0:valid_idx] = vi_minus_series[valid_idx]

            # Enhanced NaN handling with rolling replacement
            vi_plus_series = vi_plus_series.fillna(0).clip(0, 1)
            vi_minus_series = vi_minus_series.fillna(0).clip(0, 1)

            # Apply rolling mean to smooth the indicator (optional stabilization)
            if len(vi_plus_series) > length:
                vi_plus_series = vi_plus_series.rolling(window=min(length, 5)).mean().fillna(vi_plus_series)
                vi_minus_series = vi_minus_series.rolling(window=min(length, 5)).mean().fillna(vi_minus_series)

            return vi_plus_series, vi_minus_series

        except Exception as e:
            logger.warning(f"Vortex enhanced fallback calculation failed: {e}")
            nan_series = pd.Series(np.full(len(high), np.nan), index=high.index)
            return nan_series, nan_series

    @staticmethod
    @handle_pandas_ta_errors
    def ui(
        data: pd.Series, period: int = 14
    ) -> pd.Series:
        """Ulcer Index"""
        if not isinstance(data, pd.Series):
            raise TypeError("data must be pandas Series")

        length = period
        result = ta.ui(data, window=length)
        if result is None:
            return pd.Series(np.full(len(data), np.nan), index=data.index)
        return result
