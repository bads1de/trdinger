"""
モメンタム系テクニカル指標（オートストラテジー最適化版）

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import talib
import numpy as np
from typing import Tuple, cast
from ..utils import (
    validate_input,
    validate_multi_input,
    handle_talib_errors,
    format_indicator_result,
    ensure_numpy_array,
    TALibError,
)


class MomentumIndicators:
    """
    モメンタム系指標クラス（オートストラテジー最適化）

    全ての指標はnumpy配列を直接処理し、Ta-libの性能を最大限活用します。
    backtesting.pyでの使用に最適化されています。
    """

    @staticmethod
    @handle_talib_errors
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index (相対力指数)"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.RSI(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "RSI"))

    @staticmethod
    @handle_talib_errors
    def macd(
        data: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moving Average Convergence Divergence (MACD)"""
        data = ensure_numpy_array(data)
        fast_period = max(2, fast_period)
        slow_period = max(fast_period + 1, slow_period)
        signal_period = max(2, signal_period)
        validate_input(data, max(fast_period, slow_period, signal_period))
        macd, signal_line, histogram = talib.MACD(
            data,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal_line, histogram), "MACD"),
        )

    @staticmethod
    @handle_talib_errors
    def macdext(
        data: np.ndarray,
        fast_period: int = 12,
        fast_ma_type: int = 0,  # MA_Type.SMA
        slow_period: int = 26,
        slow_ma_type: int = 0,  # MA_Type.SMA
        signal_period: int = 9,
        signal_ma_type: int = 0,  # MA_Type.SMA
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD with controllable MA type"""
        data = ensure_numpy_array(data)

        # matypeは整数値として直接使用

        validate_input(data, max(fast_period, slow_period, signal_period))
        macd, signal, histogram = talib.MACDEXT(
            data,
            fastperiod=fast_period,
            fastmatype=fast_ma_type,  # type: ignore
            slowperiod=slow_period,
            slowmatype=slow_ma_type,  # type: ignore
            signalperiod=signal_period,
            signalmatype=signal_ma_type,  # type: ignore
        )
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal, histogram), "MACDEXT"),
        )

    @staticmethod
    @handle_talib_errors
    def macdfix(
        data: np.ndarray, signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moving Average Convergence/Divergence Fix 12/26"""
        data = ensure_numpy_array(data)
        validate_input(data, max(26, signal_period))
        macd, signal, histogram = talib.MACDFIX(data, signalperiod=signal_period)
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal, histogram), "MACDFIX"),
        )

    @staticmethod
    @handle_talib_errors
    def stoch(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowk_matype: int = 0,  # MA_Type.SMA
        slowd_period: int = 3,
        slowd_matype: int = 0,  # MA_Type.SMA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic (ストキャスティクス)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        fastk_period = max(2, fastk_period)
        slowk_period = max(2, slowk_period)
        slowd_period = max(2, slowd_period)

        # matypeは整数値として直接使用

        validate_multi_input(
            high, low, close, max(fastk_period, slowk_period, slowd_period)
        )
        slowk, slowd = talib.STOCH(
            high,
            low,
            close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=slowk_matype,  # type: ignore
            slowd_period=slowd_period,
            slowd_matype=slowd_matype,  # type: ignore
        )
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((slowk, slowd), "STOCH"),
        )

    @staticmethod
    @handle_talib_errors
    def stochf(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        fastk_period: int = 5,
        fastd_period: int = 3,
        fastd_matype: int = 0,  # MA_Type.SMA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Fast (高速ストキャスティクス)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        # matypeは整数値として直接使用

        validate_multi_input(high, low, close, max(fastk_period, fastd_period))
        fastk, fastd = talib.STOCHF(
            high,
            low,
            close,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype,  # type: ignore
        )
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((fastk, fastd), "STOCHF"),
        )

    @staticmethod
    @handle_talib_errors
    def stochrsi(
        data: np.ndarray,
        period: int = 14,
        fastk_period: int = 5,
        fastd_period: int = 3,
        fastd_matype: int = 0,  # MA_Type.SMA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Relative Strength Index"""
        data = ensure_numpy_array(data)

        # matypeは整数値として直接使用

        validate_input(data, max(period, fastk_period, fastd_period))
        fastk, fastd = talib.STOCHRSI(
            data,
            timeperiod=period,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype,  # type: ignore
        )
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((fastk, fastd), "STOCHRSI"),
        )

    @staticmethod
    @handle_talib_errors
    def williams_r(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Williams' %R (ウィリアムズ%R)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.WILLR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "WILLR"))

    @staticmethod
    @handle_talib_errors
    def cci(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Commodity Channel Index (商品チャネル指数)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.CCI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CCI"))

    @staticmethod
    @handle_talib_errors
    def cmo(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Chande Momentum Oscillator"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.CMO(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CMO"))

    @staticmethod
    @handle_talib_errors
    def roc(data: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of change"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.ROC(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROC"))

    @staticmethod
    @handle_talib_errors
    def rocp(data: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of change Percentage"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.ROCP(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCP"))

    @staticmethod
    @handle_talib_errors
    def rocr(data: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of change ratio"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.ROCR(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR"))

    @staticmethod
    @handle_talib_errors
    def rocr100(data: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of change ratio 100 scale"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.ROCR100(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR100"))

    @staticmethod
    @handle_talib_errors
    def mom(data: np.ndarray, period: int = 10) -> np.ndarray:
        """Momentum (モメンタム)"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.MOM(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MOM"))

    @staticmethod
    @handle_talib_errors
    def adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average Directional Movement Index"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.ADX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADX"))

    @staticmethod
    @handle_talib_errors
    def adxr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average Directional Movement Index Rating"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.ADXR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADXR"))

    @staticmethod
    @handle_talib_errors
    def aroon(
        high: np.ndarray, low: np.ndarray, period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aroon (アルーン)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        aroondown, aroonup = talib.AROON(high, low, timeperiod=period)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((aroondown, aroonup), "AROON"),
        )

    @staticmethod
    @handle_talib_errors
    def aroonosc(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """Aroon Oscillator (アルーンオシレーター)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        result = talib.AROONOSC(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "AROONOSC"))

    @staticmethod
    @handle_talib_errors
    def dx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Directional Movement Index"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.DX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "DX"))

    @staticmethod
    @handle_talib_errors
    def mfi(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Money Flow Index (マネーフローインデックス)"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        volume = ensure_numpy_array(volume)
        validate_multi_input(high, low, close, period)
        if len(volume) != len(close):
            raise TALibError(
                f"出来高データの長さが一致しません。Volume: {len(volume)}, Close: {len(close)}"
            )
        result = talib.MFI(high, low, close, volume, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MFI"))

    @staticmethod
    @handle_talib_errors
    def plus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Plus Directional Indicator"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.PLUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DI"))

    @staticmethod
    @handle_talib_errors
    def minus_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Minus Directional Indicator"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        result = talib.MINUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DI"))

    @staticmethod
    @handle_talib_errors
    def plus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """Plus Directional Movement"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        result = talib.PLUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DM"))

    @staticmethod
    @handle_talib_errors
    def minus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """Minus Directional Movement"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        result = talib.MINUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DM"))

    @staticmethod
    @handle_talib_errors
    def ppo(
        data: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        matype: int = 0,  # MA_Type.SMA
    ) -> np.ndarray:
        """Percentage Price Oscillator"""
        data = ensure_numpy_array(data)
        validate_input(data, max(fastperiod, slowperiod))
        result = talib.PPO(
            data,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            matype=matype,  # type: ignore
        )
        return cast(np.ndarray, format_indicator_result(result, "PPO"))

    @staticmethod
    @handle_talib_errors
    def trix(data: np.ndarray, period: int = 30) -> np.ndarray:
        """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
        data = ensure_numpy_array(data)
        validate_input(data, period)
        result = talib.TRIX(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "TRIX"))

    @staticmethod
    @handle_talib_errors
    def ultosc(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeperiod1: int = 7,
        timeperiod2: int = 14,
        timeperiod3: int = 28,
    ) -> np.ndarray:
        """Ultimate Oscillator"""
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(
            high, low, close, max(timeperiod1, timeperiod2, timeperiod3)
        )
        result = talib.ULTOSC(
            high,
            low,
            close,
            timeperiod1=timeperiod1,
            timeperiod2=timeperiod2,
            timeperiod3=timeperiod3,
        )
        return cast(np.ndarray, format_indicator_result(result, "ULTOSC"))

    @staticmethod
    @handle_talib_errors
    def bop(
        open_data: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Balance Of Power"""
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )
        validate_input(close, 1)
        result = talib.BOP(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "BOP"))

    @staticmethod
    @handle_talib_errors
    def apo(
        data: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        matype: int = 0,  # MA_Type.SMA
    ) -> np.ndarray:
        """Absolute Price Oscillator"""
        data = ensure_numpy_array(data)
        validate_input(data, max(fastperiod, slowperiod))
        result = talib.APO(
            data,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            matype=matype,  # type: ignore
        )
        return cast(np.ndarray, format_indicator_result(result, "APO"))
