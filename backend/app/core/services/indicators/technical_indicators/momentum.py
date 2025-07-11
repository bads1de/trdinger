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
    log_indicator_calculation,
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
        """
        Relative Strength Index (相対力指数)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            RSI値のnumpy配列（0-100の範囲）
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("RSI", {"period": period}, len(data))

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
        """
        Moving Average Convergence Divergence (MACD)

        Args:
            data: 価格データ（numpy配列）
            fast_period: 高速期間（デフォルト: 12）
            slow_period: 低速期間（デフォルト: 26）
            signal_period: シグナル期間（デフォルト: 9）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        # パラメータのバリデーションと調整
        fast_period = max(2, fast_period)
        slow_period = max(fast_period + 1, slow_period)
        signal_period = max(2, signal_period)

        validate_input(data, max(fast_period, slow_period, signal_period))
        log_indicator_calculation(
            "MACD",
            {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
            len(data),
        )

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
        fast_ma_type: int = 0,
        slow_period: int = 26,
        slow_ma_type: int = 0,
        signal_period: int = 9,
        signal_ma_type: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD with controllable MA type (MA種別制御可能MACD)

        Args:
            data: 価格データ（numpy配列）
            fast_period: 高速期間
            fast_ma_type: 高速MA種別
            slow_period: 低速期間
            slow_ma_type: 低速MA種別
            signal_period: シグナル期間
            signal_ma_type: シグナルMA種別

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        validate_input(data, max(fast_period, slow_period, signal_period))

        params = {
            "fastperiod": fast_period,
            "fastmatype": fast_ma_type,
            "slowperiod": slow_period,
            "slowmatype": slow_ma_type,
            "signalperiod": signal_period,
            "signalmatype": signal_ma_type,
        }
        log_indicator_calculation("MACDEXT", params, len(data))

        macd, signal_line, histogram = talib.MACDEXT(data, **params)  # type: ignore
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal_line, histogram), "MACDEXT"),
        )

    @staticmethod
    @handle_talib_errors
    def macdfix(
        data: np.ndarray, signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence/Divergence Fix 12/26 (固定12/26 MACD)

        Args:
            data: 価格データ（numpy配列）
            signal_period: シグナル期間（デフォルト: 9）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        validate_input(data, max(26, signal_period))
        log_indicator_calculation(
            "MACDFIX", {"signal_period": signal_period}, len(data)
        )

        macd, signal_line, histogram = talib.MACDFIX(data, signalperiod=signal_period)
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal_line, histogram), "MACDFIX"),
        )

    @staticmethod
    @handle_talib_errors
    def stoch(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowk_matype: int = 0,
        slowd_period: int = 3,
        slowd_matype: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic (ストキャスティクス)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            fastk_period: Fast %K期間
            slowk_period: Slow %K期間
            slowk_matype: Slow %K MA種別
            slowd_period: Slow %D期間
            slowd_matype: Slow %D MA種別

        Returns:
            (%K, %D)のtuple
        """
        # パラメータのバリデーションと調整
        fastk_period = max(2, fastk_period)
        slowk_period = max(2, slowk_period)
        slowd_period = max(2, slowd_period)

        validate_multi_input(
            high, low, close, max(fastk_period, slowk_period, slowd_period)
        )

        params = {
            "fastk_period": fastk_period,
            "slowk_period": slowk_period,
            "slowk_matype": slowk_matype,
            "slowd_period": slowd_period,
            "slowd_matype": slowd_matype,
        }
        log_indicator_calculation("STOCH", params, len(close))

        slowk, slowd = talib.STOCH(high, low, close, **params)  # type: ignore
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
        fastd_matype: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Fast (高速ストキャスティクス)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            fastk_period: Fast %K期間
            fastd_period: Fast %D期間
            fastd_matype: Fast %D MA種別

        Returns:
            (Fast %K, Fast %D)のtuple
        """
        validate_multi_input(high, low, close, max(fastk_period, fastd_period))

        params = {
            "fastk_period": fastk_period,
            "fastd_period": fastd_period,
            "fastd_matype": fastd_matype,
        }
        log_indicator_calculation("STOCHF", params, len(close))

        fastk, fastd = talib.STOCHF(high, low, close, **params)  # type: ignore
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
        fastd_matype: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Relative Strength Index (ストキャスティクスRSI)

        Args:
            data: 価格データ（numpy配列）
            period: RSI期間
            fastk_period: Fast %K期間
            fastd_period: Fast %D期間
            fastd_matype: Fast %D MA種別

        Returns:
            (Fast %K, Fast %D)のtuple
        """
        validate_input(data, max(period, fastk_period, fastd_period))

        params = {
            "timeperiod": period,
            "fastk_period": fastk_period,
            "fastd_period": fastd_period,
            "fastd_matype": fastd_matype,
        }
        log_indicator_calculation("STOCHRSI", params, len(data))

        fastk, fastd = talib.STOCHRSI(data, **params)  # type: ignore
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((fastk, fastd), "STOCHRSI"),
        )

    @staticmethod
    @handle_talib_errors
    def williams_r(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Williams' %R (ウィリアムズ%R)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            Williams %R値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("WILLR", {"period": period}, len(close))

        result = talib.WILLR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "WILLR"))

    @staticmethod
    @handle_talib_errors
    def cci(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """
        Commodity Channel Index (商品チャネル指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            CCI値のnumpy配列
        """
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("CCI", {"period": period}, len(close))

        result = talib.CCI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CCI"))

    @staticmethod
    @handle_talib_errors
    def cmo(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Chande Momentum Oscillator (チャンデモメンタムオシレーター)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            CMO値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("CMO", {"period": period}, len(data))

        result = talib.CMO(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CMO"))

    @staticmethod
    @handle_talib_errors
    def roc(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change (変化率)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROC値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("ROC", {"period": period}, len(data))

        result = talib.ROC(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROC"))

    @staticmethod
    @handle_talib_errors
    def rocp(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change Percentage (変化率パーセンテージ)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCP値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("ROCP", {"period": period}, len(data))

        result = talib.ROCP(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCP"))

    @staticmethod
    @handle_talib_errors
    def rocr(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change ratio (変化率比)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCR値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("ROCR", {"period": period}, len(data))

        result = talib.ROCR(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR"))

    @staticmethod
    @handle_talib_errors
    def rocr100(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change ratio 100 scale (変化率比100スケール)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCR100値のnumpy配列
        """
        validate_input(data, period)
        log_indicator_calculation("ROCR100", {"period": period}, len(data))

        result = talib.ROCR100(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR100"))

    @staticmethod
    @handle_talib_errors
    def mom(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Momentum (モメンタム)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            MOM値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("MOM", {"period": period}, len(data))

        result = talib.MOM(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MOM"))

    @staticmethod
    @handle_talib_errors
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average Directional Movement Index (平均方向性指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADX値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("ADX", {"period": period}, len(close))

        result = talib.ADX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADX"))

    @staticmethod
    @handle_talib_errors
    def adxr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average Directional Movement Index Rating (平均方向性指数レーティング)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            ADXR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("ADXR", {"period": period}, len(close))

        result = talib.ADXR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ADXR"))

    @staticmethod
    @handle_talib_errors
    def aroon(high: np.ndarray, low: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aroon (アルーン)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            (AroonDown, AroonUp)のtuple
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("AROON", {"period": period}, len(high))

        aroondown, aroonup = talib.AROON(high, low, timeperiod=period)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((aroondown, aroonup), "AROON"),
        )

    @staticmethod
    @handle_talib_errors
    def aroonosc(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Aroon Oscillator (アルーンオシレーター)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            AROONOSC値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("AROONOSC", {"period": period}, len(high))

        result = talib.AROONOSC(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "AROONOSC"))

    @staticmethod
    @handle_talib_errors
    def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Commodity Channel Index (商品チャネル指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            CCI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("CCI", {"period": period}, len(close))

        result = talib.CCI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "CCI"))

    @staticmethod
    @handle_talib_errors
    def dx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Directional Movement Index (方向性指数)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            DX値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("DX", {"period": period}, len(close))

        result = talib.DX(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "DX"))

    @staticmethod
    @handle_talib_errors
    def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Money Flow Index (マネーフローインデックス)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            volume: 出来高データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MFI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        volume = ensure_numpy_array(volume)

        validate_multi_input(high, low, close, period)
        if len(volume) != len(close):
            raise TALibError(f"出来高データの長さが一致しません。Volume: {len(volume)}, Close: {len(close)}")

        log_indicator_calculation("MFI", {"period": period}, len(close))

        result = talib.MFI(high, low, close, volume, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MFI"))

    @staticmethod
    @handle_talib_errors
    def plus_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Plus Directional Indicator (プラス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("PLUS_DI", {"period": period}, len(close))

        result = talib.PLUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DI"))

    @staticmethod
    @handle_talib_errors
    def minus_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Minus Directional Indicator (マイナス方向性指標)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DI値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("MINUS_DI", {"period": period}, len(close))

        result = talib.MINUS_DI(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DI"))

    @staticmethod
    @handle_talib_errors
    def plus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Plus Directional Movement (プラス方向性移動)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            PLUS_DM値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("PLUS_DM", {"period": period}, len(high))

        result = talib.PLUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "PLUS_DM"))

    @staticmethod
    @handle_talib_errors
    def minus_dm(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Minus Directional Movement (マイナス方向性移動)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            MINUS_DM値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        validate_multi_input(high, low, high, period)
        log_indicator_calculation("MINUS_DM", {"period": period}, len(high))

        result = talib.MINUS_DM(high, low, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MINUS_DM"))

    @staticmethod
    @handle_talib_errors
    def ppo(data: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> np.ndarray:
        """
        Percentage Price Oscillator (パーセンテージ価格オシレーター)

        Args:
            data: 価格データ（numpy配列）
            fastperiod: 高速期間（デフォルト: 12）
            slowperiod: 低速期間（デフォルト: 26）
            matype: 移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            PPO値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, max(fastperiod, slowperiod))
        log_indicator_calculation(
            "PPO", {"fastperiod": fastperiod, "slowperiod": slowperiod, "matype": matype}, len(data)
        )

        result = talib.PPO(data, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        return cast(np.ndarray, format_indicator_result(result, "PPO"))

    @staticmethod
    @handle_talib_errors
    def roc(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change : ((price/prevPrice)-1)*100 (変化率)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROC値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("ROC", {"period": period}, len(data))

        result = talib.ROC(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROC"))

    @staticmethod
    @handle_talib_errors
    def rocp(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change Percentage: (price-prevPrice)/prevPrice (変化率パーセンテージ)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCP値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("ROCP", {"period": period}, len(data))

        result = talib.ROCP(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCP"))

    @staticmethod
    @handle_talib_errors
    def rocr(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change ratio: (price/prevPrice) (変化率比)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCR値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("ROCR", {"period": period}, len(data))

        result = talib.ROCR(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR"))

    @staticmethod
    @handle_talib_errors
    def rocr100(data: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Rate of change ratio 100 scale: (price/prevPrice)*100 (変化率比100スケール)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 10）

        Returns:
            ROCR100値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("ROCR100", {"period": period}, len(data))

        result = talib.ROCR100(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "ROCR100"))

    @staticmethod
    @handle_talib_errors
    def stochf(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        fastk_period: int = 5,
        fastd_period: int = 3,
        fastd_matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Fast (高速ストキャスティクス)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            fastk_period: FastK期間（デフォルト: 5）
            fastd_period: FastD期間（デフォルト: 3）
            fastd_matype: FastD移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            (FastK, FastD)のtuple
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, max(fastk_period, fastd_period))

        params = {
            "fastk_period": fastk_period,
            "fastd_period": fastd_period,
            "fastd_matype": fastd_matype,
        }
        log_indicator_calculation("STOCHF", params, len(close))

        fastk, fastd = talib.STOCHF(high, low, close, **params)
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
        fastd_matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Relative Strength Index (ストキャスティクスRSI)

        Args:
            data: 価格データ（numpy配列）
            period: RSI期間（デフォルト: 14）
            fastk_period: FastK期間（デフォルト: 5）
            fastd_period: FastD期間（デフォルト: 3）
            fastd_matype: FastD移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            (FastK, FastD)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, max(period, fastk_period, fastd_period))

        params = {
            "timeperiod": period,
            "fastk_period": fastk_period,
            "fastd_period": fastd_period,
            "fastd_matype": fastd_matype,
        }
        log_indicator_calculation("STOCHRSI", params, len(data))

        fastk, fastd = talib.STOCHRSI(data, **params)
        return cast(
            Tuple[np.ndarray, np.ndarray],
            format_indicator_result((fastk, fastd), "STOCHRSI"),
        )

    @staticmethod
    @handle_talib_errors
    def trix(data: np.ndarray, period: int = 30) -> np.ndarray:
        """
        1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (トリプルスムーズEMAの1日変化率)

        Args:
            data: 価格データ（numpy配列）
            period: 期間（デフォルト: 30）

        Returns:
            TRIX値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, period)
        log_indicator_calculation("TRIX", {"period": period}, len(data))

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
        timeperiod3: int = 28
    ) -> np.ndarray:
        """
        Ultimate Oscillator (アルティメットオシレーター)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            timeperiod1: 期間1（デフォルト: 7）
            timeperiod2: 期間2（デフォルト: 14）
            timeperiod3: 期間3（デフォルト: 28）

        Returns:
            ULTOSC値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, max(timeperiod1, timeperiod2, timeperiod3))

        params = {
            "timeperiod1": timeperiod1,
            "timeperiod2": timeperiod2,
            "timeperiod3": timeperiod3,
        }
        log_indicator_calculation("ULTOSC", params, len(close))

        result = talib.ULTOSC(high, low, close, **params)
        return cast(np.ndarray, format_indicator_result(result, "ULTOSC"))

    @staticmethod
    @handle_talib_errors
    def willr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Williams' %R (ウィリアムズ%R)

        Args:
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）
            period: 期間（デフォルト: 14）

        Returns:
            WILLR値のnumpy配列
        """
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)
        validate_multi_input(high, low, close, period)
        log_indicator_calculation("WILLR", {"period": period}, len(close))

        result = talib.WILLR(high, low, close, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "WILLR"))

    @staticmethod
    @handle_talib_errors
    def bop(open_data: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Balance Of Power (バランスオブパワー)

        Args:
            open_data: 始値データ（numpy配列）
            high: 高値データ（numpy配列）
            low: 安値データ（numpy配列）
            close: 終値データ（numpy配列）

        Returns:
            BOP値のnumpy配列
        """
        open_data = ensure_numpy_array(open_data)
        high = ensure_numpy_array(high)
        low = ensure_numpy_array(low)
        close = ensure_numpy_array(close)

        # 全データの長さが一致することを確認
        if not (len(open_data) == len(high) == len(low) == len(close)):
            raise TALibError(
                f"OHLCデータの長さが一致しません。Open: {len(open_data)}, High: {len(high)}, Low: {len(low)}, Close: {len(close)}"
            )

        validate_input(close, 1)
        log_indicator_calculation("BOP", {}, len(close))

        result = talib.BOP(open_data, high, low, close)
        return cast(np.ndarray, format_indicator_result(result, "BOP"))

    @staticmethod
    @handle_talib_errors
    def apo(data: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> np.ndarray:
        """
        Absolute Price Oscillator (絶対価格オシレーター)

        Args:
            data: 価格データ（numpy配列）
            fastperiod: 高速期間（デフォルト: 12）
            slowperiod: 低速期間（デフォルト: 26）
            matype: 移動平均タイプ（デフォルト: 0=SMA）

        Returns:
            APO値のnumpy配列
        """
        data = ensure_numpy_array(data)
        validate_input(data, max(fastperiod, slowperiod))
        log_indicator_calculation(
            "APO", {"fastperiod": fastperiod, "slowperiod": slowperiod, "matype": matype}, len(data)
        )

        result = talib.APO(data, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        return cast(np.ndarray, format_indicator_result(result, "APO"))

    @staticmethod
    @handle_talib_errors
    def macdext(
        data: np.ndarray,
        fastperiod: int = 12,
        fastmatype: int = 0,
        slowperiod: int = 26,
        slowmatype: int = 0,
        signalperiod: int = 9,
        signalmatype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD with controllable MA type (制御可能なMA型MACD)

        Args:
            data: 価格データ（numpy配列）
            fastperiod: 高速期間（デフォルト: 12）
            fastmatype: 高速MA型（デフォルト: 0=SMA）
            slowperiod: 低速期間（デフォルト: 26）
            slowmatype: 低速MA型（デフォルト: 0=SMA）
            signalperiod: シグナル期間（デフォルト: 9）
            signalmatype: シグナルMA型（デフォルト: 0=SMA）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, max(fastperiod, slowperiod, signalperiod))

        params = {
            "fastperiod": fastperiod,
            "fastmatype": fastmatype,
            "slowperiod": slowperiod,
            "slowmatype": slowmatype,
            "signalperiod": signalperiod,
            "signalmatype": signalmatype,
        }
        log_indicator_calculation("MACDEXT", params, len(data))

        macd, signal, histogram = talib.MACDEXT(data, **params)
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal, histogram), "MACDEXT"),
        )

    @staticmethod
    @handle_talib_errors
    def macdfix(data: np.ndarray, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence/Divergence Fix 12/26 (固定12/26 MACD)

        Args:
            data: 価格データ（numpy配列）
            signalperiod: シグナル期間（デフォルト: 9）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        data = ensure_numpy_array(data)
        validate_input(data, max(26, signalperiod))
        log_indicator_calculation("MACDFIX", {"signalperiod": signalperiod}, len(data))

        macd, signal, histogram = talib.MACDFIX(data, signalperiod=signalperiod)
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal, histogram), "MACDFIX"),
        )
