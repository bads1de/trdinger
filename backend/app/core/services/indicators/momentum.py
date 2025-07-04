"""
モメンタム系テクニカル指標（オートストラテジー最適化版）

このモジュールはnumpy配列ベースでTa-libを直接使用し、
backtesting.pyとの完全な互換性を提供します。
pandas Seriesの変換は一切行いません。
"""

import talib
import numpy as np
from typing import Tuple, cast
from .utils import (
    validate_input,
    validate_multi_input,
    handle_talib_errors,
    log_indicator_calculation,
    format_indicator_result,
    ensure_numpy_array,
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
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence Divergence (MACD)

        Args:
            data: 価格データ（numpy配列）
            fastperiod: 高速期間（デフォルト: 12）
            slowperiod: 低速期間（デフォルト: 26）
            signalperiod: シグナル期間（デフォルト: 9）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        # パラメータのバリデーションと調整
        fastperiod = max(2, fastperiod)
        slowperiod = max(fastperiod + 1, slowperiod)
        signalperiod = max(2, signalperiod)

        validate_input(data, max(fastperiod, slowperiod, signalperiod))
        log_indicator_calculation(
            "MACD",
            {
                "fastperiod": fastperiod,
                "slowperiod": slowperiod,
                "signalperiod": signalperiod,
            },
            len(data),
        )

        macd, signal_line, histogram = talib.MACD(
            data,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal_line, histogram), "MACD"),
        )

    @staticmethod
    @handle_talib_errors
    def macdext(
        data: np.ndarray,
        fastperiod: int = 12,
        fastmatype: int = 0,
        slowperiod: int = 26,
        slowmatype: int = 0,
        signalperiod: int = 9,
        signalmatype: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD with controllable MA type (MA種別制御可能MACD)

        Args:
            data: 価格データ（numpy配列）
            fastperiod: 高速期間
            fastmatype: 高速MA種別
            slowperiod: 低速期間
            slowmatype: 低速MA種別
            signalperiod: シグナル期間
            signalmatype: シグナルMA種別

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
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

        macd, signal_line, histogram = talib.MACDEXT(data, **params)  # type: ignore
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            format_indicator_result((macd, signal_line, histogram), "MACDEXT"),
        )

    @staticmethod
    @handle_talib_errors
    def macdfix(
        data: np.ndarray, signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence/Divergence Fix 12/26 (固定12/26 MACD)

        Args:
            data: 価格データ（numpy配列）
            signalperiod: シグナル期間（デフォルト: 9）

        Returns:
            (MACD, Signal, Histogram)のtuple
        """
        validate_input(data, max(26, signalperiod))
        log_indicator_calculation("MACDFIX", {"signalperiod": signalperiod}, len(data))

        macd, signal_line, histogram = talib.MACDFIX(data, signalperiod=signalperiod)
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
        validate_input(data, period)
        log_indicator_calculation("MOM", {"period": period}, len(data))

        result = talib.MOM(data, timeperiod=period)
        return cast(np.ndarray, format_indicator_result(result, "MOM"))
