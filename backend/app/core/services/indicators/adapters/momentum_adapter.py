"""
モメンタム系テクニカル指標アダプター

オシレーター系やモメンタム系の指標計算を担当します。
RSI, MACD, Stochastic, CCI, Williams %R, ADX, Aroon, MOM, ROC, MFI などを提供します。
"""

import talib
import pandas as pd
from typing import Dict
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class MomentumAdapter(BaseAdapter):
    """モメンタム系指標のTA-Libアダプター"""

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (相対力指数) を計算

        Args:
            data: 価格データ（pandas Series）
            period: RSIの期間（デフォルト: 14）

        Returns:
            RSI値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("RSI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.RSI, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"RSI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("RSI", e)
            raise TALibCalculationError(f"RSI計算失敗: {e}")

    @staticmethod
    def macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) を計算

        Args:
            data: 価格データ（pandas Series）
            fast: 短期EMAの期間（デフォルト: 12）
            slow: 長期EMAの期間（デフォルト: 26）
            signal: シグナル線の期間（デフォルト: 9）

        Returns:
            MACD値を含む辞書（macd_line, signal_line, histogram）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, slow)  # 最も長い期間で検証
        MomentumAdapter._log_calculation_start(
            "MACD", fast=fast, slow=slow, signal=signal
        )

        try:
            macd_line, signal_line, histogram = MomentumAdapter._safe_talib_calculation(
                talib.MACD,
                data.values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )

            return {
                "macd_line": pd.Series(macd_line, index=data.index, name="MACD"),
                "signal_line": pd.Series(
                    signal_line, index=data.index, name="MACD_Signal"
                ),
                "histogram": pd.Series(
                    histogram, index=data.index, name="MACD_Histogram"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MACD", e)
            raise TALibCalculationError(f"MACD計算失敗: {e}")

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator (ストキャスティクス) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            k_period: %Kの期間（デフォルト: 14）
            d_period: %Dの期間（デフォルト: 3）

        Returns:
            ストキャスティクス値を含む辞書（k_percent, d_percent）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, k_period)
        MomentumAdapter._log_calculation_start(
            "STOCH", k_period=k_period, d_period=d_period
        )

        try:
            k_percent, d_percent = MomentumAdapter._safe_talib_calculation(
                talib.STOCH,
                high.values,
                low.values,
                close.values,
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period,
            )

            return {
                "k_percent": pd.Series(
                    k_percent, index=close.index, name=f"STOCH_K_{k_period}"
                ),
                "d_percent": pd.Series(
                    d_percent, index=close.index, name=f"STOCH_D_{d_period}"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("STOCH", e)
            raise TALibCalculationError(f"ストキャスティクス計算失敗: {e}")

    @staticmethod
    def cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index (CCI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: CCIの期間（デフォルト: 20）

        Returns:
            CCI値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("CCI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.CCI, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"CCI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("CCI", e)
            raise TALibCalculationError(f"CCI計算失敗: {e}")

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Williams %R を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: Williams %Rの期間（デフォルト: 14）

        Returns:
            Williams %R値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("WILLR", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.WILLR, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"WILLR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("WILLR", e)
            raise TALibCalculationError(f"Williams %R計算失敗: {e}")

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average Directional Movement Index (ADX) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: ADXの期間（デフォルト: 14）

        Returns:
            ADX値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("ADX", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ADX, high.values, low.values, close.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"ADX_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ADX", e)
            raise TALibCalculationError(f"ADX計算失敗: {e}")

    @staticmethod
    def aroon(
        high: pd.Series, low: pd.Series, period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Aroon (アルーン) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: アルーンの期間（デフォルト: 14）

        Returns:
            アルーン値を含む辞書（aroon_down, aroon_up）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low)
        MomentumAdapter._validate_input(high, period)
        MomentumAdapter._log_calculation_start("AROON", period=period)

        try:
            aroon_down, aroon_up = MomentumAdapter._safe_talib_calculation(
                talib.AROON, high.values, low.values, timeperiod=period
            )

            return {
                "aroon_down": pd.Series(
                    aroon_down, index=high.index, name=f"AROON_DOWN_{period}"
                ),
                "aroon_up": pd.Series(
                    aroon_up, index=high.index, name=f"AROON_UP_{period}"
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("AROON", e)
            raise TALibCalculationError(f"アルーン計算失敗: {e}")

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum (モメンタム) を計算

        Args:
            data: 価格データ（pandas Series）
            period: モメンタムの期間（デフォルト: 10）

        Returns:
            モメンタム値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("MOM", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.MOM, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"MOM_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MOM", e)
            raise TALibCalculationError(f"モメンタム計算失敗: {e}")

    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC) を計算

        Args:
            data: 価格データ（pandas Series）
            period: ROCの期間（デフォルト: 10）

        Returns:
            ROC値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_input(data, period)
        MomentumAdapter._log_calculation_start("ROC", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.ROC, data.values, timeperiod=period
            )
            return MomentumAdapter._create_series_result(
                result, data.index, f"ROC_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("ROC", e)
            raise TALibCalculationError(f"ROC計算失敗: {e}")

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index (MFI) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: MFIの期間（デフォルト: 14）

        Returns:
            MFI値のpandas Series（0-100の範囲）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        MomentumAdapter._validate_multi_input(high, low, close, volume)
        MomentumAdapter._validate_input(close, period)
        MomentumAdapter._log_calculation_start("MFI", period=period)

        try:
            result = MomentumAdapter._safe_talib_calculation(
                talib.MFI,
                high.values,
                low.values,
                close.values,
                volume.values,
                timeperiod=period,
            )
            return MomentumAdapter._create_series_result(
                result, close.index, f"MFI_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            MomentumAdapter._log_calculation_error("MFI", e)
            raise TALibCalculationError(f"MFI計算失敗: {e}")
