"""
トレンド系テクニカル指標アダプター

移動平均系やトレンドフォロー系の指標計算を担当します。
SMA, EMA, TEMA, DEMA, KAMA, T3 などのトレンド系指標を提供します。
"""

import talib
import pandas as pd
from typing import Dict
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class TrendAdapter(BaseAdapter):
    """トレンド系指標のTA-Libアダプター"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (単純移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            SMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("SMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.SMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"SMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("SMA", e)
            raise TALibCalculationError(f"SMA計算失敗: {e}")

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (指数移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            EMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("EMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.EMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"EMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("EMA", e)
            raise TALibCalculationError(f"EMA計算失敗: {e}")

    @staticmethod
    def tema(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Triple Exponential Moving Average (TEMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: TEMAの期間（デフォルト: 30）

        Returns:
            TEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("TEMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.TEMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"TEMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("TEMA", e)
            raise TALibCalculationError(f"TEMA計算失敗: {e}")

    @staticmethod
    def dema(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Double Exponential Moving Average (DEMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: DEMAの期間（デフォルト: 30）

        Returns:
            DEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("DEMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.DEMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"DEMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("DEMA", e)
            raise TALibCalculationError(f"DEMA計算失敗: {e}")

    @staticmethod
    def kama(data: pd.Series, period: int = 30) -> pd.Series:
        """
        Kaufman Adaptive Moving Average (KAMA) を計算

        Args:
            data: 価格データ（pandas Series）
            period: KAMAの期間（デフォルト: 30）

        Returns:
            KAMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("KAMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.KAMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"KAMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("KAMA", e)
            raise TALibCalculationError(f"KAMA計算失敗: {e}")

    @staticmethod
    def t3(data: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
        """
        Triple Exponential Moving Average (T3) を計算

        Args:
            data: 価格データ（pandas Series）
            period: T3の期間（デフォルト: 5）
            vfactor: ボリュームファクター（デフォルト: 0.7）

        Returns:
            T3値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("T3", period=period, vfactor=vfactor)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.T3, data.values, timeperiod=period, vfactor=vfactor
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"T3_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("T3", e)
            raise TALibCalculationError(f"T3計算失敗: {e}")

    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """
        Weighted Moving Average (加重移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            WMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("WMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.WMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"WMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("WMA", e)
            raise TALibCalculationError(f"WMA計算失敗: {e}")

    @staticmethod
    def trima(data: pd.Series, period: int) -> pd.Series:
        """
        Triangular Moving Average (三角移動平均) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            TRIMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("TRIMA", period=period)

        try:
            result = TrendAdapter._safe_talib_calculation(
                talib.TRIMA, data.values, timeperiod=period
            )
            return TrendAdapter._create_series_result(
                result, data.index, f"TRIMA_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("TRIMA", e)
            raise TALibCalculationError(f"TRIMA計算失敗: {e}")

    @staticmethod
    def mama(data: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05) -> Dict[str, pd.Series]:
        """
        MESA Adaptive Moving Average (MAMA) を計算

        Args:
            data: 価格データ（pandas Series）
            fast_limit: 高速制限（デフォルト: 0.5）
            slow_limit: 低速制限（デフォルト: 0.05）

        Returns:
            MAMA値を含む辞書（mama, fama）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        if len(data) < 32:  # MAMAは最低32データポイントが必要
            raise TALibCalculationError(f"MAMAには最低32データポイントが必要です: {len(data)}")
        
        TrendAdapter._log_calculation_start("MAMA", fast_limit=fast_limit, slow_limit=slow_limit)

        try:
            mama, fama = TrendAdapter._safe_talib_calculation(
                talib.MAMA, data.values, fastlimit=fast_limit, slowlimit=slow_limit
            )

            return {
                "mama": pd.Series(mama, index=data.index, name="MAMA"),
                "fama": pd.Series(fama, index=data.index, name="FAMA"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("MAMA", e)
            raise TALibCalculationError(f"MAMA計算失敗: {e}")
