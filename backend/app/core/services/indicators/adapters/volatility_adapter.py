"""
ボラティリティ系テクニカル指標アダプター

ボラティリティやレンジ系の指標計算を担当します。
ATR, Bollinger Bands, NATR, TRANGE などのボラティリティ系指標を提供します。
"""

import talib
import pandas as pd
from typing import Dict
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class VolatilityAdapter(BaseAdapter):
    """ボラティリティ系指標のTA-Libアダプター"""

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average True Range (平均真の値幅) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: ATRの期間（デフォルト: 14）

        Returns:
            ATR値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_multi_input(high, low, close)
        VolatilityAdapter._validate_input(close, period)
        VolatilityAdapter._log_calculation_start("ATR", period=period)

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.ATR, high.values, low.values, close.values, timeperiod=period
            )
            return VolatilityAdapter._create_series_result(
                result, close.index, f"ATR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("ATR", e)
            raise TALibCalculationError(f"ATR計算失敗: {e}")

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands (ボリンジャーバンド) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間（デフォルト: 20）
            std_dev: 標準偏差の倍数（デフォルト: 2.0）

        Returns:
            ボリンジャーバンド値を含む辞書（upper, middle, lower）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(data, period)
        VolatilityAdapter._log_calculation_start("BBANDS", period=period, std_dev=std_dev)

        try:
            upper, middle, lower = VolatilityAdapter._safe_talib_calculation(
                talib.BBANDS,
                data.values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
            )

            return {
                "upper": pd.Series(upper, index=data.index, name=f"BB_Upper_{period}"),
                "middle": pd.Series(
                    middle, index=data.index, name=f"BB_Middle_{period}"
                ),
                "lower": pd.Series(lower, index=data.index, name=f"BB_Lower_{period}"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("BBANDS", e)
            raise TALibCalculationError(f"ボリンジャーバンド計算失敗: {e}")

    @staticmethod
    def natr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Normalized Average True Range (NATR) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: NATRの期間（デフォルト: 14）

        Returns:
            NATR値のpandas Series（パーセンテージ）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_multi_input(high, low, close)
        VolatilityAdapter._validate_input(close, period)
        VolatilityAdapter._log_calculation_start("NATR", period=period)

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.NATR, high.values, low.values, close.values, timeperiod=period
            )
            return VolatilityAdapter._create_series_result(
                result, close.index, f"NATR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("NATR", e)
            raise TALibCalculationError(f"NATR計算失敗: {e}")

    @staticmethod
    def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        True Range (真の値幅) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）

        Returns:
            True Range値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_multi_input(high, low, close)
        VolatilityAdapter._log_calculation_start("TRANGE")

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.TRANGE, high.values, low.values, close.values
            )
            return VolatilityAdapter._create_series_result(
                result, close.index, "TRANGE"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("TRANGE", e)
            raise TALibCalculationError(f"True Range計算失敗: {e}")

    @staticmethod
    def stddev(data: pd.Series, period: int = 5, nbdev: float = 1.0) -> pd.Series:
        """
        Standard Deviation (標準偏差) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差の倍数（デフォルト: 1.0）

        Returns:
            標準偏差値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(data, period)
        VolatilityAdapter._log_calculation_start("STDDEV", period=period, nbdev=nbdev)

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.STDDEV, data.values, timeperiod=period, nbdev=nbdev
            )
            return VolatilityAdapter._create_series_result(
                result, data.index, f"STDDEV_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("STDDEV", e)
            raise TALibCalculationError(f"標準偏差計算失敗: {e}")

    @staticmethod
    def var(data: pd.Series, period: int = 5, nbdev: float = 1.0) -> pd.Series:
        """
        Variance (分散) を計算

        Args:
            data: 価格データ（pandas Series）
            period: 期間（デフォルト: 5）
            nbdev: 標準偏差の倍数（デフォルト: 1.0）

        Returns:
            分散値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(data, period)
        VolatilityAdapter._log_calculation_start("VAR", period=period, nbdev=nbdev)

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.VAR, data.values, timeperiod=period, nbdev=nbdev
            )
            return VolatilityAdapter._create_series_result(
                result, data.index, f"VAR_{period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("VAR", e)
            raise TALibCalculationError(f"分散計算失敗: {e}")
