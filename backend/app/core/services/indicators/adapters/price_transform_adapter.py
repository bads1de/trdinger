"""
価格変換系指標のTA-Libアダプター

AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE の計算を提供します。
"""

import pandas as pd
import talib
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class PriceTransformAdapter(BaseAdapter):
    """価格変換系指標のTA-Libアダプター"""

    @staticmethod
    def avgprice(
        open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        AVGPRICE（Average Price）を計算

        Args:
            open_prices: 始値のSeries
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries

        Returns:
            AVGPRICE値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        PriceTransformAdapter._log_calculation_start("AVGPRICE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.AVGPRICE,
                open_prices.values,
                high.values,
                low.values,
                close.values,
            )
            return PriceTransformAdapter._create_series_result_with_config(
                result, close.index, "AVGPRICE", {}
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("AVGPRICE", e)
            raise TALibCalculationError(f"AVGPRICE計算失敗: {e}")

    @staticmethod
    def medprice(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        MEDPRICE（Median Price）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries

        Returns:
            MEDPRICE値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        PriceTransformAdapter._validate_input(high, 1)
        PriceTransformAdapter._log_calculation_start("MEDPRICE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.MEDPRICE, high.values, low.values
            )
            return PriceTransformAdapter._create_series_result_with_config(
                result, high.index, "MEDPRICE", {}
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("MEDPRICE", e)
            raise TALibCalculationError(f"MEDPRICE計算失敗: {e}")

    @staticmethod
    def typprice(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        TYPPRICE（Typical Price）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries

        Returns:
            TYPPRICE値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("TYPPRICE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.TYPPRICE, high.values, low.values, close.values
            )
            return PriceTransformAdapter._create_series_result_with_config(
                result, close.index, "TYPPRICE", {}
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("TYPPRICE", e)
            raise TALibCalculationError(f"TYPPRICE計算失敗: {e}")

    @staticmethod
    def wclprice(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        WCLPRICE（Weighted Close Price）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries
            close: 終値のSeries

        Returns:
            WCLPRICE値のSeries

        Raises:
            TALibCalculationError: TA-Lib計算エラーの場合
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("WCLPRICE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.WCLPRICE, high.values, low.values, close.values
            )
            return PriceTransformAdapter._create_series_result_with_config(
                result, close.index, "WCLPRICE", {}
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("WCLPRICE", e)
            raise TALibCalculationError(f"WCLPRICE計算失敗: {e}")
