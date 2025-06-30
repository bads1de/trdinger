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
        PriceTransformAdapter._validate_multi_input(open_prices, high, low, close)
        PriceTransformAdapter._log_calculation_start("AVGPRICE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.AVGPRICE,
                open_prices.values,
                high.values,
                low.values,
                close.values,
            )
            return PriceTransformAdapter._create_series_result(
                result, close.index, "AVGPRICE"
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
            return PriceTransformAdapter._create_series_result(
                result, high.index, "MEDPRICE"
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
            return PriceTransformAdapter._create_series_result(
                result, close.index, "TYPPRICE"
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
            return PriceTransformAdapter._create_series_result(
                result, close.index, "WCLPRICE"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("WCLPRICE", e)
            raise TALibCalculationError(f"WCLPRICE計算失敗: {e}")

    @staticmethod
    def ht_dcperiod(close: pd.Series) -> pd.Series:
        """
        HT_DCPERIOD（Hilbert Transform - Dominant Cycle Period）を計算

        Args:
            close: 終値のSeries

        Returns:
            HT_DCPERIOD値のSeries
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("HT_DCPERIOD")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.HT_DCPERIOD, close.values
            )
            return PriceTransformAdapter._create_series_result(
                result, close.index, "HT_DCPERIOD"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("HT_DCPERIOD", e)
            raise TALibCalculationError(f"HT_DCPERIOD計算失敗: {e}")

    @staticmethod
    def ht_dcphase(close: pd.Series) -> pd.Series:
        """
        HT_DCPHASE（Hilbert Transform - Dominant Cycle Phase）を計算

        Args:
            close: 終値のSeries

        Returns:
            HT_DCPHASE値のSeries
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("HT_DCPHASE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.HT_DCPHASE, close.values
            )
            return PriceTransformAdapter._create_series_result(
                result, close.index, "HT_DCPHASE"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("HT_DCPHASE", e)
            raise TALibCalculationError(f"HT_DCPHASE計算失敗: {e}")

    @staticmethod
    def ht_phasor(close: pd.Series) -> pd.Series:
        """
        HT_PHASOR（Hilbert Transform - Phasor Components）を計算

        Args:
            close: 終値のSeries

        Returns:
            HT_PHASOR値のSeries（最初の成分）
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("HT_PHASOR")

        try:
            inphase, quadrature = PriceTransformAdapter._safe_talib_calculation(
                talib.HT_PHASOR, close.values
            )
            # 最初の成分（inphase）を返す
            return PriceTransformAdapter._create_series_result(
                inphase, close.index, "HT_PHASOR"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("HT_PHASOR", e)
            raise TALibCalculationError(f"HT_PHASOR計算失敗: {e}")

    @staticmethod
    def ht_sine(close: pd.Series) -> pd.Series:
        """
        HT_SINE（Hilbert Transform - SineWave）を計算

        Args:
            close: 終値のSeries

        Returns:
            HT_SINE値のSeries（最初の成分）
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("HT_SINE")

        try:
            sine, leadsine = PriceTransformAdapter._safe_talib_calculation(
                talib.HT_SINE, close.values
            )
            # 最初の成分（sine）を返す
            return PriceTransformAdapter._create_series_result(
                sine, close.index, "HT_SINE"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("HT_SINE", e)
            raise TALibCalculationError(f"HT_SINE計算失敗: {e}")

    @staticmethod
    def ht_trendmode(close: pd.Series) -> pd.Series:
        """
        HT_TRENDMODE（Hilbert Transform - Trend vs Cycle Mode）を計算

        Args:
            close: 終値のSeries

        Returns:
            HT_TRENDMODE値のSeries
        """
        PriceTransformAdapter._validate_input(close, 1)
        PriceTransformAdapter._log_calculation_start("HT_TRENDMODE")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.HT_TRENDMODE, close.values
            )
            return PriceTransformAdapter._create_series_result(
                result, close.index, "HT_TRENDMODE"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("HT_TRENDMODE", e)
            raise TALibCalculationError(f"HT_TRENDMODE計算失敗: {e}")

    @staticmethod
    def fama(close: pd.Series, period: int) -> pd.Series:
        """
        FAMA（Following Adaptive Moving Average）を計算

        Args:
            close: 終値のSeries
            period: 期間

        Returns:
            FAMA値のSeries
        """
        PriceTransformAdapter._validate_input(close, period)
        PriceTransformAdapter._log_calculation_start("FAMA")

        try:
            # FAMAはMAMAの第2成分として計算される
            mama, fama = PriceTransformAdapter._safe_talib_calculation(
                talib.MAMA, close.values, fastlimit=0.5, slowlimit=0.05
            )
            return PriceTransformAdapter._create_series_result(
                fama, close.index, "FAMA"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("FAMA", e)
            raise TALibCalculationError(f"FAMA計算失敗: {e}")

    @staticmethod
    def sarext(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        SAREXT（Parabolic SAR - Extended）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries

        Returns:
            SAREXT値のSeries
        """
        PriceTransformAdapter._validate_input(high, 1)
        PriceTransformAdapter._log_calculation_start("SAREXT")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.SAREXT, high.values, low.values
            )
            return PriceTransformAdapter._create_series_result(
                result, high.index, "SAREXT"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("SAREXT", e)
            raise TALibCalculationError(f"SAREXT計算失敗: {e}")

    @staticmethod
    def sar(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        SAR（Parabolic SAR）を計算

        Args:
            high: 高値のSeries
            low: 安値のSeries

        Returns:
            SAR値のSeries
        """
        PriceTransformAdapter._validate_input(high, 1)
        PriceTransformAdapter._log_calculation_start("SAR")

        try:
            result = PriceTransformAdapter._safe_talib_calculation(
                talib.SAR, high.values, low.values
            )
            return PriceTransformAdapter._create_series_result(
                result, high.index, "SAR"
            )
        except TALibCalculationError:
            raise
        except Exception as e:
            PriceTransformAdapter._log_calculation_error("SAR", e)
            raise TALibCalculationError(f"SAR計算失敗: {e}")
