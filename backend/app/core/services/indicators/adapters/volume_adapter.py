"""
ボリューム系テクニカル指標アダプター

出来高ベースの指標計算を担当します。
AD, ADOSC, OBV などのボリューム系指標を提供します。
"""

import talib
import pandas as pd
import logging

from .base_adapter import BaseAdapter, TALibCalculationError

logger = logging.getLogger(__name__)


class VolumeAdapter(BaseAdapter):
    """ボリューム系指標のTA-Libアダプター"""

    @staticmethod
    def ad(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Chaikin A/D Line (Accumulation/Distribution Line) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            A/D Line値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_multi_input(high, low, close, volume)
        VolumeAdapter._log_calculation_start("AD")

        try:
            result = VolumeAdapter._safe_talib_calculation(
                talib.AD, high.values, low.values, close.values, volume.values
            )
            return VolumeAdapter._create_series_result(
                result, close.index, "AD"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("AD", e)
            raise TALibCalculationError(f"A/D Line計算失敗: {e}")

    @staticmethod
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> pd.Series:
        """
        Chaikin A/D Oscillator を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            fast_period: 短期期間（デフォルト: 3）
            slow_period: 長期期間（デフォルト: 10）

        Returns:
            A/D Oscillator値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_multi_input(high, low, close, volume)
        VolumeAdapter._validate_input(close, slow_period)  # 長い期間で検証
        VolumeAdapter._log_calculation_start("ADOSC", fast_period=fast_period, slow_period=slow_period)

        try:
            result = VolumeAdapter._safe_talib_calculation(
                talib.ADOSC,
                high.values,
                low.values,
                close.values,
                volume.values,
                fastperiod=fast_period,
                slowperiod=slow_period,
            )
            return VolumeAdapter._create_series_result(
                result, close.index, f"ADOSC_{fast_period}_{slow_period}"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("ADOSC", e)
            raise TALibCalculationError(f"A/D Oscillator計算失敗: {e}")

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume (OBV) を計算

        Args:
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            OBV値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_multi_input(close, volume)
        VolumeAdapter._log_calculation_start("OBV")

        try:
            result = VolumeAdapter._safe_talib_calculation(
                talib.OBV, close.values, volume.values
            )
            return VolumeAdapter._create_series_result(
                result, close.index, "OBV"
            )

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("OBV", e)
            raise TALibCalculationError(f"OBV計算失敗: {e}")

    @staticmethod
    def ad_line(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Accumulation/Distribution Line を計算（adのエイリアス）

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            A/D Line値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        return VolumeAdapter.ad(high, low, close, volume)

    @staticmethod
    def chaikin_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> pd.Series:
        """
        Chaikin Oscillator を計算（adoscのエイリアス）

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            fast_period: 短期期間（デフォルト: 3）
            slow_period: 長期期間（デフォルト: 10）

        Returns:
            Chaikin Oscillator値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        return VolumeAdapter.adosc(high, low, close, volume, fast_period, slow_period)

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume を計算（obvのエイリアス）

        Args:
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            OBV値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        return VolumeAdapter.obv(close, volume)
