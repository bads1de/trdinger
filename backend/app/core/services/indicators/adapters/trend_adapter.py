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
    def hma(data: pd.Series, period: int) -> pd.Series:
        """
        Hull Moving Average (ハル移動平均) を計算

        HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

        Hull Moving Averageは、Alan Hullによって開発された移動平均で、
        従来の移動平均のラグを大幅に削減しながら滑らかさを保つ特徴があります。

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            HMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        import math

        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("HMA", period=period)

        try:
            # HMAの計算パラメータ
            half_period = max(1, int(period / 2))  # 最小値1を保証
            sqrt_period = max(1, int(math.sqrt(period)))  # 最小値1を保証

            # 最小データ数の確認（より保守的に）
            min_required = period + sqrt_period + 5  # 余裕を持たせる
            if len(data) < min_required:
                raise TALibCalculationError(
                    f"HMA計算には最低{min_required}個のデータが必要です（現在: {len(data)}個）"
                )

            # Step 1: WMA(n/2) を計算
            wma_half = TrendAdapter.wma(data, half_period)

            # Step 2: WMA(n) を計算
            wma_full = TrendAdapter.wma(data, period)

            # Step 3: 2 * WMA(n/2) - WMA(n) を計算
            # この計算により、ラグを削減する
            diff_series = 2 * wma_half - wma_full

            # NaNを除去してからWMAを計算
            diff_series_clean = diff_series.dropna()

            if len(diff_series_clean) < sqrt_period:
                raise TALibCalculationError(
                    f"HMA計算の中間結果が不足しています（必要: {sqrt_period}個、実際: {len(diff_series_clean)}個）"
                )

            # Step 4: WMA(2 * WMA(n/2) - WMA(n), sqrt(n)) を計算
            # 最終的な平滑化を行う
            hma_result = TrendAdapter.wma(diff_series_clean, sqrt_period)

            # 元のインデックスに合わせて結果を調整
            result = pd.Series(index=data.index, dtype=float, name=f"HMA_{period}")

            # HMA結果を元のインデックスにマッピング
            for idx in hma_result.index:
                if idx in result.index:
                    result.loc[idx] = hma_result.loc[idx]

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("HMA", e)
            raise TALibCalculationError(f"HMA計算失敗: {e}")

    @staticmethod
    def mama(data: pd.Series, fastlimit: float = 0.5, slowlimit: float = 0.05) -> dict:
        """
        MESA Adaptive Moving Average (MAMA) を計算

        Args:
            data: 価格データ（pandas Series）
            fastlimit: 高速制限（デフォルト: 0.5）
            slowlimit: 低速制限（デフォルト: 0.05）

        Returns:
            MAMA値を含む辞書（mama, fama）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, 32)  # MAMAには最低32個のデータが必要

        # パラメータ検証
        if fastlimit <= 0 or fastlimit > 1:
            raise TALibCalculationError(
                f"fastlimitは0より大きく1以下である必要があります: {fastlimit}"
            )
        if slowlimit <= 0 or slowlimit > 1:
            raise TALibCalculationError(
                f"slowlimitは0より大きく1以下である必要があります: {slowlimit}"
            )
        if slowlimit >= fastlimit:
            raise TALibCalculationError(
                f"slowlimitはfastlimitより小さい必要があります: slow={slowlimit}, fast={fastlimit}"
            )

        TrendAdapter._log_calculation_start(
            "MAMA", fastlimit=fastlimit, slowlimit=slowlimit
        )

        try:
            mama, fama = TrendAdapter._safe_talib_calculation(
                talib.MAMA, data.values, fastlimit=fastlimit, slowlimit=slowlimit
            )

            return {
                "mama": TrendAdapter._create_series_result(mama, data.index, "MAMA"),
                "fama": TrendAdapter._create_series_result(fama, data.index, "FAMA"),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("MAMA", e)
            raise TALibCalculationError(f"MAMA計算失敗: {e}")

    @staticmethod
    def vwma(price_data: pd.Series, volume_data: pd.Series, period: int) -> pd.Series:
        """
        Volume Weighted Moving Average (出来高加重移動平均) を計算

        VWMA = Σ(Price × Volume) / Σ(Volume) for period

        Args:
            price_data: 価格データ（pandas Series）
            volume_data: 出来高データ（pandas Series）
            period: 移動平均の期間

        Returns:
            VWMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(price_data, period)
        TrendAdapter._log_calculation_start("VWMA", period=period)

        try:
            # データの長さチェック
            if len(price_data) != len(volume_data):
                raise TALibCalculationError(
                    f"価格データと出来高データの長さが一致しません（価格: {len(price_data)}, 出来高: {len(volume_data)}）"
                )

            # 最小データ数の確認
            if len(price_data) < period:
                raise TALibCalculationError(
                    f"VWMA計算には最低{period}個のデータが必要です（現在: {len(price_data)}個）"
                )

            # 出来高データの検証
            if volume_data.isna().any():
                raise TALibCalculationError("出来高データにNaN値が含まれています")

            if (volume_data <= 0).any():
                raise TALibCalculationError("出来高データに0以下の値が含まれています")

            # VWMA計算
            # 価格 × 出来高の積
            price_volume = price_data * volume_data

            # 期間ごとの移動合計を計算
            price_volume_sum = price_volume.rolling(
                window=period, min_periods=period
            ).sum()
            volume_sum = volume_data.rolling(window=period, min_periods=period).sum()

            # VWMA = Σ(Price × Volume) / Σ(Volume)
            vwma_result = price_volume_sum / volume_sum

            # 結果のSeries作成
            result = pd.Series(
                vwma_result.values, index=price_data.index, name=f"VWMA_{period}"
            )

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("VWMA", e)
            raise TALibCalculationError(f"VWMA計算失敗: {e}")

    @staticmethod
    def zlema(data: pd.Series, period: int) -> pd.Series:
        """
        Zero Lag Exponential Moving Average (ZLEMA) を計算

        ZLEMA = EMA(data + (data - data[lag]), period)
        lag = (period - 1) / 2

        ZLEMAは、John Ehlers によって開発された指標で、
        従来のEMAのラグを削減することを目的としています。

        Args:
            data: 価格データ（pandas Series）
            period: 移動平均の期間

        Returns:
            ZLEMA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        TrendAdapter._validate_input(data, period)
        TrendAdapter._log_calculation_start("ZLEMA", period=period)

        try:
            # ラグの計算
            lag = int((period - 1) / 2)

            # 最小データ数の確認
            min_required = period + lag + 5  # 余裕を持たせる
            if len(data) < min_required:
                raise TALibCalculationError(
                    f"ZLEMA計算には最低{min_required}個のデータが必要です（現在: {len(data)}個）"
                )

            # ラグ調整されたデータの計算
            # ZLEMA = EMA(data + (data - data[lag]), period)
            lagged_data = data.shift(lag)
            adjusted_data = data + (data - lagged_data)

            # NaN値を除去
            adjusted_data_clean = adjusted_data.dropna()

            if len(adjusted_data_clean) < period:
                raise TALibCalculationError(
                    f"ZLEMA計算の調整後データが不足しています（必要: {period}個、実際: {len(adjusted_data_clean)}個）"
                )

            # EMAを計算
            zlema_result = TrendAdapter.ema(adjusted_data_clean, period)

            # 元のインデックスに合わせて結果を調整
            result = pd.Series(index=data.index, dtype=float, name=f"ZLEMA_{period}")

            # ZLEMA結果を元のインデックスにマッピング
            for idx in zlema_result.index:
                if idx in result.index:
                    result.loc[idx] = zlema_result.loc[idx]

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            TrendAdapter._log_calculation_error("ZLEMA", e)
            raise TALibCalculationError(f"ZLEMA計算失敗: {e}")

    @staticmethod
    def mama(
        data: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05
    ) -> Dict[str, pd.Series]:
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
            raise TALibCalculationError(
                f"MAMAには最低32データポイントが必要です: {len(data)}"
            )

        TrendAdapter._log_calculation_start(
            "MAMA", fast_limit=fast_limit, slow_limit=slow_limit
        )

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
