"""
ボリューム系テクニカル指標アダプター

出来高ベースの指標計算を担当します。
AD, ADOSC, OBV などのボリューム系指標を提供します。
"""

import talib
import pandas as pd
import numpy as np
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
            return VolumeAdapter._create_series_result(result, close.index, "AD")

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
        VolumeAdapter._log_calculation_start(
            "ADOSC", fast_period=fast_period, slow_period=slow_period
        )

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
            return VolumeAdapter._create_series_result(result, close.index, "OBV")

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

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Volume Weighted Average Price (出来高加重平均価格) を計算

        VWAP = Σ(Typical Price × Volume) / Σ(Volume) for period
        Typical Price = (High + Low + Close) / 3

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: 移動平均の期間

        Returns:
            VWAP値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_input(close, period)
        VolumeAdapter._log_calculation_start("VWAP", period=period)

        try:
            # データの長さチェック
            data_lengths = [len(high), len(low), len(close), len(volume)]
            if len(set(data_lengths)) > 1:
                raise TALibCalculationError(
                    f"全てのデータの長さが一致しません（高値: {len(high)}, 安値: {len(low)}, 終値: {len(close)}, 出来高: {len(volume)}）"
                )

            # 最小データ数の確認
            if len(close) < period:
                raise TALibCalculationError(
                    f"VWAP計算には最低{period}個のデータが必要です（現在: {len(close)}個）"
                )

            # 出来高データの検証
            if volume.isna().any():
                raise TALibCalculationError("出来高データにNaN値が含まれています")

            if (volume <= 0).any():
                raise TALibCalculationError("出来高データに0以下の値が含まれています")

            # Typical Price = (High + Low + Close) / 3
            typical_price = (high + low + close) / 3

            # VWAP計算
            # Typical Price × Volume
            price_volume = typical_price * volume

            # 期間ごとの移動合計を計算
            price_volume_sum = price_volume.rolling(
                window=period, min_periods=period
            ).sum()
            volume_sum = volume.rolling(window=period, min_periods=period).sum()

            # VWAP = Σ(Typical Price × Volume) / Σ(Volume)
            vwap_result = price_volume_sum / volume_sum

            # 結果のSeries作成
            result = pd.Series(
                vwap_result.values, index=close.index, name=f"VWAP_{period}"
            )

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("VWAP", e)
            raise TALibCalculationError(f"VWAP計算失敗: {e}")

    @staticmethod
    def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Price Volume Trend (PVT) を計算

        PVT = Previous PVT + (Volume × ((Close - Previous Close) / Previous Close))

        PVTは、価格変化率と出来高を組み合わせた累積指標です。
        OBVと似ていますが、価格変化の大きさも考慮に入れます。

        Args:
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            PVT値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_multi_input(close, volume)
        VolumeAdapter._log_calculation_start("PVT")

        try:
            # データの長さチェック
            if len(close) != len(volume):
                raise TALibCalculationError(
                    f"終値と出来高データの長さが一致しません（終値: {len(close)}, 出来高: {len(volume)}）"
                )

            # 最小データ数の確認（PVTは前日比較が必要なので最低2個）
            if len(close) < 2:
                raise TALibCalculationError(
                    f"PVT計算には最低2個のデータが必要です（現在: {len(close)}個）"
                )

            # 出来高データの検証
            if volume.isna().any():
                raise TALibCalculationError("出来高データにNaN値が含まれています")

            if (volume < 0).any():
                raise TALibCalculationError("出来高データに負の値が含まれています")

            # 価格変化率の計算
            # (Close - Previous Close) / Previous Close
            price_change_rate = close.pct_change()

            # PVT計算
            # Volume × Price Change Rate
            pvt_increment = volume * price_change_rate

            # 累積合計（最初の値は0から開始）
            pvt_result = pvt_increment.cumsum()

            # 最初の値をNaNから0に変更
            pvt_result.iloc[0] = 0

            # 結果のSeries作成
            result = pd.Series(pvt_result.values, index=close.index, name="PVT")

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("PVT", e)
            raise TALibCalculationError(f"PVT計算失敗: {e}")

    @staticmethod
    def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Price Volume Trend を計算（pvtのエイリアス）

        Args:
            close: 終値データ（pandas Series）
            volume: 出来高データ（pandas Series）

        Returns:
            PVT値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        return VolumeAdapter.pvt(close, volume)

    @staticmethod
    def emv(
        high: pd.Series, low: pd.Series, volume: pd.Series, period: int
    ) -> pd.Series:
        """
        Ease of Movement (EMV) を計算

        EMV = SMA(Distance Moved / Box Height, period)
        Distance Moved = (High + Low) / 2 - (Previous High + Previous Low) / 2
        Box Height = Volume / (High - Low)

        EMVは、価格変動の容易さを測定する指標です。
        出来高が少なく価格変動が大きい場合、移動が容易であることを示します。

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: 移動平均の期間

        Returns:
            EMV値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolumeAdapter._validate_multi_input(high, low, volume)
        VolumeAdapter._log_calculation_start("EMV", period=period)

        try:
            # データの長さチェック
            if not (len(high) == len(low) == len(volume)):
                raise TALibCalculationError(
                    f"高値、安値、出来高データの長さが一致しません"
                    f"（高値: {len(high)}, 安値: {len(low)}, 出来高: {len(volume)}）"
                )

            # 最小データ数の確認
            min_required = period + 5  # 余裕を持たせる
            if len(high) < min_required:
                raise TALibCalculationError(
                    f"EMV計算には最低{min_required}個のデータが必要です（現在: {len(high)}個）"
                )

            # 出来高データの検証
            if volume.isna().any():
                raise TALibCalculationError("出来高データにNaN値が含まれています")

            if (volume <= 0).any():
                raise TALibCalculationError("出来高データに0以下の値が含まれています")

            # Distance Moved の計算
            # (High + Low) / 2 の移動距離
            mid_point = (high + low) / 2
            distance_moved = mid_point.diff()

            # Box Height の計算
            # Volume / (High - Low)
            price_range = high - low

            # 価格レンジが0の場合の処理
            price_range = price_range.replace(0, np.nan)
            box_height = volume / price_range

            # EMV Raw の計算
            # Distance Moved / Box Height
            emv_raw = distance_moved / box_height

            # 無限大値の処理
            emv_raw = emv_raw.replace([np.inf, -np.inf], np.nan)

            # 移動平均の計算
            emv_result = emv_raw.rolling(window=period, min_periods=period).mean()

            # 結果のSeries作成
            result = pd.Series(
                emv_result.values, index=high.index, name=f"EMV_{period}"
            )

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            VolumeAdapter._log_calculation_error("EMV", e)
            raise TALibCalculationError(f"EMV計算失敗: {e}")

    @staticmethod
    def ease_of_movement(
        high: pd.Series, low: pd.Series, volume: pd.Series, period: int
    ) -> pd.Series:
        """
        Ease of Movement を計算（emvのエイリアス）

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            volume: 出来高データ（pandas Series）
            period: 移動平均の期間

        Returns:
            EMV値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        return VolumeAdapter.emv(high, low, volume, period)
