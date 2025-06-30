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

            return VolatilityAdapter._create_series_result(result, close.index, "ATR")

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
        VolatilityAdapter._log_calculation_start(
            "BBANDS", period=period, std_dev=std_dev
        )

        try:
            upper, middle, lower = VolatilityAdapter._safe_talib_calculation(
                talib.BBANDS,
                data.values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
            )

            return {
                "upper": pd.Series(upper, index=data.index, name="BB_Upper"),
                "middle": pd.Series(middle, index=data.index, name="BB_Middle"),
                "lower": pd.Series(lower, index=data.index, name="BB_Lower"),
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
            return VolatilityAdapter._create_series_result(result, close.index, "NATR")

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
            return VolatilityAdapter._create_series_result(result, data.index, "STDDEV")

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
            return VolatilityAdapter._create_series_result(result, data.index, "VAR")

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("VAR", e)
            raise TALibCalculationError(f"分散計算失敗: {e}")

    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """
        Keltner Channels (ケルトナーチャネル) を計算

        Keltner Channels = EMA(Close) ± (multiplier × ATR)

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            close: 終値データ（pandas Series）
            period: 期間
            multiplier: ATRの倍数（デフォルト: 2.0）

        Returns:
            Keltner Channelsのpandas DataFrame (upper, middle, lower)

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(close, period)
        VolatilityAdapter._log_calculation_start(
            "KELTNER", period=period, multiplier=multiplier
        )

        try:
            # データの長さチェック
            data_lengths = [len(high), len(low), len(close)]
            if len(set(data_lengths)) > 1:
                raise TALibCalculationError(
                    f"全てのデータの長さが一致しません（高値: {len(high)}, 安値: {len(low)}, 終値: {len(close)}）"
                )

            # 最小データ数の確認
            if len(close) < period:
                raise TALibCalculationError(
                    f"Keltner Channels計算には最低{period}個のデータが必要です（現在: {len(close)}個）"
                )

            # Middle Line: EMA(Close)
            middle_line = VolatilityAdapter._safe_talib_calculation(
                talib.EMA, close.values, timeperiod=period
            )

            # ATR計算
            atr_values = VolatilityAdapter._safe_talib_calculation(
                talib.ATR, high.values, low.values, close.values, timeperiod=period
            )

            # Upper and Lower Lines
            upper_line = middle_line + (multiplier * atr_values)
            lower_line = middle_line - (multiplier * atr_values)

            # 結果のDataFrame作成
            result = pd.DataFrame(
                {
                    "upper": VolatilityAdapter._create_series_result(
                        upper_line, close.index, "KELTNER_Upper"
                    ),
                    "middle": VolatilityAdapter._create_series_result(
                        middle_line, close.index, "KELTNER_Middle"
                    ),
                    "lower": VolatilityAdapter._create_series_result(
                        lower_line, close.index, "KELTNER_Lower"
                    ),
                },
                index=close.index,
            )

            return result

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("KELTNER", e)
            raise TALibCalculationError(f"Keltner Channels計算失敗: {e}")

    @staticmethod
    def psar(
        high: pd.Series,
        low: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> pd.Series:
        """
        Parabolic SAR (パラボリックSAR) を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            acceleration: 加速因子（デフォルト: 0.02）
            maximum: 最大加速因子（デフォルト: 0.2）

        Returns:
            PSAR値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_multi_input(
            high, low, high
        )  # closeの代わりにhighを使用
        VolatilityAdapter._log_calculation_start(
            "PSAR", acceleration=acceleration, maximum=maximum
        )

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.SAR,
                high.values,
                low.values,
                acceleration=acceleration,
                maximum=maximum,
            )
            return VolatilityAdapter._create_series_result(result, high.index, "PSAR")

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("PSAR", e)
            raise TALibCalculationError(f"PSAR計算失敗: {e}")

    @staticmethod
    def donchian_channels(
        high: pd.Series, low: pd.Series, period: int
    ) -> Dict[str, pd.Series]:
        """
        Donchian Channels (ドンチャンチャネル) を計算

        Donchian Channelsは、指定期間の最高値と最低値を使用したチャネル指標です。
        - Upper Channel: 指定期間の最高値
        - Lower Channel: 指定期間の最低値
        - Middle Channel: (Upper + Lower) / 2

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: 期間

        Returns:
            Donchian Channelsを含む辞書（upper, lower, middle）

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(high, period)
        VolatilityAdapter._log_calculation_start("DONCHIAN", period=period)

        try:
            # データの長さチェック
            if len(high) != len(low):
                raise TALibCalculationError(
                    f"高値と安値データの長さが一致しません（高値: {len(high)}, 安値: {len(low)}）"
                )

            # 最小データ数の確認
            if len(high) < period:
                raise TALibCalculationError(
                    f"Donchian Channels計算には最低{period}個のデータが必要です（現在: {len(high)}個）"
                )

            # Upper Channel: 指定期間の最高値
            upper_channel = high.rolling(window=period, min_periods=period).max()

            # Lower Channel: 指定期間の最低値
            lower_channel = low.rolling(window=period, min_periods=period).min()

            # Middle Channel: (Upper + Lower) / 2
            middle_channel = (upper_channel + lower_channel) / 2

            return {
                "upper": VolatilityAdapter._create_series_result(
                    upper_channel.to_numpy(),
                    high.index,
                    "DONCHIAN_Upper",
                ),
                "lower": VolatilityAdapter._create_series_result(
                    lower_channel.to_numpy(),
                    low.index,
                    "DONCHIAN_Lower",
                ),
                "middle": VolatilityAdapter._create_series_result(
                    middle_channel.to_numpy(),
                    high.index,
                    "DONCHIAN_Middle",
                ),
            }

        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("DONCHIAN", e)
            raise TALibCalculationError(f"Donchian Channels計算失敗: {e}")

    @staticmethod
    def beta(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """
        BETA（Beta）を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: 期間

        Returns:
            BETA値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(high, period)
        VolatilityAdapter._log_calculation_start("BETA")

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.BETA, high.values, low.values, timeperiod=period
            )
            return VolatilityAdapter._create_series_result(result, high.index, "BETA")
        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("BETA", e)
            raise TALibCalculationError(f"BETA計算失敗: {e}")

    @staticmethod
    def correl(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """
        CORREL（Correlation Coefficient）を計算

        Args:
            high: 高値データ（pandas Series）
            low: 安値データ（pandas Series）
            period: 期間

        Returns:
            CORREL値のpandas Series

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        VolatilityAdapter._validate_input(high, period)
        VolatilityAdapter._log_calculation_start("CORREL")

        try:
            result = VolatilityAdapter._safe_talib_calculation(
                talib.CORREL, high.values, low.values, timeperiod=period
            )
            return VolatilityAdapter._create_series_result(result, high.index, "CORREL")
        except TALibCalculationError:
            raise
        except Exception as e:
            VolatilityAdapter._log_calculation_error("CORREL", e)
            raise TALibCalculationError(f"CORREL計算失敗: {e}")
