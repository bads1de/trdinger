"""
価格特徴量計算クラス

OHLCV価格データから基本的な価格関連特徴量を計算します。
単一責任原則に従い、価格特徴量の計算のみを担当します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.unified_error_handler import safe_ml_operation
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class PriceFeatureCalculator(BaseFeatureCalculator):
    """
    価格特徴量計算クラス

    OHLCV価格データから基本的な価格関連特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        価格特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            価格特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})
        return self.calculate_price_features(df, lookback_periods)

    @safe_ml_operation(
        default_return=None, context="価格特徴量計算でエラーが発生しました"
    )
    def calculate_price_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        価格特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            価格特徴量が追加されたDataFrame
        """
        if not self.validate_input_data(df, ["Close", "Open", "High", "Low"]):
            return df

        result_df = self.create_result_dataframe(df)

        # 移動平均比率
        short_ma = lookback_periods.get("short_ma", 10)
        long_ma = lookback_periods.get("long_ma", 50)

        result_df[f"MA_{short_ma}"] = self.safe_rolling_mean_calculation(
            result_df["Close"], window=short_ma
        )
        result_df[f"MA_{long_ma}"] = self.safe_rolling_mean_calculation(
            result_df["Close"], window=long_ma
        )

        result_df["Price_MA_Ratio_Short"] = self.safe_ratio_calculation(
            result_df["Close"], result_df[f"MA_{short_ma}"], default_value=1.0
        )
        result_df["Price_MA_Ratio_Long"] = self.safe_ratio_calculation(
            result_df["Close"], result_df[f"MA_{long_ma}"], default_value=1.0
        )

        # 価格モメンタム（安全な計算）
        momentum_period = lookback_periods.get("momentum", 14)
        result_df["Price_Momentum_14"] = self.safe_pct_change_calculation(
            result_df["Close"].shift(momentum_period)
        )

        # 高値・安値ポジション
        result_df["High_Low_Position"] = self.safe_ratio_calculation(
            result_df["Close"] - result_df["Low"],
            result_df["High"] - result_df["Low"],
            default_value=0.5,
        )

        # 価格変化率（安全な計算）
        result_df["Price_Change_1"] = self.safe_pct_change_calculation(
            result_df["Close"]
        )
        result_df["Price_Change_5"] = self.safe_ratio_calculation(
            result_df["Close"] - result_df["Close"].shift(5),
            result_df["Close"].shift(5),
            default_value=0.0,
        )
        result_df["Price_Change_20"] = self.safe_ratio_calculation(
            result_df["Close"] - result_df["Close"].shift(20),
            result_df["Close"].shift(20),
            default_value=0.0,
        )

        # 価格レンジ
        result_df["Price_Range"] = self.safe_ratio_calculation(
            result_df["High"] - result_df["Low"], result_df["Close"], default_value=0.0
        )

        # ボディサイズ（実体の大きさ）
        result_df["Body_Size"] = self.safe_ratio_calculation(
            abs(result_df["Close"] - result_df["Open"]),
            result_df["Close"],
            default_value=0.0,
        )

        # 上ヒゲ・下ヒゲ
        result_df["Upper_Shadow"] = self.safe_ratio_calculation(
            result_df["High"] - np.maximum(result_df["Open"], result_df["Close"]),
            result_df["Close"],
            default_value=0.0,
        )
        result_df["Lower_Shadow"] = self.safe_ratio_calculation(
            np.minimum(result_df["Open"], result_df["Close"]) - result_df["Low"],
            result_df["Close"],
            default_value=0.0,
        )

        # 価格位置（期間内での相対位置）
        period = lookback_periods.get("volatility", 20)
        close_min = result_df["Close"].rolling(window=period, min_periods=1).min()
        close_max = result_df["Close"].rolling(window=period, min_periods=1).max()
        result_df["Price_Position"] = self.safe_ratio_calculation(
            result_df["Close"] - close_min, close_max - close_min, default_value=0.5
        )

        # ギャップ（前日終値との差）
        result_df["Gap"] = self.safe_ratio_calculation(
            result_df["Open"] - result_df["Close"].shift(1),
            result_df["Close"].shift(1),
            default_value=0.0,
        )

        logger.debug("価格特徴量計算完了")
        return result_df

    def calculate_volatility_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        ボラティリティ特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            ボラティリティ特徴量が追加されたDataFrame
        """
        try:
            if not self.validate_input_data(df, ["Close", "High", "Low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volatility_period = lookback_periods.get("volatility", 20)

            # リターンを計算（安全な計算）
            result_df["Returns"] = self.safe_pct_change_calculation(result_df["Close"])

            # 実現ボラティリティ（安全な計算）
            result_df["Realized_Volatility_20"] = self.safe_rolling_std_calculation(
                result_df["Returns"], window=volatility_period
            ) * np.sqrt(24)

            # ボラティリティスパイク（安全な計算）
            vol_ma = self.safe_rolling_mean_calculation(
                result_df["Realized_Volatility_20"], window=volatility_period
            )
            result_df["Volatility_Spike"] = self.safe_ratio_calculation(
                result_df["Realized_Volatility_20"], vol_ma, default_value=1.0
            )

            # ATR（Average True Range）
            # 最初の行のNaN値を避けるため、shift(1)の結果をfillnaで補完
            prev_close = result_df["Close"].shift(1).fillna(result_df["Close"])
            result_df["TR"] = np.maximum(
                result_df["High"] - result_df["Low"],
                np.maximum(
                    abs(result_df["High"] - prev_close),
                    abs(result_df["Low"] - prev_close),
                ),
            )
            result_df["ATR_20"] = self.safe_rolling_mean_calculation(
                result_df["TR"], window=volatility_period
            )

            # 正規化ATR（安全な計算）
            result_df["ATR_Normalized"] = self.safe_ratio_calculation(
                result_df["ATR_20"], result_df["Close"], default_value=0.01
            )

            # ボラティリティレジーム
            vol_quantile = (
                result_df["Realized_Volatility_20"]
                .rolling(window=volatility_period * 2, min_periods=1)
                .quantile(0.8)
            )
            result_df["High_Vol_Regime"] = (
                result_df["Realized_Volatility_20"] > vol_quantile
            ).astype(int)

            # ボラティリティ変化率（安全な計算）
            vol_change = self.safe_pct_change_calculation(
                result_df["Realized_Volatility_20"]
            )
            # 異常に大きな値をクリップ（±500%に制限）
            result_df["Vol_Change"] = self.clip_extreme_values(vol_change, -5.0, 5.0)

            self.log_feature_calculation_complete("ボラティリティ")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "ボラティリティ特徴量計算", df)

    def calculate_volume_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        出来高特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            出来高特徴量が追加されたDataFrame
        """
        try:
            if not self.validate_input_data(df, ["Volume", "Close", "High", "Low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volume_period = lookback_periods.get("volume", 20)

            # 出来高移動平均（安全な計算）
            volume_ma = self.safe_rolling_mean_calculation(
                result_df["Volume"], window=volume_period
            )
            # 異常に大きな値をクリップ（最大値を制限）
            volume_max = (
                result_df["Volume"].quantile(0.99) * 10
            )  # 99%分位点の10倍を上限とする
            result_df[f"Volume_MA_{volume_period}"] = np.clip(volume_ma, 0, volume_max)

            # 出来高比率
            result_df["Volume_Ratio"] = self.safe_ratio_calculation(
                result_df["Volume"],
                result_df[f"Volume_MA_{volume_period}"],
                default_value=1.0,
            )

            # 価格・出来高トレンド（安全な計算）
            price_change = self.safe_pct_change_calculation(result_df["Close"])
            volume_change = self.safe_pct_change_calculation(result_df["Volume"])
            result_df["Price_Volume_Trend"] = self.safe_multiply_calculation(
                price_change, volume_change
            )

            # 出来高加重平均価格（VWAP）
            typical_price = (
                result_df["High"] + result_df["Low"] + result_df["Close"]
            ) / 3
            result_df["VWAP"] = self.safe_ratio_calculation(
                (typical_price * result_df["Volume"])
                .rolling(window=volume_period)
                .sum(),
                result_df["Volume"].rolling(window=volume_period).sum(),
                default_value=np.nan,
            )
            # NaNになったVWAP値をtypical_priceで埋める
            result_df["VWAP"] = result_df["VWAP"].fillna(typical_price)

            # VWAPからの乖離
            result_df["VWAP_Deviation"] = self.safe_ratio_calculation(
                result_df["Close"] - result_df["VWAP"],
                result_df["VWAP"],
                default_value=0.0,
            )

            # 出来高スパイク
            vol_threshold = (
                result_df["Volume"].rolling(window=volume_period).quantile(0.9)
            )
            result_df["Volume_Spike"] = (result_df["Volume"] > vol_threshold).astype(
                int
            )

            # 出来高トレンド
            result_df["Volume_Trend"] = self.safe_ratio_calculation(
                result_df["Volume"].rolling(window=5).mean(),
                result_df["Volume"].rolling(window=volume_period).mean(),
                default_value=1.0,
            )

            self.log_feature_calculation_complete("出来高")
            return result_df

        except Exception as e:
            return self.handle_calculation_error(e, "出来高特徴量計算", df)

    def get_feature_names(self) -> list:
        """
        生成される価格特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 価格特徴量
            "Price_MA_Ratio_Short",
            "Price_MA_Ratio_Long",
            "Price_Momentum_14",
            "High_Low_Position",
            "Price_Change_1",
            "Price_Change_5",
            "Price_Change_20",
            "Price_Range",
            "Body_Size",
            "Upper_Shadow",
            "Lower_Shadow",
            "Price_Position",
            "Gap",
            # ボラティリティ特徴量
            "Realized_Volatility_20",
            "Volatility_Spike",
            "ATR_20",
            "ATR_Normalized",
            "High_Vol_Regime",
            "Vol_Change",
            # 出来高特徴量
            "Volume_Ratio",
            "Price_Volume_Trend",
            "VWAP_Deviation",
            "Volume_Spike",
            "Volume_Trend",
        ]
