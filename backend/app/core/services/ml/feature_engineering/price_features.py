"""
価格特徴量計算クラス

OHLCV価格データから基本的な価格関連特徴量を計算します。
単一責任原則に従い、価格特徴量の計算のみを担当します。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from ....utils.ml_error_handler import safe_ml_operation
from ....utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class PriceFeatureCalculator:
    """
    価格特徴量計算クラス

    OHLCV価格データから基本的な価格関連特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        pass

    @safe_ml_operation(
        default_value=None, error_message="価格特徴量計算でエラーが発生しました"
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
        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        result_df = df.copy()

        # 移動平均比率
        short_ma = lookback_periods.get("short_ma", 10)
        long_ma = lookback_periods.get("long_ma", 50)

        result_df[f"MA_{short_ma}"] = DataValidator.safe_rolling_mean(
            result_df["Close"], window=short_ma
        )
        result_df[f"MA_{long_ma}"] = DataValidator.safe_rolling_mean(
            result_df["Close"], window=long_ma
        )

        result_df["Price_MA_Ratio_Short"] = DataValidator.safe_divide(
            result_df["Close"], result_df[f"MA_{short_ma}"], default_value=1.0
        )
        result_df["Price_MA_Ratio_Long"] = DataValidator.safe_divide(
            result_df["Close"], result_df[f"MA_{long_ma}"], default_value=1.0
        )

        # 価格モメンタム（安全な計算）
        momentum_period = lookback_periods.get("momentum", 14)
        result_df["Price_Momentum_14"] = DataValidator.safe_pct_change(
            result_df["Close"], periods=momentum_period
        )

        # 高値・安値ポジション
        result_df["High_Low_Position"] = (result_df["Close"] - result_df["Low"]) / (
            result_df["High"] - result_df["Low"] + 1e-8
        )

        # 価格変化率（安全な計算）
        result_df["Price_Change_1"] = DataValidator.safe_pct_change(
            result_df["Close"], periods=1
        )
        result_df["Price_Change_5"] = DataValidator.safe_pct_change(
            result_df["Close"], periods=5
        )
        result_df["Price_Change_20"] = DataValidator.safe_pct_change(
            result_df["Close"], periods=20
        )

        # 価格レンジ
        result_df["Price_Range"] = (result_df["High"] - result_df["Low"]) / result_df[
            "Close"
        ]

        # ボディサイズ（実体の大きさ）
        result_df["Body_Size"] = (
            abs(result_df["Close"] - result_df["Open"]) / result_df["Close"]
        )

        # 上ヒゲ・下ヒゲ
        result_df["Upper_Shadow"] = (
            result_df["High"] - np.maximum(result_df["Open"], result_df["Close"])
        ) / result_df["Close"]
        result_df["Lower_Shadow"] = (
            np.minimum(result_df["Open"], result_df["Close"]) - result_df["Low"]
        ) / result_df["Close"]

        # 価格位置（期間内での相対位置）
        period = lookback_periods.get("volatility", 20)
        close_min = result_df["Close"].rolling(window=period, min_periods=1).min()
        close_max = result_df["Close"].rolling(window=period, min_periods=1).max()
        result_df["Price_Position"] = DataValidator.safe_divide(
            result_df["Close"] - close_min, close_max - close_min, default_value=0.5
        )

        # ギャップ（前日終値との差）
        result_df["Gap"] = DataValidator.safe_divide(
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
            result_df = df.copy()

            volatility_period = lookback_periods.get("volatility", 20)

            # リターンを計算（安全な計算）
            result_df["Returns"] = DataValidator.safe_pct_change(result_df["Close"])

            # 実現ボラティリティ（安全な計算）
            result_df["Realized_Volatility_20"] = DataValidator.safe_rolling_std(
                result_df["Returns"], window=volatility_period
            ) * np.sqrt(24)

            # ボラティリティスパイク（安全な計算）
            vol_ma = DataValidator.safe_rolling_mean(
                result_df["Realized_Volatility_20"], window=volatility_period
            )
            result_df["Volatility_Spike"] = DataValidator.safe_divide(
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
            result_df["ATR_20"] = DataValidator.safe_rolling_mean(
                result_df["TR"], window=volatility_period
            )

            # 正規化ATR（安全な計算）
            result_df["ATR_Normalized"] = DataValidator.safe_divide(
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
            vol_change = DataValidator.safe_pct_change(
                result_df["Realized_Volatility_20"]
            )
            # 異常に大きな値をクリップ（±500%に制限）
            result_df["Vol_Change"] = np.clip(vol_change, -5.0, 5.0)

            logger.debug("ボラティリティ特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"ボラティリティ特徴量計算エラー: {e}")
            return df

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
            result_df = df.copy()

            volume_period = lookback_periods.get("volume", 20)

            # 出来高移動平均（安全な計算）
            volume_ma = DataValidator.safe_rolling_mean(
                result_df["Volume"], window=volume_period
            )
            # 異常に大きな値をクリップ（最大値を制限）
            volume_max = (
                result_df["Volume"].quantile(0.99) * 10
            )  # 99%分位点の10倍を上限とする
            result_df[f"Volume_MA_{volume_period}"] = np.clip(volume_ma, 0, volume_max)

            # 出来高比率
            result_df["Volume_Ratio"] = DataValidator.safe_divide(
                result_df["Volume"],
                result_df[f"Volume_MA_{volume_period}"],
                default_value=1.0,
            )

            # 価格・出来高トレンド（安全な計算）
            price_change = DataValidator.safe_pct_change(result_df["Close"])
            volume_change = DataValidator.safe_pct_change(result_df["Volume"])
            result_df["Price_Volume_Trend"] = DataValidator.safe_multiply(
                price_change, volume_change
            )

            # 出来高加重平均価格（VWAP）
            typical_price = (
                result_df["High"] + result_df["Low"] + result_df["Close"]
            ) / 3
            result_df["VWAP"] = DataValidator.safe_divide(
                (typical_price * result_df["Volume"])
                .rolling(window=volume_period)
                .sum(),
                result_df["Volume"].rolling(window=volume_period).sum(),
                default_value=np.nan,
            )
            # NaNになったVWAP値をtypical_priceで埋める
            result_df["VWAP"] = result_df["VWAP"].fillna(typical_price)

            # VWAPからの乖離
            result_df["VWAP_Deviation"] = DataValidator.safe_divide(
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
            result_df["Volume_Trend"] = DataValidator.safe_divide(
                result_df["Volume"].rolling(window=5).mean(),
                result_df["Volume"].rolling(window=volume_period).mean(),
                default_value=1.0,
            )

            logger.debug("出来高特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"出来高特徴量計算エラー: {e}")
            return df

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
