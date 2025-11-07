"""
価格特徴量計算クラス

OHLCV価格データから基本的な価格関連特徴量を計算します。
単一責任原則に従い、価格特徴量の計算のみを担当します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.error_handler import safe_ml_operation
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
        if not self.validate_input_data(df, ["open", "high", "low", "close", "volume"]):
            return df

        result_df = self.create_result_dataframe(df)

        # 移動平均比率（pandas-ta使用）
        import pandas_ta as ta

        short_ma = lookback_periods.get("short_ma", 10)
        long_ma = lookback_periods.get("long_ma", 50)

        ma_short = ta.sma(result_df["close"], length=short_ma)
        if ma_short is not None:
            ma_short_series = pd.Series(ma_short, index=result_df.index)
            result_df[f"MA_{short_ma}"] = ma_short_series.fillna(result_df["close"])
        else:
            result_df[f"MA_{short_ma}"] = result_df["close"]
        ma_long = ta.sma(result_df["close"], length=long_ma)
        if ma_long is not None:
            ma_long_series = pd.Series(ma_long, index=result_df.index)
            result_df[f"MA_{long_ma}"] = ma_long_series.fillna(result_df["close"])
        else:
            result_df[f"MA_{long_ma}"] = result_df["close"]

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: Price_MA_Ratio_Short, Price_MA_Ratio_Long
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # 価格モメンタム（pandas-ta使用）
        momentum_period = lookback_periods.get("momentum", 14)
        momentum_result = ta.mom(result_df["close"], length=momentum_period)
        if momentum_result is not None:
            result_df["Price_Momentum_14"] = pd.Series(
                momentum_result, index=result_df.index
            ).fillna(0.0)
        else:
            result_df["Price_Momentum_14"] = 0.0

        # 高値・安値ポジション
        position_ratio = (result_df["close"] - result_df["low"]) / (
            result_df["high"] - result_df["low"]
        )
        position_ratio = np.where(np.isinf(position_ratio), np.nan, position_ratio)
        result_df["High_Low_Position"] = pd.Series(
            position_ratio, index=result_df.index
        ).fillna(0.5)

        # 価格変化率（pandas-ta使用）
        roc1 = ta.roc(result_df["close"], length=1)
        result_df["Price_Change_1"] = (
            pd.Series(roc1, index=result_df.index).fillna(0.0)
            if roc1 is not None
            else 0.0
        )
        roc5 = ta.roc(result_df["close"], length=5)
        result_df["Price_Change_5"] = (
            pd.Series(roc5, index=result_df.index).fillna(0.0)
            if roc5 is not None
            else 0.0
        )
        roc20 = ta.roc(result_df["close"], length=20)
        result_df["Price_Change_20"] = (
            pd.Series(roc20, index=result_df.index).fillna(0.0)
            if roc20 is not None
            else 0.0
        )

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: Price_Range
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # ボディサイズ（実体の大きさ）
        result_df["Body_Size"] = (
            (abs(result_df["close"] - result_df["open"]) / result_df["close"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: Upper_Shadow
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # 下ヒゲのみ保持
        lower_shadow = (
            np.minimum(result_df["open"], result_df["close"]) - result_df["low"]
        ) / result_df["close"]
        lower_shadow = np.where(np.isinf(lower_shadow), np.nan, lower_shadow)
        result_df["Lower_Shadow"] = pd.Series(
            lower_shadow, index=result_df.index
        ).fillna(0.0)

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: Price_Position
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # Removed: Gap特徴量（低寄与度特徴量削除: 2025-01-05）

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
            if not self.validate_input_data(df, ["close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volatility_period = lookback_periods.get("volatility", 20)

            # リターンを計算（pandas-ta ROCP: fractional rate of change）
            import pandas_ta as ta

            roc_result = ta.roc(result_df["close"], length=1)
            result_df["Returns"] = (
                pd.Series(roc_result, index=result_df.index).fillna(0.0) / 100.0
                if roc_result is not None
                else 0.0
            )

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: Realized_Volatility_20, Volatility_Spike_MA
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

            # ボラティリティスパイクのみ計算（中間変数として使用）
            stdev_result = ta.stdev(result_df["Returns"], length=volatility_period)
            realized_vol = (
                pd.Series(stdev_result, index=result_df.index).fillna(0.0) * np.sqrt(24)
                if stdev_result is not None
                else pd.Series(0.0, index=result_df.index)
            )
            vol_ma = realized_vol.rolling(
                window=volatility_period, min_periods=1
            ).mean()
            result_df["Volatility_Spike"] = (
                (realized_vol / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            )

            # ATR（Average True Range）- pandas-ta
            atr_result = ta.atr(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                length=volatility_period,
            )
            result_df["ATR_20"] = (
                pd.Series(atr_result, index=result_df.index).fillna(0.0)
                if atr_result is not None
                else 0.0
            )

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: ATR_Normalized
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

            # 削除: High_Vol_Regime - 理由: 極低重要度（分析日: 2025-01-07）
            # ボラティリティレジーム（realized_volを使用）
            # vol_quantile = realized_vol.rolling(
            #     window=volatility_period * 2, min_periods=1
            # ).quantile(0.8)
            # result_df["High_Vol_Regime"] = (realized_vol > vol_quantile).astype(int)

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: Vol_Change
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

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
            if not self.validate_input_data(df, ["volume", "close", "high", "low"]):
                return df

            result_df = self.create_result_dataframe(df)

            volume_period = lookback_periods.get("volume", 20)

            # pandas-ta のローカルインポート（この関数内でのみ使用）
            import pandas_ta as ta

            # 出来高移動平均（pandas-ta使用）
            volume_ma_result = ta.sma(result_df["volume"], length=volume_period)
            volume_ma = (
                pd.Series(volume_ma_result, index=result_df.index).fillna(
                    result_df["volume"]
                )
                if volume_ma_result is not None
                else result_df["volume"]
            )
            # 異常に大きな値をクリップ（最大値を制限）
            volume_max = (
                result_df["volume"].quantile(0.99) * 10
            )  # 99%分位点の10倍を上限とする
            result_df[f"Volume_MA_{volume_period}"] = np.clip(volume_ma, 0, volume_max)

            # 削除: Volume_Ratio - 理由: 低重要度（分析日: 2025-01-07）
            # 出来高比率
            # result_df["Volume_Ratio"] = (
            #     (result_df["volume"] / result_df[f"Volume_MA_{volume_period}"])
            #     .replace([np.inf, -np.inf], np.nan)
            #     .fillna(1.0)
            # )

            # 価格・出来高トレンド（pandas-ta使用）
            price_change_result = ta.roc(result_df["close"], length=1)
            price_change = (
                pd.Series(price_change_result, index=result_df.index).fillna(0.0)
                if price_change_result is not None
                else 0.0
            )
            volume_change_result = ta.roc(result_df["volume"], length=1)
            volume_change = (
                pd.Series(volume_change_result, index=result_df.index).fillna(0.0)
                if volume_change_result is not None
                else 0.0
            )
            result_df["Price_Volume_Trend"] = price_change * volume_change

            # 出来高加重平均価格（VWAP）（pandas-ta使用）
            result_df["VWAP"] = ta.vwap(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                volume=result_df["volume"],
                length=volume_period,
            ).fillna(result_df["close"])

            # VWAPからの乖離
            result_df["VWAP_Deviation"] = (
                ((result_df["close"] - result_df["VWAP"]) / result_df["VWAP"])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 削除: Volume_Spike - 理由: 低重要度（分析日: 2025-01-07）
            # 出来高スパイク
            # vol_threshold = (
            #     result_df["volume"].rolling(window=volume_period).quantile(0.9)
            # )
            # result_df["Volume_Spike"] = (result_df["volume"] > vol_threshold).astype(
            #     int
            # )

            # 出来高トレンド
            volume_trend = (
                result_df["volume"].rolling(window=5).mean()
                / result_df["volume"].rolling(window=volume_period).mean()
            )
            volume_trend = np.where(np.isinf(volume_trend), np.nan, volume_trend)
            result_df["Volume_Trend"] = pd.Series(
                volume_trend, index=result_df.index
            ).fillna(1.0)

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
            # Removed: "Price_MA_Ratio_Short", "Price_MA_Ratio_Long"
            # (低寄与度特徴量削除: 2025-01-05)
            "Price_Momentum_14",
            "High_Low_Position",
            "Price_Change_1",
            "Price_Change_5",
            "Price_Change_20",
            # Removed: "Price_Range" (低寄与度特徴量削除: 2025-01-05)
            "Body_Size",
            # Removed: "Upper_Shadow" (低寄与度特徴量削除: 2025-01-05)
            "Lower_Shadow",
            # Removed: "Price_Position" (低寄与度特徴量削除: 2025-01-05)
            # Removed: "Gap" (低寄与度特徴量削除: 2025-01-05)
            # ボラティリティ特徴量
            # Removed: "Realized_Volatility_20", "Volatility_Spike_MA"
            # (低寄与度特徴量削除: 2025-01-05)
            "Volatility_Spike",
            "ATR_20",
            # Removed: "ATR_Normalized" (低寄与度特徴量削除: 2025-01-05)
            # Removed: "High_Vol_Regime" (極低重要度: 2025-01-07)
            # Removed: "Vol_Change" (低寄与度特徴量削除: 2025-01-05)
            # 出来高特徴量
            # Removed: "Volume_Ratio" (低重要度: 2025-01-07)
            "Price_Volume_Trend",
            "VWAP_Deviation",
            # Removed: "Volume_Spike" (低重要度: 2025-01-07)
            "Volume_Trend",
        ]
