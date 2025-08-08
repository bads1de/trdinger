"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....utils.data_validation import DataValidator
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class MarketDataFeatureCalculator(BaseFeatureCalculator):
    """
    市場データ特徴量計算クラス

    ファンディングレート、建玉残高データから特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        市場データ特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periods、funding_rate_data、open_interest_dataを含む）

        Returns:
            市場データ特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})
        funding_rate_data = config.get("funding_rate_data")
        open_interest_data = config.get("open_interest_data")

        result_df = df

        if funding_rate_data is not None:
            result_df = self.calculate_funding_rate_features(
                result_df, funding_rate_data, lookback_periods
            )

        if open_interest_data is not None:
            result_df = self.calculate_open_interest_features(
                result_df, open_interest_data, lookback_periods
            )

        if funding_rate_data is not None and open_interest_data is not None:
            result_df = self.calculate_combined_features(
                result_df, funding_rate_data, open_interest_data, lookback_periods
            )

        return result_df

    def calculate_funding_rate_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        ファンディングレート特徴量を計算

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            lookback_periods: 計算期間設定

        Returns:
            ファンディングレート特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # ファンディングレートデータをOHLCVデータにマージ
            if "timestamp" in funding_rate_data.columns:
                funding_rate_data = funding_rate_data.set_index("timestamp")

            # インデックスを合わせてマージ
            merged_df = result_df.join(funding_rate_data, how="left", rsuffix="_fr")

            # ファンディングレートカラムを特定
            fr_column = None
            for col in ["funding_rate", "fundingRate", "rate"]:
                if col in merged_df.columns:
                    fr_column = col
                    break

            if fr_column is None:
                logger.warning("ファンディングレートカラムが見つかりません")
                return result_df

            # 欠損値を前方補完
            merged_df[fr_column] = merged_df[fr_column].ffill()

            # ファンディングレート移動平均（安全な計算）
            result_df["FR_MA_24"] = DataValidator.safe_rolling_mean(
                merged_df[fr_column], window=24
            )
            result_df["FR_MA_168"] = DataValidator.safe_rolling_mean(
                merged_df[fr_column], window=168
            )  # 1週間

            # ファンディングレート変化（安全な計算）
            result_df["FR_Change"] = merged_df[fr_column].diff().fillna(0.0)
            result_df["FR_Change_Rate"] = DataValidator.safe_pct_change(
                merged_df[fr_column]
            )

            # ファンディングレートと価格の乖離（安全な計算）
            price_change = DataValidator.safe_pct_change(result_df["Close"])
            result_df["Price_FR_Divergence"] = price_change - merged_df[fr_column]

            # ファンディングレートの極値
            fr_quantile_high = merged_df[fr_column].rolling(window=168).quantile(0.9)
            fr_quantile_low = merged_df[fr_column].rolling(window=168).quantile(0.1)

            result_df["FR_Extreme_High"] = (
                merged_df[fr_column] > fr_quantile_high
            ).astype(int)
            result_df["FR_Extreme_Low"] = (
                merged_df[fr_column] < fr_quantile_low
            ).astype(int)

            # ファンディングレートの正規化
            result_df["FR_Normalized"] = DataValidator.safe_normalize(
                merged_df[fr_column], window=168, default_value=0.0
            )

            # ファンディングレートトレンド
            result_df["FR_Trend"] = result_df["FR_MA_24"] - result_df["FR_MA_168"]

            # ファンディングレートボラティリティ（安全な計算）
            result_df["FR_Volatility"] = DataValidator.safe_rolling_std(
                merged_df[fr_column], window=24
            )

            logger.debug("ファンディングレート特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"ファンディングレート特徴量計算エラー: {e}")
            return df

    def calculate_open_interest_features(
        self,
        df: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        建玉残高特徴量を計算

        Args:
            df: OHLCV価格データ
            open_interest_data: 建玉残高データ
            lookback_periods: 計算期間設定

        Returns:
            建玉残高特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # 建玉残高データをOHLCVデータにマージ
            if "timestamp" in open_interest_data.columns:
                open_interest_data = open_interest_data.set_index("timestamp")

            # インデックスを合わせてマージ
            merged_df = result_df.join(open_interest_data, how="left", rsuffix="_oi")

            # 建玉残高カラムを特定
            oi_column = None
            for col in ["open_interest", "openInterest", "oi"]:
                if col in merged_df.columns:
                    oi_column = col
                    break

            if oi_column is None:
                logger.warning("建玉残高カラムが見つかりません")
                return result_df

            # 欠損値を前方補完
            merged_df[oi_column] = merged_df[oi_column].ffill()

            # 建玉残高変化率（安全な計算）
            result_df["OI_Change_Rate"] = DataValidator.safe_pct_change(
                merged_df[oi_column]
            )
            result_df["OI_Change_Rate_24h"] = DataValidator.safe_pct_change(
                merged_df[oi_column], periods=24
            )

            # 建玉残高急増
            oi_threshold = merged_df[oi_column].rolling(window=168).quantile(0.9)
            result_df["OI_Surge"] = (merged_df[oi_column] > oi_threshold).astype(int)

            # ボラティリティ調整建玉残高（安全な計算）
            price_change = DataValidator.safe_pct_change(result_df["Close"])
            volatility = DataValidator.safe_rolling_std(price_change, window=24)
            result_df["Volatility_Adjusted_OI"] = DataValidator.safe_divide(
                merged_df[oi_column],
                volatility,
                default_value=np.nan,  # デフォルト値をnp.nanに設定
            )
            # NaNになった値を元のOIデータで埋める
            result_df["Volatility_Adjusted_OI"] = result_df[
                "Volatility_Adjusted_OI"
            ].fillna(merged_df[oi_column])

            # 建玉残高移動平均（安全な計算）
            result_df["OI_MA_24"] = DataValidator.safe_rolling_mean(
                merged_df[oi_column], window=24
            )
            result_df["OI_MA_168"] = DataValidator.safe_rolling_mean(
                merged_df[oi_column], window=168
            )

            # 建玉残高トレンド（安全な計算）
            result_df["OI_Trend"] = (
                DataValidator.safe_divide(
                    result_df["OI_MA_24"], result_df["OI_MA_168"], default_value=1.0
                )
                - 1
            )

            # 建玉残高と価格の関係（安全な計算）
            price_change = DataValidator.safe_pct_change(result_df["Close"])
            oi_change = result_df["OI_Change_Rate"]

            # 建玉残高と価格の相関（簡易実装）
            # 相関計算は複雑なため、共分散ベースの簡易実装を使用
            result_df["OI_Price_Correlation"] = DataValidator.safe_multiply(
                price_change, oi_change, default_value=0.0
            )

            # 建玉残高の正規化
            result_df["OI_Normalized"] = DataValidator.safe_normalize(
                merged_df[oi_column], window=168, default_value=0.0
            )

            logger.debug("建玉残高特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"建玉残高特徴量計算エラー: {e}")
            return df

    def calculate_composite_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        複合特徴量を計算（FR + OI）

        Args:
            df: OHLCV価格データ
            funding_rate_data: ファンディングレートデータ
            open_interest_data: 建玉残高データ
            lookback_periods: 計算期間設定

        Returns:
            複合特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # 両方のデータをマージ
            if "timestamp" in funding_rate_data.columns:
                funding_rate_data = funding_rate_data.set_index("timestamp")
            if "timestamp" in open_interest_data.columns:
                open_interest_data = open_interest_data.set_index("timestamp")

            merged_df = result_df.join(funding_rate_data, how="left", rsuffix="_fr")
            merged_df = merged_df.join(open_interest_data, how="left", rsuffix="_oi")

            # カラムを特定
            fr_column = None
            for col in ["funding_rate", "fundingRate", "rate"]:
                if col in merged_df.columns:
                    fr_column = col
                    break

            oi_column = None
            for col in ["open_interest", "openInterest", "oi"]:
                if col in merged_df.columns:
                    oi_column = col
                    break

            if fr_column is None or oi_column is None:
                logger.warning(
                    "ファンディングレートまたは建玉残高カラムが見つかりません"
                )
                return result_df

            # 欠損値を前方補完
            merged_df[fr_column] = merged_df[fr_column].ffill()
            merged_df[oi_column] = merged_df[oi_column].ffill()

            # FR/OI比率
            result_df["FR_OI_Ratio"] = DataValidator.safe_divide(
                merged_df[fr_column], merged_df[oi_column], default_value=0.0
            )

            # 市場ヒートインデックス（FR * OI変化率）（安全な計算）
            oi_change = DataValidator.safe_pct_change(merged_df[oi_column])
            result_df["Market_Heat_Index"] = DataValidator.safe_multiply(
                merged_df[fr_column], oi_change, default_value=0.0
            )

            # 市場ストレス指標
            fr_normalized = DataValidator.safe_normalize(
                merged_df[fr_column], window=168, default_value=0.0
            )
            oi_normalized = DataValidator.safe_normalize(
                merged_df[oi_column], window=168, default_value=0.0
            )
            result_df["Market_Stress"] = np.sqrt(fr_normalized**2 + oi_normalized**2)

            # 市場バランス指標
            result_df["Market_Balance"] = (merged_df[fr_column] > 0).astype(int) * (
                oi_change > 0
            ).astype(int)

            logger.debug("複合特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"複合特徴量計算エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        """
        生成される市場データ特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # ファンディングレート特徴量
            "FR_MA_24",
            "FR_MA_168",
            "FR_Change",
            "FR_Change_Rate",
            "Price_FR_Divergence",
            "FR_Extreme_High",
            "FR_Extreme_Low",
            "FR_Normalized",
            "FR_Trend",
            "FR_Volatility",
            # 建玉残高特徴量
            "OI_Change_Rate",
            "OI_Change_Rate_24h",
            "OI_Surge",
            "Volatility_Adjusted_OI",
            "OI_Trend",
            "OI_Price_Correlation",
            "OI_Normalized",
            # 複合特徴量
            "FR_OI_Ratio",
            "Market_Heat_Index",
            "Market_Stress",
            "Market_Balance",
        ]
