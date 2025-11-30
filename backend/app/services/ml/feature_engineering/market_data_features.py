"""
市場データ特徴量計算クラス

ファンディングレート（FR）、建玉残高（OI）データから
市場の歪みや偏りを捉える特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

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
            result_df = self.calculate_composite_features(
                result_df, funding_rate_data, open_interest_data, lookback_periods
            )
            result_df = self.calculate_market_dynamics_features(
                result_df, funding_rate_data, open_interest_data, lookback_periods
            )

        return result_df

    def calculate_market_dynamics_features(
        self,
        df: pd.DataFrame,
        funding_rate_data: pd.DataFrame,
        open_interest_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        市場ダイナミクス特徴量を計算 (OI/FR/Priceの高度な相互作用)
        AdvancedFeatureEngineerから移行
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
                return result_df

            # 欠損値を前方補完
            merged_df[fr_column] = merged_df[fr_column].ffill()
            merged_df[oi_column] = merged_df[oi_column].ffill()

            # 1. OI Weighted FR
            result_df["OI_Weighted_FR"] = merged_df[fr_column] * merged_df[oi_column]

            # 2. Cumulative OI Weighted FR
            result_df["Cumulative_OI_Weighted_FR_24h"] = (
                result_df["OI_Weighted_FR"].rolling(24).sum()
            ).fillna(0.0)

            # 3. OI/Price Divergence
            price_pct = result_df["close"].pct_change(fill_method=None)
            oi_pct = merged_df[oi_column].pct_change(fill_method=None)
            epsilon = 1e-6
            result_df["OI_Price_Divergence"] = (
                oi_pct / (price_pct.abs() + epsilon)
            ).fillna(0.0)

            # 4. FR/Price Divergence
            result_df["FR_Price_Correlation_24h"] = (
                merged_df[fr_column].rolling(24).corr(result_df["close"])
            ).fillna(0.0)

            return result_df

        except Exception as e:
            logger.error(f"市場ダイナミクス特徴量計算エラー: {e}")
            return df

        return result_df

    def _process_market_data(
        self,
        df: pd.DataFrame,
        data: pd.DataFrame,
        column_candidates: list[str],
        suffix: str,
    ) -> tuple[pd.DataFrame, str | None]:
        """
        市場データをマージし、カラムを特定して前処理を行う共通メソッド

        Args:
            df: ベースとなるDataFrame
            data: マージする市場データ
            column_candidates: カラム名の候補リスト
            suffix: マージ時のサフィックス

        Returns:
            (マージ済みDataFrame, 特定されたカラム名)
        """
        result_df = df.copy()

        # データをOHLCVデータにマージ
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        # インデックスを合わせてマージ
        merged_df = result_df.join(data, how="left", rsuffix=suffix)

        # カラムを特定
        target_column = None
        for col in column_candidates:
            if col in merged_df.columns:
                target_column = col
                break

        if target_column is not None:
            # 欠損値を前方補完
            merged_df[target_column] = merged_df[target_column].ffill()

        return merged_df, target_column

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
            # 共通処理でデータマージとカラム特定
            merged_df, fr_column = self._process_market_data(
                df, funding_rate_data, ["funding_rate", "fundingRate", "rate"], "_fr"
            )

            if fr_column is None:
                logger.warning("ファンディングレートカラムが見つかりません")
                return df

            # 削除: funding_rate (生データ) - 理由: 加工済み特徴量で代替（分析日: 2025-01-07）
            # 生のファンディングレートデータは使用せず、加工済み特徴量（複合特徴量）のみを使用
            # 実際に削除処理を実行
            result_df = merged_df
            if fr_column in result_df.columns:
                result_df = result_df.drop(columns=[fr_column])
                logger.info(f"Removed raw funding_rate column: {fr_column}")

            # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
            # 削除された特徴量: FR_MA_24, FR_MA_168, FR_Change, FR_Change_Rate,
            # Price_FR_Divergence, FR_Extreme_High, FR_Extreme_Low, FR_Normalized,
            # FR_Trend, FR_Volatility
            # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）
            # 注: これらの特徴量は低寄与度のため削除されましたが、
            # 他の関数で使用される可能性があるため、必要に応じて中間計算は保持

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
            # 共通処理でデータマージとカラム特定
            merged_df, oi_column = self._process_market_data(
                df, open_interest_data, ["open_interest", "openInterest", "oi"], "_oi"
            )

            if oi_column is None:
                logger.warning("建玉残高カラムが見つかりません")
                return df

            result_df = merged_df

            # 削除: open_interest (生データ) - 理由: 加工済み特徴量で代替（分析日: 2025-01-07）
            # 生の建玉残高データは使用せず、加工済み特徴量（変化率、正規化値等）のみを使用
            # 実際に削除処理を実行
            if oi_column in result_df.columns:
                # 計算用にSeriesを保持してから削除
                oi_series = result_df[oi_column]
                result_df = result_df.drop(columns=[oi_column])
                logger.info(f"Removed raw open_interest column: {oi_column}")
            else:
                # 万が一カラムがない場合（通常ありえないが）
                logger.warning(f"OI column {oi_column} not found in merged dataframe")
                return df

            # 共通ロジックを使用してOI特徴量を計算
            result_df = self._calculate_oi_derived_features(result_df, oi_series)

            return result_df

        except Exception as e:
            logger.error(f"建玉残高特徴量計算エラー: {e}")
            return df

    def calculate_pseudo_open_interest_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        建玉残高疑似特徴量を生成

        Args:
            df: 価格データ
            lookback_periods: 計算期間設定

        Returns:
            疑似特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # ボリュームベースの疑似建玉残高
            # volumeは必須カラムと想定
            if "volume" not in result_df.columns:
                logger.warning("volumeカラムがないため、疑似OI特徴量を生成できません")
                return result_df

            pseudo_oi = result_df["volume"].rolling(24).mean() * 10
            # 明示的にpandas Seriesであることを保証
            pseudo_oi = pd.Series(pseudo_oi, index=result_df.index)

            # 共通ロジックを使用してOI特徴量を計算
            result_df = self._calculate_oi_derived_features(result_df, pseudo_oi)

            logger.info("建玉残高疑似特徴量を生成しました")
            return result_df

        except Exception as e:
            logger.error(f"建玉残高疑似特徴量生成エラー: {e}")
            return df

    def _calculate_oi_derived_features(
        self, df: pd.DataFrame, oi_series: pd.Series
    ) -> pd.DataFrame:
        """
        建玉残高（または疑似建玉残高）から派生特徴量を計算する共通メソッド

        Args:
            df: 特徴量を追加するDataFrame (価格データを含む)
            oi_series: 建玉残高のSeries

        Returns:
            特徴量が追加されたDataFrame
        """
        result_df = df.copy()

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: OI_Change_Rate
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）
        result_df["OI_Change_Rate_24h"] = (
            oi_series.pct_change(periods=24)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: OI_Surge
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # ボラティリティ調整建玉残高（安全な計算）
        price_change = (
            result_df["close"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        volatility = price_change.rolling(window=24, min_periods=1).std().fillna(0.0)
        result_df["Volatility_Adjusted_OI"] = (oi_series / volatility).replace(
            [np.inf, -np.inf], np.nan
        )
        # NaNになった値を元のOIデータで埋める
        result_df["Volatility_Adjusted_OI"] = result_df[
            "Volatility_Adjusted_OI"
        ].fillna(oi_series)

        # 建玉残高移動平均（中間計算用）
        oi_ma_24 = oi_series.rolling(window=24, min_periods=1).mean()
        oi_ma_168 = oi_series.rolling(window=168, min_periods=1).mean()

        # 建玉残高トレンド（安全な計算）
        result_df["OI_Trend"] = (oi_ma_24 / oi_ma_168).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0) - 1

        # 建玉残高と価格の関係
        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: OI_Price_Correlation (OI_Change_Rateに依存)
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）

        # 建玉残高の正規化
        oi_mean = oi_series.rolling(window=168, min_periods=1).mean()
        oi_std = oi_series.rolling(window=168, min_periods=1).std().replace(0, np.nan)
        result_df["OI_Normalized"] = (
            ((oi_series - oi_mean) / oi_std)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        return result_df

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
            result_df["FR_OI_Ratio"] = (
                (merged_df[fr_column] / merged_df[oi_column])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 市場ヒートインデックス（FR * OI変化率）
            oi_change = (
                merged_df[oi_column]
                .pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            result_df["Market_Heat_Index"] = (
                (merged_df[fr_column] * oi_change)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            # 市場ストレス指標
            fr_mean = merged_df[fr_column].rolling(window=168, min_periods=1).mean()
            fr_std = (
                merged_df[fr_column]
                .rolling(window=168, min_periods=1)
                .std()
                .replace(0, np.nan)
            )
            fr_normalized = (
                ((merged_df[fr_column] - fr_mean) / fr_std)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            oi_mean = merged_df[oi_column].rolling(window=168, min_periods=1).mean()
            oi_std = (
                merged_df[oi_column]
                .rolling(window=168, min_periods=1)
                .std()
                .replace(0, np.nan)
            )
            oi_normalized = (
                ((merged_df[oi_column] - oi_mean) / oi_std)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            result_df["Market_Stress"] = np.sqrt(fr_normalized**2 + oi_normalized**2)

            # 市場バランス指標
            result_df["Market_Balance"] = (merged_df[fr_column] > 0).astype(int) * (
                oi_change > 0
            ).astype(int)

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
            # Removed: "FR_MA_24", "FR_MA_168", "FR_Change", "FR_Change_Rate",
            # "Price_FR_Divergence", "FR_Extreme_High", "FR_Extreme_Low",
            # "FR_Normalized", "FR_Trend", "FR_Volatility"
            # (低寄与度特徴量削除: 2025-01-05)
            # 建玉残高特徴量
            # Removed: "OI_Change_Rate" (低寄与度特徴量削除: 2025-01-05)
            "OI_Change_Rate_24h",
            # Removed: "OI_Surge" (低寄与度特徴量削除: 2025-01-05)
            "Volatility_Adjusted_OI",
            "OI_Trend",
            # Removed: "OI_Price_Correlation" (低寄与度特徴量削除: 2025-01-05)
            "OI_Normalized",
            # 複合特徴量
            "FR_OI_Ratio",
            "Market_Heat_Index",
            "Market_Stress",
            "Market_Stress",
            "Market_Balance",
            # 市場ダイナミクス
            "OI_Weighted_FR",
            "Cumulative_OI_Weighted_FR_24h",
            "OI_Price_Divergence",
            "FR_Price_Correlation_24h",
        ]
