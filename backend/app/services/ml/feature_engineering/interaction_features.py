"""
相互作用特徴量計算クラス

既存の特徴量同士を組み合わせて、より複雑な関係性を捉える
相互作用特徴量を計算します。
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from ....utils.error_handler import safe_ml_operation

logger = logging.getLogger(__name__)


class InteractionFeatureCalculator:
    """
    相互作用特徴量計算クラス

    既存の特徴量同士を組み合わせて相互作用特徴量を計算します。
    """

    def __init__(self):
        """初期化"""

    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相互作用特徴量を計算

        Args:
            df: 既存の特徴量を含むDataFrame

        Returns:
            相互作用特徴量が追加されたDataFrame
        """
        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        # 必要な特徴量が存在するかチェック
        if not self._check_required_features(df):
            logger.warning("相互作用特徴量の計算に必要な特徴量が不足しています")
            # 不足している場合は元のDataFrameをそのまま返す
            return df

        try:
            return self._calculate_interaction_features_internal(df)
        except Exception as e:
            logger.error(f"相互作用特徴量計算でエラー: {e}")
            # エラーが発生した場合は元のDataFrameを返す
            return df

    @safe_ml_operation(
        default_return=None, context="相互作用特徴量計算でエラーが発生しました"
    )
    def _calculate_interaction_features_internal(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        内部的な相互作用特徴量計算メソッド
        """
        result_df = df.copy()

        # ボラティリティ × モメンタム相互作用
        result_df = self._calculate_volatility_momentum_interactions(result_df)

        # OI × 価格変動相互作用
        # Removed: OI_Price_Divergence (OI_Change_Rate削除のため)
        # result_df = self._calculate_oi_price_interactions(result_df)

        return result_df

    def _check_required_features(self, df: pd.DataFrame) -> bool:
        """必要な特徴量が存在するかチェック"""
        # 基本的な特徴量のリスト（必須ではない特徴量は除外）
        # Removed: Price_Momentum_14（重複特徴量削除: 2025-01-09）
        basic_required_features = [
            "Price_Change_5",
            "RSI",
        ]

        # オプション特徴量（存在しなくても処理を続行）
        optional_features = [
            "OI_Trend",
        ]

        # ATRは複数の名前で存在する可能性があるため、別途チェック
        atr_variants = ["ATR", "ATR_20", "ATR_14"]
        has_atr = any(variant in df.columns for variant in atr_variants)

        # 基本的な特徴量の不足をチェック
        missing_basic_features = [
            feature for feature in basic_required_features if feature not in df.columns
        ]

        # オプション特徴量の不足を警告として記録
        missing_optional_features = [
            feature for feature in optional_features if feature not in df.columns
        ]

        if missing_basic_features:
            logger.warning(f"不足している基本特徴量: {missing_basic_features}")
            return False

        if missing_optional_features:
            logger.warning(f"不足している特徴量: {missing_optional_features}")

        if not has_atr:
            logger.warning(
                f"ATR系の特徴量が見つかりません。利用可能なATR: {[col for col in df.columns if 'ATR' in col]}"
            )

        # 基本特徴量があれば処理を続行
        return True

    def _calculate_volatility_momentum_interactions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """ボラティリティ × モメンタム相互作用を計算"""
        result_df = df.copy()

        # ATR × Price_Momentum_14（ATRの名前を動的に検出）
        atr_column = None
        for atr_variant in ["ATR", "ATR_20", "ATR_14"]:
            if atr_variant in df.columns:
                atr_column = atr_variant
                break

        # Removed: ATR × Price_Momentum_14（重複特徴量削除: 2025-01-09）
        # 理由: Price_Momentum_14が削除されたため

        # Removed: Volatility_Spike × Price_Change_5（重複特徴量削除: 2025-01-09）
        # 理由: Volatility_Spikeが削除されたため

        # ATR × Momentum（代替特徴量を使用）
        if atr_column and "Momentum" in df.columns:
            try:
                # データ型を確認して安全に変換
                atr_values = self._safe_numeric_conversion(pd.Series(df[atr_column]))
                momentum_values = self._safe_numeric_conversion(
                    pd.Series(df["Momentum"])
                )

                if atr_values is not None and momentum_values is not None:
                    interaction = atr_values * momentum_values
                    # 無限大値や極端に大きな値をクリップ
                    interaction = np.clip(interaction, -1e6, 1e6)
                    result_df["Volatility_Momentum_Interaction"] = interaction
            except Exception as e:
                logger.warning(f"Volatility_Momentum_Interaction計算エラー: {e}")

        return result_df

    def _calculate_volume_trend_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高 × トレンド相互作用を計算 (削除済み特徴量依存のためスキップ)"""
        # Volume_Ratio, Trend_Strength, Volume_Spike, Breakout_Strength は削除されました
        return df

    def _calculate_fr_rsi_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """FR × RSI相互作用を計算 (削除済み特徴量依存のためスキップ)"""
        # FR_Normalized, FR_Extreme_High, FR_Extreme_Low は削除されました
        return df

    def _calculate_oi_price_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """OI × 価格変動相互作用を計算"""
        result_df = df.copy()

        # Removed: OI_Change_Rate × Price_Change_5 (OI_Change_Rate削除のため)

        # OI_Trend × Momentum
        if "OI_Trend" in df.columns and "Momentum" in df.columns:
            try:
                oi_trend = self._safe_numeric_conversion(pd.Series(df["OI_Trend"]))
                momentum = self._safe_numeric_conversion(pd.Series(df["Momentum"]))

                if oi_trend is not None and momentum is not None:
                    interaction = oi_trend * momentum
                    # 無限大値や極端に大きな値をクリップ
                    interaction = np.clip(interaction, -1e6, 1e6)
                    # NaN値を0で補完
                    result_df["OI_Momentum_Alignment"] = np.nan_to_num(
                        interaction, nan=0.0
                    )
            except Exception as e:
                logger.warning(f"OI_Momentum_Alignment計算エラー: {e}")

        return result_df

    def _safe_numeric_conversion(self, series: pd.Series) -> pd.Series | None:
        """
        安全な数値変換を実行

        Args:
            series: 変換するSeries

        Returns:
            変換されたSeries、またはエラーの場合はNone
        """
        try:
            # データ型をチェック
            if series.dtype == "object":
                # object型の場合は数値変換を試行
                series = pd.Series(pd.to_numeric(series, errors="coerce"))

            # 無限大値をNaNに置換
            series = series.replace([np.inf, -np.inf], np.nan)

            # NaN値を0で補完
            series = series.fillna(0.0)

            return series

        except Exception as e:
            logger.warning(f"数値変換エラー: {e}")
            return None

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        return [
            "Volatility_Momentum_Interaction",
            "OI_Momentum_Alignment",
        ]
