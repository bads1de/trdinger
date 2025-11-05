"""
時間的特徴量計算クラス

取引セッション、曜日効果、時間帯、週末効果などの
時間に関連する特徴量を計算します。
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ....utils.error_handler import safe_ml_operation
from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TemporalFeatureCalculator(BaseFeatureCalculator):
    """
    時間的特徴量計算クラス

    タイムスタンプ情報から時間に関連する特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        時間的特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            時間的特徴量が追加されたDataFrame
        """
        # 時間的特徴量を計算
        result_df = self.calculate_temporal_features(df)

        return result_df

    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間的特徴量を計算

        Args:
            df: OHLCV価格データ（DatetimeIndexを持つ）

        Returns:
            時間的特徴量が追加されたDataFrame
        """
        if df is None or df.empty:
            logger.warning("空のデータが提供されました")
            return df

        # インデックスがDatetimeIndexかチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrameのインデックスはDatetimeIndexである必要があります"
            )

        return self._calculate_temporal_features_internal(df)

    @safe_ml_operation(
        default_return=None, context="時間的特徴量計算でエラーが発生しました"
    )
    def _calculate_temporal_features_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        内部的な時間的特徴量計算メソッド
        """

        result_df = df.copy()

        # UTCタイムゾーンに変換（タイムゾーン情報がない場合はUTCとして扱う）
        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(result_df.index, pd.DatetimeIndex):
            # このパスは_calculate_temporal_features_internalが呼ばれる前にチェック済みのため、通常は到達しない
            # ただし、型ヒントのために含める
            result_df.index = pd.to_datetime(result_df.index)

        if result_df.index.tz is None:
            result_df.index = result_df.index.tz_localize("UTC")
        else:
            result_df.index = result_df.index.tz_convert("UTC")

        # 新しい特徴量を辞書で収集（DataFrame断片化対策）
        new_features = {}

        # 基本的な時間特徴量
        new_features.update(self._calculate_basic_time_features_dict(result_df))

        # 取引セッション特徴量
        new_features.update(self._calculate_trading_session_features_dict(result_df))

        # 周期的エンコーディング
        new_features.update(self._calculate_cyclical_features_dict(result_df))

        # セッション重複時間
        new_features.update(self._calculate_session_overlap_features_dict(result_df))

        # 一括で結合
        result_df = pd.concat(
            [result_df, pd.DataFrame(new_features, index=result_df.index)], axis=1
        )

        return result_df

    def _calculate_basic_time_features_dict(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """基本的な時間特徴量を計算（辞書版）"""
        new_features = {}

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        hour_series = df.index.to_series().dt.hour
        dayofweek_series = df.index.to_series().dt.dayofweek

        # 時間（0-23）
        new_features["Hour_of_Day"] = hour_series

        # 曜日（0=月曜日, 6=日曜日）
        new_features["Day_of_Week"] = dayofweek_series

        # 週末フラグ（土日）
        new_features["Is_Weekend"] = dayofweek_series.isin([5, 6])

        # 月曜効果
        new_features["Is_Monday"] = dayofweek_series == 0

        # 金曜効果
        new_features["Is_Friday"] = dayofweek_series == 4

        return new_features

    def _calculate_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な時間特徴量を計算（レガシー互換用）"""
        new_features = self._calculate_basic_time_features_dict(df)
        return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    def _calculate_trading_session_features_dict(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """取引セッション特徴量を計算（辞書版）"""
        new_features = {}

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        hour = df.index.to_series().dt.hour

        # Asia Session (UTC 0:00-9:00)
        new_features["Asia_Session"] = (hour >= 0) & (hour < 9)

        # Europe Session (UTC 7:00-16:00)
        new_features["Europe_Session"] = (hour >= 7) & (hour < 16)

        # US Session (UTC 13:00-22:00)
        new_features["US_Session"] = (hour >= 13) & (hour < 22)

        return new_features

    def _calculate_trading_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """取引セッション特徴量を計算（レガシー互換用）"""
        new_features = self._calculate_trading_session_features_dict(df)
        return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    def _calculate_cyclical_features_dict(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """周期的エンコーディング特徴量を計算（辞書版）"""
        new_features = {}

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        hour = df.index.to_series().dt.hour
        dayofweek = df.index.to_series().dt.dayofweek

        # 時間の周期的エンコーディング（24時間周期）
        hour_angle = 2 * np.pi * hour / 24
        new_features["Hour_Sin"] = np.sin(hour_angle)
        new_features["Hour_Cos"] = np.cos(hour_angle)

        # 曜日の周期的エンコーディング（7日周期）
        day_angle = 2 * np.pi * dayofweek / 7
        new_features["Day_Sin"] = np.sin(day_angle)
        new_features["Day_Cos"] = np.cos(day_angle)

        return new_features

    def _calculate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """周期的エンコーディング特徴量を計算（レガシー互換用）"""
        new_features = self._calculate_cyclical_features_dict(df)
        return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    def _calculate_session_overlap_features_dict(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """セッション重複時間の特徴量を計算（辞書版）"""
        # Removed: 低寄与度特徴量削除（LightGBM+XGBoost統合分析: 2025-01-05）
        # 削除された特徴量: Session_Overlap_Asia_Europe, Session_Overlap_Europe_US
        # 性能への影響: LightGBM -0.43%, XGBoost -0.43%（許容範囲内）
        return {}

    def _calculate_session_overlap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """セッション重複時間の特徴量を計算（レガシー互換用）"""
        new_features = self._calculate_session_overlap_features_dict(df)
        return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    def get_feature_names(self) -> List[str]:
        """生成される特徴量名のリストを取得"""
        return [
            "Hour_of_Day",
            "Day_of_Week",
            "Is_Weekend",
            "Is_Monday",
            "Is_Friday",
            "Asia_Session",
            "Europe_Session",
            "US_Session",
            # Removed: "Session_Overlap_Asia_Europe",
            # "Session_Overlap_Europe_US" (低寄与度: 2025-01-05)
            "Hour_Sin",
            "Hour_Cos",
            "Day_Sin",
            "Day_Cos",
        ]
