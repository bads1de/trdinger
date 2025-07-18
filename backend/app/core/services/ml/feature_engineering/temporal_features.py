"""
時間的特徴量計算クラス

取引セッション、曜日効果、時間帯、週末効果などの
時間に関連する特徴量を計算します。
"""

import logging
import pandas as pd
import numpy as np
from typing import List

from ....utils.unified_error_handler import safe_ml_operation


logger = logging.getLogger(__name__)


class TemporalFeatureCalculator:
    """
    時間的特徴量計算クラス

    タイムスタンプ情報から時間に関連する特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        pass

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

        # 基本的な時間特徴量
        result_df = self._calculate_basic_time_features(result_df)

        # 取引セッション特徴量
        result_df = self._calculate_trading_session_features(result_df)

        # 周期的エンコーディング
        result_df = self._calculate_cyclical_features(result_df)

        # セッション重複時間
        result_df = self._calculate_session_overlap_features(result_df)

        logger.debug("時間的特徴量計算完了")
        return result_df

    def _calculate_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な時間特徴量を計算"""
        result_df = df.copy()

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)

        # 時間（0-23）
        result_df["Hour_of_Day"] = result_df.index.hour

        # 曜日（0=月曜日, 6=日曜日）
        result_df["Day_of_Week"] = result_df.index.dayofweek

        # 週末フラグ（土日）
        result_df["Is_Weekend"] = result_df.index.dayofweek.isin([5, 6])

        # 月曜効果
        result_df["Is_Monday"] = result_df.index.dayofweek == 0

        # 金曜効果
        result_df["Is_Friday"] = result_df.index.dayofweek == 4

        return result_df

    def _calculate_trading_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """取引セッション特徴量を計算"""
        result_df = df.copy()

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)

        hour = result_df.index.hour

        # Asia Session (UTC 0:00-9:00)
        result_df["Asia_Session"] = (hour >= 0) & (hour < 9)

        # Europe Session (UTC 7:00-16:00)
        result_df["Europe_Session"] = (hour >= 7) & (hour < 16)

        # US Session (UTC 13:00-22:00)
        result_df["US_Session"] = (hour >= 13) & (hour < 22)

        return result_df

    def _calculate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """周期的エンコーディング特徴量を計算"""
        result_df = df.copy()

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)

        # 時間の周期的エンコーディング（24時間周期）
        hour_angle = 2 * np.pi * result_df.index.hour / 24
        result_df["Hour_Sin"] = np.sin(hour_angle)
        result_df["Hour_Cos"] = np.cos(hour_angle)

        # 曜日の周期的エンコーディング（7日周期）
        day_angle = 2 * np.pi * result_df.index.dayofweek / 7
        result_df["Day_Sin"] = np.sin(day_angle)
        result_df["Day_Cos"] = np.cos(day_angle)

        return result_df

    def _calculate_session_overlap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """セッション重複時間の特徴量を計算"""
        result_df = df.copy()

        # Pylance が df.index の型を正しく認識できるように明示的にキャスト
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)

        hour = result_df.index.hour

        # Asia-Europe overlap (UTC 7:00-9:00)
        result_df["Session_Overlap_Asia_Europe"] = (hour >= 7) & (hour < 9)

        # Europe-US overlap (UTC 13:00-16:00)
        result_df["Session_Overlap_Europe_US"] = (hour >= 13) & (hour < 16)

        return result_df

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
            "Session_Overlap_Asia_Europe",
            "Session_Overlap_Europe_US",
            "Hour_Sin",
            "Hour_Cos",
            "Day_Sin",
            "Day_Cos",
        ]
