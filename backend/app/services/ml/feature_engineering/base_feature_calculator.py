"""
特徴量計算の抽象基底クラス

特徴量計算クラス間の共通処理を集約し、コードの重複を削減します。
共通の初期化、検証、エラーハンドリング、計算パターンを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ....utils.data_validation import DataValidator


logger = logging.getLogger(__name__)


class BaseFeatureCalculator(ABC):
    """
    特徴量計算の抽象基底クラス

    各特徴量計算クラスの共通処理を提供します。
    - 共通の初期化処理
    - データ検証
    - エラーハンドリング
    - 共通の計算パターン
    """

    def __init__(self):
        """
        初期化

        DataValidatorインスタンスを作成し、共通の設定を行います。
        """
        self.validator = DataValidator()

    def validate_input_data(
        self, df: pd.DataFrame, required_columns: Optional[list] = None
    ) -> bool:
        """
        入力データの妥当性を検証

        Args:
            df: 検証するDataFrame
            required_columns: 必須カラムのリスト

        Returns:
            データが有効な場合True、無効な場合False
        """
        if df is None:
            logger.warning("入力データがNoneです")
            return False

        if df.empty:
            logger.warning("入力データが空です")
            return False

        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"必須カラムが不足しています: {missing_columns}")
                return False

        return True

    def create_result_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        結果用のDataFrameを作成

        Args:
            df: 元のDataFrame

        Returns:
            コピーされたDataFrame
        """
        return df.copy()

    def safe_rolling_mean_calculation(
        self, series: pd.Series, window: int
    ) -> pd.Series:
        """
        安全な移動平均計算

        Args:
            series: 計算対象のSeries
            window: ウィンドウサイズ

        Returns:
            移動平均のSeries
        """
        return DataValidator.safe_rolling_mean(series, window=window)

    def safe_ratio_calculation(
        self,
        numerator: Union[pd.Series, np.ndarray, float],
        denominator: Union[pd.Series, np.ndarray, float],
        default_value: float = 1.0,
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        安全な比率計算

        Args:
            numerator: 分子
            denominator: 分母
            default_value: 分母が0の場合のデフォルト値

        Returns:
            比率の計算結果
        """
        return DataValidator.safe_divide(
            numerator, denominator, default_value=default_value
        )

    def safe_pct_change_calculation(self, series: pd.Series) -> pd.Series:
        """
        安全な変化率計算

        Args:
            series: 計算対象のSeries

        Returns:
            変化率のSeries
        """
        return DataValidator.safe_pct_change(series)

    def safe_normalize_calculation(
        self, data: pd.Series, window: int, default_value: float = 0.0
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        安全な正規化計算

        Args:
            data: 正規化するデータ
            window: 計算ウィンドウ
            default_value: 標準偏差が0の場合のデフォルト値

        Returns:
            正規化されたデータ
        """
        return DataValidator.safe_normalize(
            data, window=window, default_value=default_value
        )

    def safe_multiply_calculation(
        self,
        series1: Union[pd.Series, np.ndarray, float],
        series2: Union[pd.Series, np.ndarray, float],
        default_value: float = 0.0,
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        安全な乗算計算

        Args:
            series1: 第1オペランド
            series2: 第2オペランド
            default_value: エラー時のデフォルト値

        Returns:
            乗算の計算結果
        """
        return DataValidator.safe_multiply(
            series1, series2, default_value=default_value
        )

    def safe_rolling_std_calculation(self, series: pd.Series, window: int) -> pd.Series:
        """
        安全な移動標準偏差計算

        Args:
            series: 計算対象のSeries
            window: ウィンドウサイズ

        Returns:
            移動標準偏差のSeries
        """
        return DataValidator.safe_rolling_std(series, window=window)

    def handle_calculation_error(
        self, error: Exception, context: str, fallback_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        計算エラーの共通ハンドリング

        Args:
            error: 発生したエラー
            context: エラーのコンテキスト
            fallback_df: フォールバック用のDataFrame

        Returns:
            エラー時に返すDataFrame
        """
        logger.error(f"{context}でエラーが発生しました: {error}")
        return fallback_df

    def clip_extreme_values(
        self, series: pd.Series, lower_bound: float = -5.0, upper_bound: float = 5.0
    ) -> pd.Series:
        """
        極値のクリッピング

        Args:
            series: クリッピング対象のSeries
            lower_bound: 下限値
            upper_bound: 上限値

        Returns:
            クリッピングされたSeries
        """
        return np.clip(series, lower_bound, upper_bound)

    @abstractmethod
    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        特徴量計算の抽象メソッド

        各サブクラスで具体的な特徴量計算ロジックを実装する必要があります。

        Args:
            df: 入力データのDataFrame
            config: 計算設定の辞書

        Returns:
            特徴量が追加されたDataFrame
        """
        pass

    def log_feature_calculation_start(self, feature_type: str) -> None:
        """
        特徴量計算開始のログ出力

        Args:
            feature_type: 特徴量の種類
        """
        logger.debug(f"{feature_type}特徴量計算を開始します")

    def log_feature_calculation_complete(self, feature_type: str) -> None:
        """
        特徴量計算完了のログ出力

        Args:
            feature_type: 特徴量の種類
        """
        logger.debug(f"{feature_type}特徴量計算が完了しました")
