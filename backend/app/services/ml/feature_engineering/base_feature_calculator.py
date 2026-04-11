"""
特徴量計算の抽象基底クラス

特徴量計算クラス間の共通処理を集約し、コードの重複を削減します。
共通の初期化、検証、エラーハンドリング、計算パターンを提供します。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from app.types import SerializableValue
from app.utils.data_processing.data_processor import _replace_inf_with_nan

logger = logging.getLogger(__name__)

SeriesOrDict = Union[pd.Series, Dict[str, pd.Series]]


def sanitize_numeric_dataframe(
    df: pd.DataFrame,
    fill_value: Optional[float] = 0.0,
    forward_fill: bool = False,
) -> pd.DataFrame:
    """
    数値カラムの inf / NaN を整形する共通処理

    Args:
        df: 対象のDataFrame
        fill_value: 最終的にNaNを埋める値。Noneならそのまま残す
        forward_fill: 直前値で前方補完するか
    """
    result_df = df.copy()

    numeric_positions = [
        idx
        for idx, dtype in enumerate(result_df.dtypes)
        if pd.api.types.is_numeric_dtype(dtype)
    ]

    if not numeric_positions:
        return result_df.fillna(fill_value) if fill_value is not None else result_df

    for position in numeric_positions:
        series = _replace_inf_with_nan(result_df.iloc[:, position])

        if forward_fill:
            series = series.ffill()

        if fill_value is not None:
            series = series.fillna(fill_value)

        result_df.iloc[:, position] = series

    return result_df


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
        """

    def validate_input_data(
        self, df: Optional[pd.DataFrame], required_columns: Optional[list] = None
    ) -> bool:
        """入力データの妥当性を検証"""
        if df is None or df.empty:
            return False

        if required_columns:
            df_cols = {c.lower() for c in df.columns}
            missing = [c for c in required_columns if c.lower() not in df_cols]
            if missing:
                logger.debug(
                    f"Missing columns in validation: {missing}, Present: {df.columns.tolist()}"
                )
                logger.warning(f"Missing columns: {missing}")
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

    @abstractmethod
    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, SerializableValue]
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

    def create_result_dataframe_efficient(
        self, df: pd.DataFrame, new_features: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        高速な結果DataFrame作成（DataFrame断片化回避）

        新しい特徴量を辞書で収集し、pd.concat()で一括追加することで
        DataFrameの断片化を防ぎ、高速処理を実現します。

        Args:
            df: 元のDataFrame
            new_features: 追加する新特徴量の辞書（Seriesの辞書）

        Returns:
            新特徴量が追加されたDataFrame
        """
        if not new_features:
            # 新規特徴量が空的場合は元のDataFrameを返す
            return df.copy()

        # DataFrame断片化を避けるため、辞書で収集 → pd.concat()で一括追加
        result_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return result_df

    def handle_calculation_error(
        self, error: Exception, context: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        計算エラーを処理し、空の結果DataFrameを返す

        Args:
            error: 発生したエラー
            context: エラーが発生したコンテキスト
            df: 元のDataFrame

        Returns:
            元のDataFrameのコピー
        """
        logger.error(f"特徴量計算エラー ({context}): {error}")
        return df.copy()

    def clip_extreme_values(
        self,
        series: pd.Series,
        lower_bound: float = -1e6,
        upper_bound: float = 1e6,
    ) -> pd.Series:
        """
        極端な値をクリッピングする

        Args:
            series: 対象のSeries
            lower_bound: 下限値
            upper_bound: 上限値

        Returns:
            クリッピングされたSeries
        """
        return series.clip(lower=lower_bound, upper=upper_bound)

    def batch_calculate_ratio(
        self,
        numerators: SeriesOrDict,
        denominators: SeriesOrDict,
        fill_value: float = 0.0,
    ) -> SeriesOrDict:
        """
        バッチで比率を計算する（ゼロ除算対応）

        Args:
            numerators: 分子（SeriesまたはSeriesの辞書）
            denominators: 分母（SeriesまたはSeriesの辞書）
            fill_value: ゼロ除算時の置換値

        Returns:
            計算された比率（入力と同じ型）
        """
        # 辞書型入力の場合
        if isinstance(numerators, dict) and isinstance(denominators, dict):
            if not numerators:
                return {}
            result = {}
            for key in numerators:
                if key in denominators:
                    num = numerators[key]
                    den = denominators[key]
                    safe_den = den.replace(0, np.nan)
                    result[key] = (num / safe_den).fillna(fill_value)
            return result

        # Series型入力の場合
        safe_denominators = denominators.replace(0, np.nan)
        result = numerators / safe_denominators
        return result.fillna(fill_value)  # type: ignore[attr-defined]

    def safe_ratio_calculation(
        self,
        numerator: pd.Series,
        denominator: pd.Series,
        fill_value: float = 0.0,
    ) -> pd.Series:
        """
        安全な比率計算の共通エイリアス。

        既存の特徴量計算クラスには `safe_ratio_calculation` を呼ぶ実装があり、
        それを Base クラス側で受けられるようにして後方互換性を保つ。
        """
        return self.batch_calculate_ratio(numerator, denominator, fill_value)
