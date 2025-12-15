"""
特徴量計算の抽象基底クラス

特徴量計算クラス間の共通処理を集約し、コードの重複を削減します。
共通の初期化、検証、エラーハンドリング、計算パターンを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

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
        """

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

        # カラム名を小文字に統一して検証
        df_columns = [col.lower() for col in df.columns]
        if required_columns is not None:
            required_columns = [col.lower() for col in required_columns]

        if df.empty:
            logger.warning("入力データが空です")
            return False

        if required_columns:
            missing_columns = [col for col in required_columns if col not in df_columns]
            if missing_columns:
                logger.warning(
                    f"必須カラムが不足しています（小文字化対応）: {missing_columns}"
                )
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
        return series.clip(lower=lower_bound, upper=upper_bound)

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

    def log_feature_calculation_complete(self, feature_type: str) -> None:
        """
        特徴量計算完了のログ出力

        Args:
            feature_type: 特徴量の種類
        """
        pass  # デバッグログを削除

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

    def batch_calculate_ratio(
        self,
        numerators: Dict[str, pd.Series],
        denominators: Dict[str, pd.Series],
        fill_value: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        比の一括計算（ゼロ除算対応）

        複数の分子と分母のペアを受け取り、safe_ratio_calculationを使用して
        一括で比を計算します。

        Args:
            numerators: 分子のSeries辞書
            denominators: 分母のSeries辞書
            fill_value: ゼロ除算時の埋め値

        Returns:
            計算結果のSeries辞書
        """
        results = {}

        for name in numerators:
            if name in denominators:
                results[name] = self.safe_ratio_calculation(
                    numerators[name], denominators[name], fill_value=fill_value
                )
            else:
                logger.warning(f"分母'{name}'が見つかりません")

        return results

    def safe_ratio_calculation(
        self,
        numerator: pd.Series | Any,
        denominator: pd.Series | Any,
        fill_value: float = 0.0,
    ) -> pd.Series:
        """
        ゼロ除算を防ぐための安全な比率計算

        Args:
            numerator: 分子
            denominator: 分母
            fill_value: ゼロ除算時の埋め値

        Returns:
            計算結果のSeries
        """
        if (
            denominator is None
            or numerator is None
            or not isinstance(numerator, pd.Series)
            or not isinstance(denominator, pd.Series)
        ):
            length = len(numerator) if isinstance(numerator, pd.Series) else 0
            return pd.Series([fill_value] * length)

        # ゼロ除算を防ぐ
        ratio = numerator / denominator.replace(0, np.nan)
        return ratio.replace([np.inf, -np.inf], np.nan).fillna(fill_value)



