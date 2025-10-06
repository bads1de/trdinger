"""
価格変化率計算用Transformer

scikit-learnのPipelineで使用するための価格変化率計算用Transformerを提供します。
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class PriceChangeTransformer(BaseEstimator, TransformerMixin):
    """
    価格データから価格変化率を計算するTransformer

    scikit-learnのPipelineで使用するためのTransformer実装。
    """

    def __init__(self, periods: int = 1):
        """
        Args:
            periods: 価格変化率計算の期間
        """
        self.periods = periods

    def fit(self, X, y=None):
        """フィット（何もしない）"""
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        価格データから価格変化率を計算

        Args:
            X: 価格データ（Series or DataFrame）

        Returns:
            価格変化率の2次元配列
        """
        if isinstance(X, pd.DataFrame):
            # DataFrameの場合、最初の列を使用
            price_series = X.iloc[:, 0]
        else:
            price_series = X

        try:
            # 価格変化率を計算（バグ22対応: エラーハンドリング追加）
            price_change = price_series.pct_change(periods=self.periods)
            price_change = price_change.fillna(0)  # NaN処理を追加
            price_change = price_change.dropna()  # 残ったNaNを除去

            # 2次元配列として返す（scikit-learn要件）
            return np.asarray(price_change.values).reshape(-1, 1)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"価格変化率計算エラー: {e}")
            raise ValueError(f"価格変化率計算に失敗: {e}") from e
