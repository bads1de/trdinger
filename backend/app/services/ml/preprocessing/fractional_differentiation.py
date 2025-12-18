import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FractionalDifferentiation:
    """
    分数次差分 (Fractional Differentiation) を計算するクラス。

    Marcos Lopez de Prado著 "Advances in Financial Machine Learning" に基づく実装。
    時系列の定常性を確保しつつ、過去の記憶（メモリ）を最大限保持することを目的とする。
    """

    def __init__(
        self, d: float = 1.0, window_size: int = 0, weight_threshold: float = 1e-4
    ):
        """
        初期化

        Args:
            d (float): 差分の次数。0 < d < 1 の範囲が一般的だが、任意の正の実数。
            window_size (int): 固定ウィンドウサイズ。0の場合は閾値に基づいて動的に決定（未実装）。
                               パフォーマンスと一貫性のために正の整数を指定することを推奨。
            weight_threshold (float): 重みの絶対値がこの値を下回った時点で計算を打ち切る閾値。
                                      window_sizeが指定されている場合は無視される場合がある。
        """
        self.d = d
        self.window_size = window_size
        self.weight_threshold = weight_threshold

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """
        分数次差分の重み係数を計算する。

        w_k = -w_{k-1} * (d - k + 1) / k

        Args:
            d (float): 差分次数
            size (int): 重みの数（ウィンドウサイズ）

        Returns:
            np.ndarray: 重み係数の配列 (w_0, w_1, ..., w_{size-1})
        """
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w)

    def transform(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """データに対して分数次差分を適用する"""
        if isinstance(data, pd.Series):
            return self._transform_series(data)
        if isinstance(data, pd.DataFrame):
            return data.apply(self._transform_series)
        raise TypeError("Input must be a Series or DataFrame")

    def _transform_series(self, series: pd.Series) -> pd.Series:
        """単一のSeriesに対して分数次差分を適用する（固定ウィンドウ方式）"""
        window_size = int(self.window_size)
        if len(series) < window_size:
            return pd.Series(np.nan, index=series.index)

        weights = self._get_weights(self.d, window_size)[::-1]
        try:
            return series.rolling(window=window_size).apply(
                lambda x: np.dot(x, weights), raw=True
            )
        except Exception as e:
            logger.error(f"Error in FractionalDifferentiation: {e}")
            raise



