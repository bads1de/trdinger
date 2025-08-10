"""
ML指標アダプタ（ダミー実装）

将来の本実装では、DBに保存された推論結果やオンライン推論を読み込み、
0-1スケールの確率系列を返す。
現状はテスト合格用の最小実装（入力長に合わせたNaN配列を返す or 0.5埋め）
"""
from typing import Union
import numpy as np
import pandas as pd

from ..utils import (
    PandasTAError,
    handle_pandas_ta_errors,
    ensure_series_minimal_conversion,
    validate_series_data,
)


class MLIndicators:
    """ML指標（ダミー）"""

    @staticmethod
    @handle_pandas_ta_errors
    def up_prob(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        series = ensure_series_minimal_conversion(data)
        # 最小チェック（長さ>0）
        validate_series_data(series, 1)
        # 一旦0.5で埋める（将来は実値に置換）
        return np.full_like(series.to_numpy(), 0.5, dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def down_prob(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, 1)
        return np.full_like(series.to_numpy(), 0.5, dtype=float)

    @staticmethod
    @handle_pandas_ta_errors
    def range_prob(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        series = ensure_series_minimal_conversion(data)
        validate_series_data(series, 1)
        return np.full_like(series.to_numpy(), 0.5, dtype=float)

