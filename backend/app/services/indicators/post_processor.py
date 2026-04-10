"""
後処理モジュール

インジケーター計算結果の後処理を担当します。
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    後処理クラス

    計算結果の統一とインデックス再編成を行います。
    """

    def post_process(
        self, result: Any, config: Dict[str, Any], df: Optional[pd.DataFrame] = None
    ) -> Union[np.ndarray, tuple]:
        """
        後処理 - 戻り値の統一

        Args:
            result: 計算結果
            config: pandas-ta設定
            df: データフレーム（オプション）

        Returns:
            後処理された結果
        """
        # タプルの場合は最初の要素を使用
        if (
            isinstance(result, tuple)
            and len(result) > 0
            and isinstance(result[0], (pd.DataFrame, pd.Series))
        ):
            result = result[0]

        # NaN処理とインデックス再編成
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if df is not None and len(result) != len(df):
                try:
                    result = result.reindex(df.index)
                except Exception:
                    pass

        # 戻り値変換
        if config["returns"] == "single":
            return self._convert_to_single(result)
        else:
            return self._convert_to_multi(result, config)

    def _convert_to_single(self, result: Any) -> np.ndarray:
        """単一値に変換"""
        if isinstance(result, pd.Series):
            return result.to_numpy()
        elif isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].to_numpy()
        else:
            return np.asarray(result)

    def _convert_to_multi(self, result: Any, config: Dict[str, Any]) -> tuple:
        """複数値に変換"""
        if isinstance(result, pd.DataFrame):
            return self._convert_dataframe_to_multi(result, config)
        elif isinstance(result, pd.Series):
            return (result.to_numpy(),)
        elif isinstance(result, tuple):
            return tuple(np.asarray(arr) for arr in result)
        else:
            return (np.asarray(result),)

    def _convert_dataframe_to_multi(
        self, result: pd.DataFrame, config: Dict[str, Any]
    ) -> tuple:
        """DataFrameを複数値に変換"""
        return_cols = config.get("return_cols") or []
        if return_cols:
            selected_cols = []
            for col in return_cols:
                if col in result.columns:
                    selected_cols.append(result[col].to_numpy())
                else:
                    matching_cols = [
                        c
                        for c in result.columns
                        if col in c or col.lower() in c.lower()
                    ]
                    if matching_cols:
                        selected_cols.append(
                            result[matching_cols[0]].to_numpy()
                        )
                    else:
                        selected_cols.append(np.full(len(result), np.nan))
            return tuple(selected_cols)
        else:
            return tuple(
                result.iloc[:, i].to_numpy() for i in range(result.shape[1])
            )
