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
    pandas-taの計算結果を統一された形式（numpy配列またはタプル）に変換し、
    入力DataFrameのインデックスに合わせて整列します。
    """

    def post_process(
        self, result: Any, config: Dict[str, Any], df: Optional[pd.DataFrame] = None
    ) -> Union[np.ndarray, tuple]:
        """
        後処理 - 戻り値の統一

        pandas-taの計算結果を統一された形式に変換します。
        タプルの最初の要素を抽出し、インデックスを再編成し、
        単一出力または複数出力に応じて適切な形式に変換します。

        Args:
            result: pandas-taの計算結果（Series、DataFrame、タプル等）
            config: pandas-ta設定（returns、return_cols等を含む）
            df: 対象のDataFrame（インデックス再編成用、オプション）

        Returns:
            Union[np.ndarray, tuple]: 後処理された結果
                                       - 単一出力: numpy配列
                                       - 複数出力: numpy配列のタプル

        処理手順:
            1. タプルの場合は最初の要素を抽出
            2. DataFrame/Seriesの場合はインデックスをdf.indexに再編成
            3. config['returns']に応じて単一または複数形式に変換
        """
        # タプルの場合は最初の要素を使用（pandas_ta の複数結果の最初の要素のみを使う場合）
        # ただし、タプルの全要素が必要な場合はそのまま保持
        if isinstance(result, tuple) and len(result) > 0:
            if isinstance(result[0], (pd.DataFrame, pd.Series)):
                result = result[0]
            # ネイティブ配列のタプルはそのまま保持（後続の_convert_to_multiで処理）

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
        """
        単一値に変換

        計算結果を単一のnumpy配列に変換します。
        Seriesの場合はnumpy配列に、DataFrameの場合は最初の列を抽出します。

        Args:
            result: 変換対象（Series、DataFrame、配列等）

        Returns:
            np.ndarray: 単一のnumpy配列
        """
        if isinstance(result, pd.Series):
            return result.to_numpy()
        elif isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].to_numpy()
        else:
            return np.asarray(result)

    def _convert_to_multi(self, result: Any, config: Dict[str, Any]) -> tuple:
        """
        複数値に変換

        計算結果を複数のnumpy配列のタプルに変換します。
        DataFrameの場合はreturn_colsに基づいて列を選択します。

        Args:
            result: 変換対象（DataFrame、Series、タプル等）
            config: pandas-ta設定（return_colsを含む）

        Returns:
            tuple: numpy配列のタプル
        """
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
        """
        DataFrameを複数値に変換

        DataFrameの列をnumpy配列のタプルに変換します。
        return_colsが指定されている場合はその列を選択し、
        指定されていない場合は全列を変換します。

        Args:
            result: 変換対象のDataFrame
            config: pandas-ta設定（return_colsを含む）

        Returns:
            tuple: 各列のnumpy配列を含むタプル

        列選択ロジック:
            - return_colsが指定されている場合:
              - 完全一致する列を優先
              - 部分一致する列をフォールバック
              - 見つからない場合はNaN配列
            - return_colsが指定されていない場合:
              - 全列を順番に変換
        """
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
