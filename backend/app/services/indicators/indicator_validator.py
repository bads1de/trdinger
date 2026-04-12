"""
インジケーターバリデーションモジュール

インジケーター計算前のバリデーションを担当します。
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from .data_validation import create_nan_result, validate_data_length_with_fallback

logger = logging.getLogger(__name__)


class IndicatorValidator:
    """
    インジケーターバリデーションクラス

    データ長と必須カラムのチェックを行います。
    インジケーター計算前のバリデーションを担当し、
    データが計算に適しているかどうかを確認します。
    """

    def basic_validation(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> bool:
        """
        基本検証 - データ長と必須カラムのチェック

        データフレームの長さがインジケーター計算に十分であるか、
        必要なカラムが存在するかを確認します。

        Args:
            df: 検証対象のDataFrame（OHLCVデータを含む）
            config: pandas-ta設定（function、data_column、data_columns等を含む）
            params: インジケーターパラメータ

        Returns:
            bool: 検証に合格した場合はTrue、不合格の場合はFalse
        """
        is_valid, _ = validate_data_length_with_fallback(df, config["function"], params)
        if not is_valid:
            return False

        if config.get("multi_column", False):
            required_columns = config.get("data_columns", [])
            for req_col in required_columns:
                if not self.resolve_column_name(df, req_col):
                    return False
        else:
            data_column = config.get("data_column")
            if data_column is not None and not self.resolve_column_name(
                df, data_column
            ):
                return False

        return True

    def create_nan_result(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Union[np.ndarray, tuple]:
        """
        NaN結果を作成

        バリデーション失敗や計算エラー時に返すNaN配列を作成します。
        入力DataFrameと同じ長さのNaN配列を返します。

        Args:
            df: 対象のDataFrame（長さを取得するために使用）
            config: pandas-ta設定（function名を含む）

        Returns:
            Union[np.ndarray, tuple]: NaN配列またはNaN配列のタプル
                                      （複数出力インジケーターの場合）
        """
        nan_result = create_nan_result(df, config["function"])
        if isinstance(nan_result, np.ndarray) and nan_result.ndim == 2:
            return tuple(nan_result[:, i] for i in range(nan_result.shape[1]))
        return nan_result

    def resolve_column_name(self, df: pd.DataFrame, data_key: str) -> str | None:
        """
        データフレームから適切なカラム名を解決

        指定されたデータキー（カラム名のエイリアス等）に対応する
        実際のカラム名をDataFrameから検索します。
        大文字小文字のバリエーションや末尾のアンダースコアを考慮します。

        Args:
            df: 対象のDataFrame
            data_key: データキー（例: 'close', 'Close', 'close_'）

        Returns:
            Any: 見つかったカラム名、見つからない場合はNone

        検索順序:
            1. data_key（そのまま）
            2. data_key.upper()
            3. data_key.lower()
            4. data_key.capitalize()
            5. clean_key（末尾のアンダースコアを削除）
            6. clean_key.upper()
            7. clean_key.lower()
            8. clean_key.capitalize()
        """
        if data_key is None:
            return None

        clean_key = data_key.rstrip("_")
        candidates = [
            data_key,
            data_key.upper(),
            data_key.lower(),
            data_key.capitalize(),
            clean_key,
            clean_key.upper(),
            clean_key.lower(),
            clean_key.capitalize(),
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        return None
