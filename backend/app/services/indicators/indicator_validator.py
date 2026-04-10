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
    """

    def basic_validation(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> bool:
        """
        基本検証 - データ長と必須カラムのチェック

        Args:
            df: データフレーム
            config: pandas-ta設定
            params: パラメータ

        Returns:
            検証結果
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
            if not self.resolve_column_name(df, config.get("data_column")):
                return False

        return True

    def create_nan_result(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Union[np.ndarray, tuple]:
        """
        NaN結果を作成

        Args:
            df: データフレーム
            config: pandas-ta設定

        Returns:
            NaN結果
        """
        nan_result = create_nan_result(df, config["function"])
        if isinstance(nan_result, np.ndarray) and nan_result.ndim == 2:
            return tuple(nan_result[:, i] for i in range(nan_result.shape[1]))
        return nan_result

    def resolve_column_name(
        self, df: pd.DataFrame, data_key: Any
    ) -> Any:
        """
        データフレームから適切なカラム名を解決

        Args:
            df: データフレーム
            data_key: データキー

        Returns:
            カラム名、またはNone
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
