"""
TA-Lib アダプター基底クラス

全てのTA-Libアダプターで共通して使用される機能を提供します。
データ変換、入力検証、エラーハンドリングなどの共通処理を担当します。
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any
import logging

from app.core.utils.data_utils import ensure_series, DataConversionError

logger = logging.getLogger(__name__)


class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""

    pass


class BaseAdapter:
    """TA-Libアダプターの基底クラス"""

    @staticmethod
    def _ensure_series(data: Union[pd.Series, list, np.ndarray]) -> pd.Series:
        """
        データをpandas.Seriesに変換（data_utilsへの委譲）

        Args:
            data: 入力データ（pandas.Series, list, numpy.ndarray）

        Returns:
            pandas.Series

        Raises:
            TALibCalculationError: サポートされていないデータ型の場合
        """
        try:
            return ensure_series(data, raise_on_error=True)
        except DataConversionError as e:
            raise TALibCalculationError(str(e))

    @staticmethod
    def _validate_input(data: pd.Series, period: int) -> None:
        """
        入力データとパラメータの検証

        Args:
            data: 入力データ
            period: 期間

        Raises:
            TALibCalculationError: 入力が無効な場合
        """
        if data is None or len(data) == 0:
            raise TALibCalculationError("入力データが空です")

        if period <= 0:
            raise TALibCalculationError(f"期間は正の整数である必要があります: {period}")

        if len(data) < period:
            raise TALibCalculationError(
                f"データ長({len(data)})が期間({period})より短いです"
            )

    @staticmethod
    def _validate_multi_input(*data_series: pd.Series) -> None:
        """
        複数のデータ系列の長さが一致することを検証

        Args:
            *data_series: 検証対象のデータ系列

        Raises:
            TALibCalculationError: データ長が一致しない場合
        """
        if not data_series:
            raise TALibCalculationError("検証対象のデータが指定されていません")

        lengths = [len(series) for series in data_series]
        if not all(length == lengths[0] for length in lengths):
            raise TALibCalculationError(f"データ系列の長さが一致しません: {lengths}")

        if lengths[0] == 0:
            raise TALibCalculationError("入力データが空です")

    @staticmethod
    def _safe_talib_calculation(func, *args, **kwargs) -> np.ndarray:
        """
        TA-Lib計算の安全な実行

        Args:
            func: TA-Lib関数
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            計算結果のnumpy配列

        Raises:
            TALibCalculationError: 計算エラーの場合
        """
        try:
            # 引数をfloat64に変換してTA-Libに渡す
            converted_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    converted_args.append(arg.astype(np.float64))
                else:
                    converted_args.append(arg)

            return func(*converted_args, **kwargs)
        except Exception as e:
            raise TALibCalculationError(f"TA-Lib計算エラー: {e}")

    @staticmethod
    def _create_series_result(
        result: np.ndarray, index: pd.Index, name: str
    ) -> pd.Series:
        """
        計算結果をpandas.Seriesに変換

        Args:
            result: TA-Libの計算結果
            index: 元データのインデックス
            name: 系列名

        Returns:
            pandas.Series
        """
        return pd.Series(result, index=index, name=name)

    @staticmethod
    def _log_calculation_start(indicator_name: str, **params) -> None:
        """
        計算開始のログ出力

        Args:
            indicator_name: 指標名
            **params: パラメータ
        """
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.debug(f"{indicator_name}計算開始: {param_str}")

    @staticmethod
    def _log_calculation_error(indicator_name: str, error: Exception) -> None:
        """
        計算エラーのログ出力

        Args:
            indicator_name: 指標名
            error: エラー
        """
        logger.error(f"{indicator_name}計算でエラー: {error}")


# 後方互換性のためのヘルパー関数
def safe_talib_calculation(func, *args, **kwargs):
    """
    TA-Lib計算の安全な実行（後方互換性用）

    Args:
        func: TA-Lib関数
        *args: 位置引数
        **kwargs: キーワード引数

    Returns:
        計算結果

    Raises:
        TALibCalculationError: 計算エラーの場合
    """
    return BaseAdapter._safe_talib_calculation(func, *args, **kwargs)
