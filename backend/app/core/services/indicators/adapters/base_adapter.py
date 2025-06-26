"""
TA-Lib アダプター基底クラス

全てのTA-Libアダプターで共通して使用される機能を提供します。
データ変換、入力検証、エラーハンドリングなどの共通処理を担当します。
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TALibCalculationError(Exception):
    """TA-Lib計算エラー"""


class BaseAdapter:
    """TA-Libアダプターの基底クラス"""

    @staticmethod
    def _ensure_series(data: Union[pd.Series, list, np.ndarray]) -> pd.Series:
        """
        データをpandas.Seriesに変換

        Args:
            data: 入力データ（pandas.Series, list, numpy.ndarray）

        Returns:
            pandas.Series

        Raises:
            TALibCalculationError: サポートされていないデータ型の場合
        """
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        else:
            raise TALibCalculationError(f"サポートされていないデータ型: {type(data)}")

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

    @staticmethod
    def _generate_indicator_name(
        indicator: str, parameters: Dict[str, Any], format_type: str = "json"
    ) -> str:
        """
        インジケーター名を生成（JSON形式：パラメータなし）

        Args:
            indicator: インジケーター名
            parameters: パラメータ辞書（使用されない）
            format_type: 形式タイプ（JSON形式固定）

        Returns:
            生成された名前（パラメータなし）
        """
        # JSON形式では指標名にパラメータを含めない
        return indicator

    @staticmethod
    def _generate_legacy_name(indicator: str, parameters: Dict[str, Any]) -> str:
        """
        レガシー形式の名前を生成（フォールバック用）

        Args:
            indicator: インジケーター名
            parameters: パラメータ辞書

        Returns:
            レガシー形式の名前
        """
        if not parameters:
            return indicator

        # 一般的なパターンに基づく生成
        if len(parameters) == 1:
            param_value = list(parameters.values())[0]
            return f"{indicator}_{param_value}"
        elif len(parameters) == 2:
            values = list(parameters.values())
            return f"{indicator}_{values[0]}_{values[1]}"
        elif len(parameters) == 3:
            values = list(parameters.values())
            return f"{indicator}_{values[0]}_{values[1]}_{values[2]}"
        else:
            # 複雑なパラメータの場合は基本名のみ
            return indicator

    @staticmethod
    def _create_series_result_with_config(
        result: np.ndarray,
        index: pd.Index,
        indicator: str,
        parameters: Dict[str, Any],
        format_type: str = "auto",
    ) -> pd.Series:
        """
        設定を考慮した計算結果のSeries変換

        Args:
            result: TA-Libの計算結果
            index: 元データのインデックス
            indicator: インジケーター名
            parameters: パラメータ辞書
            format_type: 名前の形式タイプ

        Returns:
            pandas.Series
        """
        name = BaseAdapter._generate_indicator_name(indicator, parameters, format_type)

        # JSON形式の場合は文字列に変換
        if isinstance(name, dict):
            # 簡易的な文字列表現
            if name.get("parameters"):
                param_str = "_".join(str(v) for v in name["parameters"].values())
                name_str = f"{name['indicator']}_{param_str}"
            else:
                name_str = name["indicator"]
        else:
            name_str = name

        return pd.Series(result, index=index, name=name_str)


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
