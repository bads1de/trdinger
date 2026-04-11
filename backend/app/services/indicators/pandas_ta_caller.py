"""
pandas-ta呼び出しモジュール

pandas-taライブラリを使用したインジケーター計算を担当します。
"""

import inspect
import logging
from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta_classic as ta

from .indicator_validator import IndicatorValidator

logger = logging.getLogger(__name__)


class PandasTaCaller:
    """
    pandas-ta呼び出しクラス

    pandas-taライブラリを使用してインジケーターを計算します。
    インジケーターバリデーターと連携して、データ検証とエラーハンドリングを行います。

    Attributes:
        validator: インジケーターバリデーター
    """

    def __init__(self, validator: IndicatorValidator):
        """
        初期化

        Args:
            validator: インジケーターバリデーター
        """
        self.validator = validator

    def get_pandas_ta_config(
        self, indicator_type: str, registry: Any
    ) -> Optional[Dict[str, Any]]:
        """
        pandas-ta用の設定辞書を取得

        指標タイプに対応するpandas-ta設定をレジストリから取得し、
        パラメータマッピングを構築します。

        Args:
            indicator_type: 指標タイプ（例: 'RSI', 'MACD'）
            registry: インジケーターレジストリ

        Returns:
            Optional[Dict[str, Any]]: pandas-ta設定辞書（関数名、データカラム、パラメータ等）、
                                      設定が見つからない場合はNone

        設定内容:
            - function: pandas-ta関数名
            - data_column: データカラム（単一カラム用）
            - data_columns: データカラムリスト（複数カラム用）
            - returns: 戻り値タイプ（'single' または 'multiple'）
            - return_cols: 戻り値カラム名
            - multi_column: マルチカラムフラグ
            - params: パラメータマッピング
            - default_values: デフォルト値
            - min_length: 最小データ長関数
        """
        config = registry.get_indicator_config(indicator_type)
        if config and config.pandas_function:
            params_mapping: Dict[str, list[str]] = {}
            if config.param_map:
                for alias, target in config.param_map.items():
                    if target and target != "data":
                        if target not in params_mapping:
                            params_mapping[target] = []
                        if alias not in params_mapping[target]:
                            params_mapping[target].append(alias)
            else:
                for param_name in config.parameters.keys():
                    params_mapping[param_name] = [param_name]

            return {
                "function": config.pandas_function,
                "data_column": config.data_column,
                "data_columns": config.data_columns,
                "returns": config.returns,
                "return_cols": config.return_cols,
                "multi_column": config.multi_column,
                "params": params_mapping,
                "default_values": config.default_values,
                "min_length": config.min_length_func,
            }
        return None

    def call_pandas_ta(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        pandas-ta直接呼び出し

        pandas-taライブラリを使用してインジケーターを計算します。

        Args:
            df: データフレーム（OHLCVデータ）
            config: pandas-ta設定辞書
            params: パラメータ辞書

        Returns:
            Optional[Any]: 計算結果（Series、DataFrame、またはtuple）、エラー時はNone

        Note:
            - マルチカラム指標の場合は_call_multi_columnを使用
            - シングルカラム指標の場合は_call_single_columnを使用
            - 関数が存在しない場合は警告ログを出力してNoneを返す
        """
        try:
            if not hasattr(ta, config["function"]):
                logger.warning(f"pandas-ta関数 {config['function']} が存在しません")
                return None

            if config.get("multi_column", False):
                return self._call_multi_column(df, config, params)
            else:
                return self._call_single_column(df, config, params)

        except Exception as e:
            logger.error(f"pandas-ta呼び出し失敗: {config['function']}, エラー: {e}")
            return None

    def _call_multi_column(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        マルチカラム指標の呼び出し

        複数のデータカラムを必要とするインジケーターを計算します。
        OHLCVカラムを適切にマッピングしてpandas-ta関数を呼び出します。

        Args:
            df: データフレーム
            config: pandas-ta設定辞書
            params: パラメータ辞書

        Returns:
            Optional[Any]: 計算結果、エラー時はNone

        Note:
            - open, high, low, close, volumeカラムは位置引数として渡す
            - その他のカラムは位置引数として渡す
            - パラメータはキーワード引数として渡す
        """
        required_columns = config.get("data_columns", [])
        ta_args = {}
        positional_args = []

        for req_col in required_columns:
            col_name = self.validator.resolve_column_name(df, req_col)
            if col_name is None:
                logger.error(f"必須カラム '{req_col}' が存在しません")
                return None

            col_lower = req_col.lower()
            if col_lower in ["open", "high", "low", "close", "volume", "open_"]:
                ta_args[col_lower] = df[col_name]
            else:
                positional_args.append(df[col_name])

        combined_params = {**params, **ta_args}
        func = getattr(ta, config["function"])
        return func(*positional_args, **combined_params)

    def _call_single_column(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        シングルカラム指標の呼び出し

        単一のデータカラムを必要とするインジケーターを計算します。

        Args:
            df: データフレーム
            config: pandas-ta設定辞書
            params: パラメータ辞書

        Returns:
            Optional[Any]: 計算結果、エラー時はNone

        Note:
            - データカラムを第一引数として渡す
            - 関数の第一引数名と同じ名前のパラメータは削除する
            - データ長がlengthパラメータ未満の場合はエラー
        """
        col_name = self.validator.resolve_column_name(df, config.get("data_column"))
        if col_name is None:
            logger.error(f"必須カラム '{config.get('data_column')}' が存在しません")
            return None

        if len(df) < params.get("length", 0):
            logger.error(f"データ長({len(df)})がlength({params.get('length')})未満")
            return None

        # 安全策: 位置引数として渡すデータと同じ名前のパラメータがあれば削除
        call_params = params.copy()
        target_arg = config.get("data_column", "").lower()
        if target_arg in call_params:
            del call_params[target_arg]

        # 追加対策: 関数の第一引数名を取得して、それがparamsに含まれていれば削除
        func = getattr(ta, config["function"])
        try:
            sig = inspect.signature(func)
            first_param_name = list(sig.parameters.keys())[0]
            if first_param_name in call_params:
                del call_params[first_param_name]
        except Exception:
            pass

        return func(df[col_name], **call_params)
