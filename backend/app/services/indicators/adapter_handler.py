"""
アダプターハンドラーモジュール

アダプター関数を使用したインジケーター計算を担当します。
"""

import inspect
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import IndicatorConfig, IndicatorResultType
from .indicator_validator import IndicatorValidator

logger = logging.getLogger(__name__)


class AdapterHandler:
    """
    アダプターハンドラークラス

    アダプター関数を使用したインジケーター計算を管理します。
    pandas_taやTA-Libなどの外部ライブラリの関数を呼び出すための
    アダプター機能を提供します。
    """

    def __init__(self, validator: IndicatorValidator):
        """
        AdapterHandlerを初期化します。

        Args:
            validator: インジケーターバリデーター（列名解決や結果検証に使用）
        """
        self.validator = validator

    def calculate_with_adapter(
        self,
        df: pd.DataFrame,
        indicator_type: str,
        params: Dict[str, Any],
        config: IndicatorConfig,
    ) -> Any:
        """
        アダプター関数を使用した指標計算

        外部ライブラリ（pandas_ta等）のアダプター関数を呼び出して
        インジケーターを計算します。パラメータのマッピング、データの準備、
        関数の呼び出し、結果の整列・変換を行います。

        Args:
            df: データフレーム（OHLCVデータを含む）
            indicator_type: インジケータータイプ（例: 'sma', 'rsi'）
            params: インジケーターパラメータ（例: {'length': 20}）
            config: インジケーター設定（アダプター関数、パラメータマップ等を含む）

        Returns:
            Any: 計算結果（numpy配列、タプル等）

        Raises:
            ValueError: アダプター関数が設定されていない場合
        """
        if not config.adapter_function:
            raise ValueError(
                f"Adapter function is not available for indicator {indicator_type}"
            )

        adapter_function = config.adapter_function

        required_data = self._prepare_adapter_data(df, config)
        converted_params, required_data = self._map_adapter_params(
            params, config, required_data
        )
        all_args = {**required_data, **converted_params}

        return self._call_adapter_function(
            adapter_function, all_args, indicator_type, config
        )

    def _prepare_adapter_data(
        self, df: pd.DataFrame, config: IndicatorConfig
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        アダプター関数に渡すデータを準備

        インジケーター設定のrequired_dataに基づいて、
        DataFrameから必要な列（open、high、low、close、volume等）を抽出します。

        Args:
            df: データフレーム（OHLCVデータを含む）
            config: インジケーター設定（required_dataを含む）

        Returns:
            Dict[str, Union[pd.Series, pd.DataFrame]]: 準備されたデータ辞書
                                                        キーは列名、値はSeriesまたはDataFrame
        """
        standard_keys = ["open", "high", "low", "close", "volume"]
        required_data: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

        if config.required_data:
            for key in config.required_data:
                key_lower = key.lower()
                if key_lower in standard_keys:
                    col_name = self.validator.resolve_column_name(df, key_lower)
                    if col_name:
                        required_data[key_lower] = df[col_name]
                elif key_lower in ["data", "df", "ohlcv"]:
                    required_data[key_lower] = df
                else:
                    col_name = self.validator.resolve_column_name(df, key)
                    if col_name:
                        required_data[key] = df[col_name]

        if not required_data:
            col_name = self.validator.resolve_column_name(df, "close")
            if col_name:
                required_data["close"] = df[col_name]
            required_data["data"] = df

        return required_data

    def _map_adapter_params(
        self,
        params: Dict[str, Any],
        config: IndicatorConfig,
        required_data: Dict[str, Union[pd.Series, pd.DataFrame]],
    ) -> Tuple[Dict[str, Any], Dict[str, Union[pd.Series, pd.DataFrame]]]:
        """
        アダプター関数のパラメータをマッピング

        パラメータを正規化し、param_mapに基づいてデータ列をパラメータにマッピングします。

        Args:
            params: ユーザー指定のパラメータ
            config: インジケーター設定（param_mapを含む）
            required_data: 必要なデータ辞書

        Returns:
            Tuple[Dict[str, Any], Dict[str, Union[pd.Series, pd.DataFrame]]]:
                (マッピングされたパラメータ, 必要なデータ) のタプル
        """
        converted_params = config.normalize_params(params)
        mapped_params = converted_params.copy()

        if config.param_map:
            for source_name, arg_name in config.param_map.items():
                if source_name in required_data and arg_name:
                    mapped_params[arg_name] = required_data[source_name]

        return mapped_params, required_data

    def _call_adapter_function(
        self,
        adapter_function: Any,
        all_args: Dict[str, Any],
        indicator_type: str,
        config: IndicatorConfig,
    ) -> Any:
        """
        アダプター関数を呼び出し

        関数シグネチャに基づいてパラメータを割り当て、アダプター関数を実行し、
        結果を整列・変換します。実行に失敗した場合はNaN結果を返します。

        Args:
            adapter_function: アダプター関数（callable）
            all_args: すべての引数（パラメータとデータの結合）
            indicator_type: インジケータータイプ（ログ用）
            config: インジケーター設定

        Returns:
            Any: 整列・変換された計算結果
        """
        sig = inspect.signature(adapter_function)

        series_data = {
            k: v
            for k, v in all_args.items()
            if isinstance(v, (pd.Series, pd.DataFrame))
        }
        scalar_params = {
            k: v
            for k, v in all_args.items()
            if not isinstance(v, (pd.Series, pd.DataFrame))
        }
        reference_index = next(
            (
                value.index
                for value in series_data.values()
                if isinstance(value, (pd.Series, pd.DataFrame))
            ),
            None,
        )

        assigned_params: Dict[str, Any] = self._assign_parameters(
            sig, all_args, series_data, scalar_params
        )

        result = self._execute_adapter_function(
            adapter_function, assigned_params, indicator_type, all_args
        )

        if result is None:
            input_ref = all_args.get("data", all_args.get("close"))
            fallback_input = input_ref if input_ref is not None else pd.DataFrame()
            return self.validator.create_nan_result(
                fallback_input, {"function": indicator_type}
            )

        result = self._align_adapter_result(result, reference_index)

        return self._convert_adapter_result(result, config)

    def _assign_parameters(
        self,
        sig: inspect.Signature,
        all_args: Dict[str, Any],
        series_data: Dict[str, Any],
        scalar_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        パラメータを割り当て

        関数シグネチャのパラメータ名に基づいて、引数を割り当てます。
        共通のパラメータ名（data、close、length等）に対して柔軟なマッピングを行います。

        マッピングの優先順位:
        1. 明示的にall_argsで指定されたパラメータ
        2. パラメータ名の大文字小文字を無視した一致
        3. デフォルト値（param.defaultが設定されている場合）
        4. 一致するパラメータがない場合、そのパラメータはスキップされる

        Args:
            sig: 関数シグネチャ
            all_args: すべての引数
            series_data: シリーズデータ（DataFrame、Series）
            scalar_params: スカラーパラメータ

        Returns:
            Dict[str, Any]: 関数に渡す割り当てられたパラメータ。
                マッピングできなかったパラメータは含まれません。
        """
        assigned_params: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            param_lower = param_name.lower()
            val = None

            if param_name in all_args:
                val = all_args[param_name]
            elif param_lower in [
                "data",
                "df",
                "ohlcv",
                "close",
                "open",
                "open_",
                "high",
                "low",
                "volume",
                "trend",
                "series",
                "market_cap",
                "funding_rate",
                "open_interest",
            ]:
                target_key = param_lower.rstrip("_")
                val = series_data.get(target_key)
                if val is None:
                    val = series_data.get("close")
                if val is None:
                    val = series_data.get("data")
            elif param_lower in ["length", "period", "window", "n"]:
                for k in ["length", "period", "window", "n"]:
                    if k in scalar_params:
                        val = scalar_params[k]
                        break

            if val is not None:
                assigned_params[param_name] = val

        for k, v in all_args.items():
            if k not in assigned_params and k in sig.parameters:
                assigned_params[k] = v

        return assigned_params

    def _execute_adapter_function(
        self,
        adapter_function: Any,
        assigned_params: Dict[str, Any],
        indicator_type: str,
        all_args: Dict[str, Any],
    ) -> Any:
        """
        アダプター関数を実行

        キーワード引数での呼び出しを優先し、失敗した場合は位置引数で呼び出します。

        Args:
            adapter_function: アダプター関数（callable）
            assigned_params: 割り当てられたパラメータ
            indicator_type: インジケータータイプ（ログ用）
            all_args: すべての引数（フォールバック用）

        Returns:
            Any: 実行結果（失敗時はNone）
        """
        sig = inspect.signature(adapter_function)
        try:
            valid_params = {
                k: v for k, v in assigned_params.items() if k in sig.parameters
            }
            result = adapter_function(**valid_params)
        except TypeError:
            try:
                result = adapter_function(*assigned_params.values())
            except Exception as e:
                logger.error(f"{indicator_type} 計算エラー: {e}")
                result = None
        except Exception as e:
            logger.error(f"{indicator_type} 計算エラー: {e}")
            result = None

        return result

    def _align_adapter_result(self, result: Any, reference_index: Any) -> Any:
        """
        入力 index に合わせて adapter の結果を再整列する

        アダプター関数の結果のインデックスを入力DataFrameのインデックスに合わせます。
        Series、DataFrame、タプルに対応します。

        Args:
            result: 計算結果（Series、DataFrame、タプル等）
            reference_index: 参照インデックス（入力DataFrameのインデックス）

        Returns:
            Any: 整列された結果（インデックスが揃えられたSeries/DataFrame）
        """
        if reference_index is None:
            return result

        if isinstance(result, pd.Series):
            return self._align_series_like_result(result, reference_index)

        if isinstance(result, pd.DataFrame):
            return self._align_series_like_result(result, reference_index)

        if isinstance(result, tuple):
            return tuple(
                self._align_adapter_result(item, reference_index) for item in result
            )

        return result

    def _align_series_like_result(
        self,
        result: Union[pd.Series, pd.DataFrame],
        reference_index: pd.Index,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Series/DataFrame を入力 index に位置またはラベルで整列する

        インデックスが一致する場合はそのまま返し、長さが同じ場合はインデックスを置換し、
        位置インデックス（RangeIndex等）の場合はラベルにマッピングしてreindexします。

        Args:
            result: 結果（SeriesまたはDataFrame）
            reference_index: 参照インデックス

        Returns:
            Union[pd.Series, pd.DataFrame]: 整列された結果
        """
        if result.index.equals(reference_index):
            return result

        if len(result) == len(reference_index):
            aligned = result.copy()
            aligned.index = reference_index
            return aligned

        positional_labels = self._map_positional_index_to_reference(
            result.index,
            reference_index,
        )
        if positional_labels is not None:
            aligned = result.copy()
            aligned.index = positional_labels
            try:
                return aligned.reindex(reference_index)
            except Exception:
                return aligned

        try:
            return result.reindex(reference_index)
        except Exception:
            return result

    def _map_positional_index_to_reference(
        self,
        result_index: pd.Index,
        reference_index: pd.Index,
    ) -> Optional[pd.Index]:
        """
        RangeIndex などの位置 index を参照 index のラベルへ写像する

        整数型のインデックスを位置として扱い、参照インデックスの対応するラベルに変換します。

        Args:
            result_index: 結果インデックス（整数型の位置インデックス）
            reference_index: 参照インデックス（ラベル付き）

        Returns:
            Optional[pd.Index]: 写像されたインデックス、または写像不可能な場合はNone
        """
        if len(result_index) == 0 or len(reference_index) == 0:
            return None

        if not pd.api.types.is_integer_dtype(result_index):
            return None

        positions = np.asarray(result_index, dtype=int)
        if positions.ndim != 1:
            return None

        if (positions < 0).any() or (positions >= len(reference_index)).any():
            return None

        return reference_index.take(positions)

    def _convert_adapter_result(self, result: Any, config: IndicatorConfig) -> Any:
        """
        アダプター結果を変換

        pandasのSeries/DataFrameをnumpy配列に変換します。
        result_typeに基づいてDataFrameの複数列をタプルに変換するか、
        最初の列のみを返すかを決定します。

        Args:
            result: 計算結果（Series、DataFrame、タプル等）
            config: インジケーター設定（result_typeを含む）

        Returns:
            Any: 変換された結果（numpy配列、タプル等）
        """
        if isinstance(result, pd.Series):
            return result.to_numpy()
        elif isinstance(result, pd.DataFrame):
            if config.result_type == IndicatorResultType.SINGLE:
                return result.iloc[:, 0].to_numpy()
            else:
                return tuple(result[col].to_numpy() for col in result.columns)
        elif isinstance(result, tuple):
            return tuple(
                (
                    arr.to_numpy()
                    if isinstance(arr, pd.Series)
                    else (
                        arr.to_numpy()
                        if isinstance(arr, pd.DataFrame)
                        else np.asarray(arr)
                    )
                )
                for arr in result
            )
        else:
            return result
