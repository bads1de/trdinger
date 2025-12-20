"""
テクニカル指標統合サービス

pandas-taと独自実装のテクニカル指標を統一的に管理し、
効率的な計算とキャッシュを提供します。

主な特徴:
- 動的指標検出による自動設定
- pandas-ta直接呼び出しによる高効率
- 独自実装へのフォールバック（アダプター方式）
- 計算結果のLRUキャッシュ
"""

import inspect
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from cachetools import LRUCache

from .config import IndicatorConfig, IndicatorResultType, indicator_registry
from .data_validation import create_nan_result, validate_data_length_with_fallback

logger = logging.getLogger(__name__)

# 後方互換性のためのエイリアスマッピング
# 注意: 新しいコードではレジストリのエイリアス機能を使用してください
INDICATOR_ALIASES = {
    "BB": "BBANDS",
    "KC": "KC",
    "DONCHIAN": "DONCHIAN",
    "STOCH": "STOCH",
    "AO": "AO",
    "APO": "APO",
    "BOP": "BOP",
    "MOMENTUM": "MOM",
    "CG": "CG",
    "AD": "AD",
    "ADOSC": "ADOSC",
    "MFI": "MFI",
    "OBV": "OBV",
    "RSI": "RSI",
    "SMA": "SMA",
    "EMA": "EMA",
    "WMA": "WMA",
}


class TechnicalIndicatorService:
    """テクニカル指標統合サービス

    pandas-ta と独自実装のテクニカル指標を統一的なインターフェースで提供し、
    オートストラテジー（GA）からの利用を最適化します。
    """

    # 計算結果キャッシュ（クラスレベル）
    _calculation_cache: LRUCache = LRUCache(maxsize=1000)

    def __init__(self):
        """サービスを初期化"""
        self.registry = indicator_registry
        self.registry.ensure_initialized()

    def _get_indicator_config(self, indicator_type: str) -> IndicatorConfig:
        """指標設定を取得"""
        config = self.registry.get_indicator_config(indicator_type)
        if not config:
            raise ValueError(f"サポートされていない指標タイプ: {indicator_type}")
        return config

    def _resolve_indicator_name(self, indicator_type: str) -> str:
        """指標名を正規化（大文字変換）"""
        return indicator_type.upper()

    def clear_cache(self) -> None:
        """計算キャッシュをクリアする"""
        self._calculation_cache.clear()
        logger.info("Indicator calculation cache cleared.")

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, pd.Series, tuple, tuple[pd.Series, ...]]:
        """
        OHLCVデータから指定されたテクニカル指標を計算

        Args:
            df: OHLCVデータを含むDataFrame
            indicator_type: 指標タイプ名（例: RSI, MACD, SMA）
            params: 指標のパラメータ

        Returns:
            計算結果（numpy配列またはタプル）
        """
        indicator_type = indicator_type.upper()

        # キャッシュチェック
        cache_key = self._make_cache_key(indicator_type, params, df)
        if cache_key and cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]

        try:
            # pandas-ta設定を取得
            pandas_config = self._get_pandas_ta_config(indicator_type)
            result = None

            if pandas_config:
                # pandas-ta方式で処理
                normalized_params = self._normalize_params(params, pandas_config)

                if not self._basic_validation(df, pandas_config, normalized_params):
                    result = self._create_nan_result(df, pandas_config)
                else:
                    raw_result = self._call_pandas_ta(
                        df, pandas_config, normalized_params
                    )
                    if raw_result is not None:
                        result = self._post_process(raw_result, pandas_config, df)

            # アダプター方式にフォールバック
            if result is None:
                try:
                    config_obj = self._get_indicator_config(indicator_type)
                    if config_obj.adapter_function:
                        result = self._calculate_with_adapter(
                            df, indicator_type, params, config_obj
                        )
                    else:
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )
                except ValueError:
                    if pandas_config:
                        result = self._create_nan_result(df, pandas_config)
                    else:
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )

            # キャッシュに保存
            if result is not None and cache_key:
                self._calculation_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            raise

    def _make_cache_key(
        self, indicator_type: str, params: Dict[str, Any], df: pd.DataFrame
    ) -> Optional[tuple]:
        """キャッシュキーを生成"""
        try:
            cache_params = frozenset(sorted([(k, str(v)) for k, v in params.items()]))
            return (indicator_type, cache_params, id(df))
        except Exception:
            return None

    def _get_pandas_ta_config(self, indicator_type: str) -> Optional[Dict[str, Any]]:
        """pandas-ta用の設定辞書を取得"""
        config = self.registry.get_indicator_config(indicator_type)
        if config and config.pandas_function:
            params_mapping = {}
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

    def _basic_validation(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> bool:
        """基本検証 - データ長と必須カラムのチェック"""
        is_valid, _ = validate_data_length_with_fallback(df, config["function"], params)
        if not is_valid:
            return False

        if config.get("multi_column", False):
            required_columns = config.get("data_columns", [])
            for req_col in required_columns:
                if not self._resolve_column_name(df, req_col):
                    return False
        else:
            if not self._resolve_column_name(df, config.get("data_column")):
                return False

        return True

    def _create_nan_result(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Union[np.ndarray, tuple]:
        """NaN結果を作成"""
        nan_result = create_nan_result(df, config["function"])
        if isinstance(nan_result, np.ndarray) and nan_result.ndim == 2:
            return tuple(nan_result[:, i] for i in range(nan_result.shape[1]))
        return nan_result

    def _normalize_params(
        self, params: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        入力パラメータをpandas-taが期待する形式に正規化

        エイリアスの解決、デフォルト値の補完、最小値ガードを適用します。
        """
        normalized = {}
        for param_name, aliases in config["params"].items():
            value = None
            for alias in aliases:
                if alias in params:
                    value = params[alias]
                    break

            if value is None:
                value = config["default_values"].get(param_name)

            if value is not None:
                # min_lengthガードの適用
                if param_name in ["length", "period"] and "min_length" in config:
                    min_length_func = config["min_length"]
                    if callable(min_length_func):
                        min_length = min_length_func({param_name: value})
                        if isinstance(value, (int, float)) and value < min_length:
                            logger.warning(
                                f"パラメータ {param_name}={value} が最小値 {min_length} 未満のため調整"
                            )
                            value = min_length
                    elif (
                        isinstance(min_length_func, (int, float))
                        and isinstance(value, (int, float))
                        and value < min_length_func
                    ):
                        value = min_length_func

                normalized[param_name] = value

        return normalized

    def _call_pandas_ta(
        self, df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Any]:
        """pandas-ta直接呼び出し"""
        try:
            if not hasattr(ta, config["function"]):
                logger.warning(f"pandas-ta関数 {config['function']} が存在しません")
                return None

            func = getattr(ta, config["function"])

            if config.get("multi_column", False):
                required_columns = config.get("data_columns", [])
                ta_args = {}
                positional_args = []

                for req_col in required_columns:
                    col_name = self._resolve_column_name(df, req_col)
                    if col_name is None:
                        logger.error(f"必須カラム '{req_col}' が存在しません")
                        return None

                    col_lower = req_col.lower()
                    if col_lower in ["open", "high", "low", "close", "volume"]:
                        ta_args[col_lower] = df[col_name]
                    else:
                        positional_args.append(df[col_name])

                combined_params = {**params, **ta_args}
                return func(*positional_args, **combined_params)
            else:
                col_name = self._resolve_column_name(df, config.get("data_column"))
                if col_name is None:
                    logger.error(
                        f"必須カラム '{config.get('data_column')}' が存在しません"
                    )
                    return None

                if len(df) < params.get("length", 0):
                    logger.error(
                        f"データ長({len(df)})がlength({params.get('length')})未満"
                    )
                    return None

                return func(df[col_name], **params)

        except Exception as e:
            logger.error(f"pandas-ta呼び出し失敗: {config['function']}, エラー: {e}")
            return None

    def _post_process(
        self, result: Any, config: Dict[str, Any], df: Optional[pd.DataFrame] = None
    ) -> Union[np.ndarray, tuple]:
        """後処理 - 戻り値の統一"""
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
            result = result.bfill().fillna(0)

        # 戻り値変換
        if config["returns"] == "single":
            if isinstance(result, pd.Series):
                return result.values
            elif isinstance(result, pd.DataFrame):
                return result.iloc[:, 0].values
            else:
                return np.asarray(result)
        else:
            if isinstance(result, pd.DataFrame):
                if "return_cols" in config and config["return_cols"]:
                    selected_cols = []
                    for col in config["return_cols"]:
                        if col in result.columns:
                            selected_cols.append(result[col].values)
                        else:
                            matching_cols = [
                                c
                                for c in result.columns
                                if col in c or col.lower() in c.lower()
                            ]
                            if matching_cols:
                                selected_cols.append(result[matching_cols[0]].values)
                            else:
                                selected_cols.append(np.full(len(result), np.nan))
                    return tuple(selected_cols)
                else:
                    return tuple(
                        result.iloc[:, i].values for i in range(result.shape[1])
                    )
            elif isinstance(result, pd.Series):
                return (result.values,)
            elif isinstance(result, tuple):
                return tuple(np.asarray(arr) for arr in result)
            else:
                return (np.asarray(result),)

    def _prepare_adapter_data(
        self, df: pd.DataFrame, config: IndicatorConfig
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """アダプター関数に渡すデータを準備"""
        standard_keys = ["open", "high", "low", "close", "volume"]
        required_data: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

        if config.required_data:
            for key in config.required_data:
                key_lower = key.lower()
                if key_lower in standard_keys:
                    col_name = self._resolve_column_name(df, key_lower)
                    if col_name:
                        required_data[key_lower] = df[col_name]
                elif key_lower in ["data", "df", "ohlcv"]:
                    required_data[key_lower] = df
                else:
                    col_name = self._resolve_column_name(df, key)
                    if col_name:
                        required_data[key] = df[col_name]

        if not required_data:
            col_name = self._resolve_column_name(df, "close")
            if col_name:
                required_data["close"] = df[col_name]
            required_data["data"] = df

        return required_data

    def _map_adapter_params(
        self,
        params: Dict[str, Any],
        config: IndicatorConfig,
        required_data: Dict[str, pd.Series],
    ) -> Tuple[Dict[str, Any], Dict[str, pd.Series]]:
        """アダプター関数のパラメータをマッピング"""
        converted_params = config.normalize_params(params)
        mapped_params = converted_params.copy()

        if config.param_map:
            for source_name, arg_name in config.param_map.items():
                if source_name in required_data and arg_name:
                    mapped_params[arg_name] = required_data[source_name]
                    del required_data[source_name]

        return mapped_params, required_data

    def _call_adapter_function(
        self,
        adapter_function: Any,
        all_args: Dict[str, Any],
        indicator_type: str,
        config: IndicatorConfig,
    ) -> Any:
        """アダプター関数を呼び出し"""
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
                "high",
                "low",
                "volume",
            ]:
                target_key = param_lower.rstrip("_")
                val = series_data.get(target_key) or series_data.get("data")
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

        if result is None:
            input_ref = all_args.get("data", all_args.get("close"))
            data_len = len(input_ref) if input_ref is not None else 0
            if config.result_type == IndicatorResultType.SINGLE:
                return np.full(data_len, np.nan)
            else:
                num_cols = len(config.return_cols) if config.return_cols else 1
                return tuple(np.full(data_len, np.nan) for _ in range(num_cols))

        # 結果の後処理
        if isinstance(result, pd.Series):
            return result.bfill().fillna(0).values
        elif isinstance(result, pd.DataFrame):
            result = result.bfill().fillna(0)
            if config.result_type == IndicatorResultType.SINGLE:
                return result.iloc[:, 0].values
            else:
                return tuple(result[col].values for col in result.columns)
        elif isinstance(result, tuple):
            return tuple(
                (
                    arr.bfill().fillna(0).values
                    if isinstance(arr, pd.Series)
                    else (
                        arr.bfill().fillna(0).values
                        if isinstance(arr, pd.DataFrame)
                        else np.nan_to_num(np.asarray(arr))
                    )
                )
                for arr in result
            )
        else:
            return result

    def _calculate_with_adapter(
        self,
        df: pd.DataFrame,
        indicator_type: str,
        params: Dict[str, Any],
        config: IndicatorConfig,
    ):
        """アダプター関数を使用した指標計算"""
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

    def _resolve_column_name(
        self, df: pd.DataFrame, data_key: Optional[str]
    ) -> Optional[str]:
        """データフレームから適切なカラム名を解決"""
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

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報辞書
        """
        infos = {}
        for name, config in self.registry.get_all_indicators().items():
            if not config.adapter_function and not config.pandas_function:
                continue
            infos[name] = {
                "parameters": config.get_parameter_ranges(),
                "result_type": config.result_type.value,
                "required_data": config.required_data,
                "scale_type": config.scale_type.value if config.scale_type else None,
            }
        return infos
