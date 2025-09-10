"""
テクニカル指標統合サービス

pandas-taを直接活用し、冗長なラッパーを削除した効率的な実装。
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import IndicatorConfig, indicator_registry, IndicatorResultType
from .technical_indicators.trend import TrendIndicators
from .data_validation import validate_data_length_with_fallback, create_nan_result
from .config.indicator_definitions import PANDAS_TA_CONFIG, POSITIONAL_DATA_FUNCTIONS




logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス"""

    def __init__(self):
        """サービスを初期化"""
        self.registry = indicator_registry

    def _get_indicator_config(self, indicator_type: str) -> IndicatorConfig:
        """指標設定を取得"""
        config = self.registry.get_indicator_config(indicator_type)
        if not config:
            raise ValueError(f"サポートされていない指標タイプ: {indicator_type}")
        return config

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, pd.Series, tuple, tuple[pd.Series, ...]]:
        """
        指定された指標を計算

        pandas-taを動的に使用し、設定ベースの効率的な実装。
        pandasオンリー移行対応により、pd.Seriesとtuple[pd.Series, ...]も返却可能。
        """
        try:
            # 1. pandas-ta動的処理を試行
            result = self._calculate_with_pandas_ta(df, indicator_type, params)
            if result is not None:
                return result

            # 2. 既存のアダプター方式にフォールバック
            config = self._get_indicator_config(indicator_type)
            if config.adapter_function:
                return self._calculate_with_adapter(df, indicator_type, params, config)
            else:
                raise ValueError(f"指標 {indicator_type} の実装が見つかりません")

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            raise
    def validate_data_length_with_fallback(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """
        データ長検証を強化 - data_validation.pyの強化版を使用

        Args:
            df: OHLCV価格データ
            indicator_type: 指標タイプ
            params: パラメータ辞書

        Returns:
            (データ長が十分かどうか, フォールバック可能な最小データ長)
        """
        # data_validation.pyの強化版を使用
        return validate_data_length_with_fallback(df, indicator_type, params)

    def _calculate_with_pandas_ta(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, tuple, None]:
        """
        pandas-taを使用した動的な指標計算

        Args:
            df: OHLCV価格データ
            indicator_type: 指標タイプ
            params: パラメータ辞書

        Returns:
            計算結果（対応していない場合はNone）
        """
        config = PANDAS_TA_CONFIG.get(indicator_type)
        if not config:
            return None  # フォールバック

        # データ長検証（強化版を使用）
        is_valid, required_length = self.validate_data_length_with_fallback(
            df, indicator_type, params
        )
        if not is_valid:
            logger.info(f"{indicator_type}: データ長不足のためNaNフィルタを適用")
            # data_validation.pyのcreate_nan_resultを使用
            nan_result = create_nan_result(df, indicator_type)
            if isinstance(nan_result, np.ndarray) and nan_result.ndim == 2:
                # 複数結果の場合、タプルにする
                return tuple(nan_result[:, i] for i in range(nan_result.shape[1]))
            return nan_result

        try:
            # パラメータ正規化
            normalized_params = {}
            for param_name, param_aliases in config["params"].items():
                value = None
                for alias in param_aliases:
                    if alias in params:
                        value = params[alias]
                        break

                # 値が見つからない場合はデフォルト値を使用
                if value is None:
                    value = config["default_values"].get(param_name)

                if value is not None:
                    normalized_params[param_name] = value

            # パラメータバリデーション（lengthパラメータがある場合のみ）
            if "length" in normalized_params and normalized_params["length"] <= 0:
                raise ValueError(
                    f"{indicator_type}: period must be positive: {normalized_params['length']}"
                )

            # Fallback indicators that skip pandas-ta and use manual implementations
            fallback_indicators = {"PPO", "STOCHF", "EMA", "TEMA", "ALMA", "FWMA", "CV", "IRM"}
            if indicator_type in fallback_indicators:
                return self._calculate_fallback_indicator(
                    df, indicator_type, normalized_params
                )

            # pandas-ta関数取得と実行
            if not hasattr(ta, config["function"]):
                logger.warning(f"pandas-ta function '{config['function']}' not found")
                return None

            func = getattr(ta, config["function"])

            # 複数カラムを使用する価格変換系指標の処理
            if config.get("multi_column", False):
                # data_columns からデータを取得
                data_args = {}
                for column in config["data_columns"]:
                    if column in df.columns:
                        # 正確なカラム名でマッピング
                        if column == "Open":
                            data_args["open"] = df[column]
                        elif column == "High":
                            data_args["high"] = df[column]
                        elif column == "Low":
                            data_args["low"] = df[column]
                        elif column == "Close":
                            data_args["close"] = df[column]

                # 必要なカラムが全て存在しない場合（大文字小文字を考慮）
                required_columns = config.get("data_columns", [])
                resolved_columns = {}
                for req_col in required_columns:
                    actual_col = self._resolve_column_name(df, req_col)
                    if not actual_col:
                        logger.warning(
                            f"{indicator_type}: 必要なカラム '{req_col}' が存在しません"
                        )
                        return None
                    resolved_columns[req_col] = actual_col

                # data_argsに解決されたカラムを使用
                data_args = {}
                for req_col in required_columns:
                    if req_col == "Open":
                        data_args["open"] = df[resolved_columns[req_col]]
                    elif req_col == "High":
                        data_args["high"] = df[resolved_columns[req_col]]
                    elif req_col == "Low":
                        data_args["low"] = df[resolved_columns[req_col]]
                    elif req_col == "Close":
                        data_args["close"] = df[resolved_columns[req_col]]
                    else:
                        data_args[req_col] = df[resolved_columns[req_col]]

                # pandas-taの関数によっては異なる引数名を使用する場合があるため、エラーハンドリングを追加
                try:
                    result = func(**data_args, **normalized_params)
                except TypeError as e:
                    # open引数がopen_である場合の処理
                    if "open" in data_args and (
                        "unexpected keyword argument 'open'" in str(e)
                        or "missing 1 required positional argument: 'open_'" in str(e)
                    ):
                        data_args["open_"] = data_args.pop("open")
                        result = func(**data_args, **normalized_params)
                    else:
                        raise
            else:
                # 単一カラムを使用する指標の処理
                column_name = self._resolve_column_name(df, config["data_column"])
                if not column_name:
                    logger.warning(
                        f"{indicator_type}: 必要なカラム '{config['data_column']}' が存在しません"
                    )
                    return None
                data_series = df[column_name]
                result = func(data_series, **normalized_params)

            # NaN処理: 結果がNaN多い場合にフィルタ
            if isinstance(result, pd.Series):
                if result.isna().sum() > len(result) * 0.7:  # 70%以上NaNの場合
                    logger.warning(f"{indicator_type}: NaN値が多すぎるためスキップ")
                    return None
                result = result.bfill().fillna(0)  # バックフィル、後方0
            elif isinstance(result, pd.DataFrame):
                if result.isna().sum().sum() > result.size * 0.7:
                    logger.warning(f"{indicator_type}: NaN値が多すぎるためスキップ")
                    return None
                result = result.bfill().fillna(0)

            # 戻り値処理
            if config["returns"] == "single":
                # Pandas Series の場合は numpy ndarray に変換
                if isinstance(result, pd.Series):
                    return result.values
                else:
                    return result
            else:  # multiple
                if result is None or result.empty:
                    raise ValueError(
                        f"pandas-ta function returned empty result for {indicator_type}"
                    )

                # タプルとして返す場合、個別の名前を付ける
                multiple_results = tuple(
                    result.iloc[:, i].values
                    for i in range(min(len(config["return_cols"]), result.shape[1]))
                )

                # 設定ベースの出力名を追加
                if hasattr(self, "registry") and self.registry:
                    config_obj = self._get_indicator_config(indicator_type)
                    if (
                        hasattr(config_obj, "output_names")
                        and config_obj.output_names
                        and len(config_obj.output_names) == len(multiple_results)
                    ):

                        # 出力に名前を付ける（デバッグ用途）
                        logger.debug(
                            f"{indicator_type} outputs: {config_obj.output_names}"
                        )
                        return multiple_results

                return multiple_results

        except Exception as e:
            logger.warning(f"pandas-ta計算失敗 {indicator_type}: {e}")
            return None  # フォールバックさせる

    def _calculate_fallback_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, tuple, None]:
        """
        Fallback implementation using TrendIndicators/MomentumIndicators classes
        for indicators that have pandas-ta issues
        """
        try:
            config = PANDAS_TA_CONFIG.get(indicator_type)
            if not config:
                return None

            if indicator_type == "PPO":
                column_name = self._resolve_column_name(df, "Close")
                if not column_name:
                    nan_array = np.full(len(df), np.nan)
                    return nan_array, nan_array, nan_array

                data_series = df[column_name]
                result = TrendIndicators.ppo(
                    data_series,
                    fast=params.get("fast", 12),
                    slow=params.get("slow", 26),
                    signal=params.get("signal", 9),
                )
                # PPO returns tuple of (ppo_line, signal_line, histogram)
                return result

            elif indicator_type == "STOCHF":
                high_column = self._resolve_column_name(df, "High")
                low_column = self._resolve_column_name(df, "Low")
                close_column = self._resolve_column_name(df, "Close")

                if not (high_column and low_column and close_column):
                    nan_array = np.full(len(df), np.nan)
                    return nan_array, nan_array

                high_series = df[high_column]
                low_series = df[low_column]
                close_series = df[close_column]

                result = TrendIndicators.stochf(
                    high_series,
                    low_series,
                    close_series,
                    length=params.get("fastd_length", 3),
                    fast_length=params.get("fastk_length", 5),
                )
                # STOCHF returns tuple of (fast_k, fast_d)
                return result

            elif indicator_type in {"EMA", "TEMA", "ALMA", "FWMA"}:
                column_name = self._resolve_column_name(df, config["data_column"])
                if not column_name:
                    return np.full(len(df), np.nan)

                data_series = df[column_name]

                if indicator_type == "EMA":
                    result = TrendIndicators.ema(
                        data_series, length=params.get("length", 20)
                    )
                elif indicator_type == "TEMA":
                    result = TrendIndicators.tema(
                        data_series, length=params.get("length", 14)
                    )
                elif indicator_type == "ALMA":
                    result = TrendIndicators.alma(
                        data_series,
                        length=params.get("length", 9),
                        sigma=params.get("sigma", 6.0),
                        offset=params.get("offset", 0.85),
                    )
                elif indicator_type == "FWMA":
                    result = TrendIndicators.fwma(
                        data_series, length=params.get("length", 10)
                    )

                # Return as single series values if needed
                if config["returns"] == "single":
                    if isinstance(result, pd.Series):
                        return result.values
                    else:
                        return result

                return result

        except Exception as e:
            logger.warning(
                f"Fallback indicator calculation failed for {indicator_type}: {e}"
            )
            # Return appropriate NaN array based on config
            if config and config["returns"] == "multiple":
                nan_array = np.full(len(df), np.nan)
                if indicator_type == "PPO":
                    return nan_array, nan_array, nan_array
                elif indicator_type == "STOCHF":
                    return nan_array, nan_array
                else:
                    return nan_array, nan_array
            else:
                return np.full(len(df), np.nan)

    def _calculate_with_adapter(
        self,
        df: pd.DataFrame,
        indicator_type: str,
        params: Dict[str, Any],
        config: IndicatorConfig,
    ):
        """アダプター関数を使用した指標計算（後方互換性用）"""
        # アダプター関数がNoneでないことを確認
        if not config.adapter_function:
            raise ValueError(
                f"Adapter function is not available for indicator {indicator_type}"
            )

        # 型チェックのため、adapter_functionがNoneでないことを確認した後の参照
        adapter_function = config.adapter_function

        # 従来のロジックを簡素化して維持
        required_data = {}

        # required_dataが空でもparam_mapを処理
        if (
            hasattr(config, "param_map")
            and config.param_map is not None
            and isinstance(config.param_map, dict)
        ):
            # param_mapのすべてのマッピングを処理
            for param_key, data_key in config.param_map.items():
                # data_key が None の場合はスキップ（パラメータマッピング用）
                if data_key is None:
                    continue
                column_name = self._resolve_column_name(df, param_key)
                if column_name:
                    # param_map は param_key -> data_key のマッピングなので、data_key をキーとして使用
                    required_data[data_key] = df[column_name]

        # 通常のrequired_data処理
        for data_key in config.required_data:
            # param_mapを使用してデータキーをマッピング
            # param_mapに含まれるキーは既に処理済みなのでスキップ
            if (
                hasattr(config, "param_map")
                and config.param_map is not None
                and isinstance(config.param_map, dict)
                and data_key in config.param_map.keys()
            ):
                continue  # param_mapで処理済みなのでスキップ

            column_name = self._resolve_column_name(df, data_key)

            if column_name:
                required_data[data_key] = df[column_name]

        # パラメータ正規化
        converted_params = config.normalize_params(params)

        # param_map を使用してパラメータ名をマッピング
        if (
            hasattr(config, "param_map")
            and config.param_map is not None
            and isinstance(config.param_map, dict)
        ):
            # パラメータ名をマッピング
            mapped_params = {}
            for param_key, param_value in converted_params.items():
                # param_map に定義されている場合はマッピング
                if param_key in config.param_map:
                    mapped_key = config.param_map[param_key]
                    # None の場合は無視（そのパラメータを使わない）
                    if mapped_key is not None:
                        mapped_params[mapped_key] = param_value
                    # None の場合は何もしない（パラメータを除外）
                else:
                    mapped_params[param_key] = param_value
            converted_params = mapped_params

            # param_map に data -> close のマッピングがある場合の特別処理
            # この場合、required_data から close を取得して data として渡す
            if "data" in config.param_map and config.param_map["data"] == "close":
                # required_data から close データを data として使用
                if "close" in required_data:
                    # close データを data として mapped_params に追加
                    mapped_params["data"] = required_data["close"]
                    # close を required_data から削除して重複を防ぐ
                    if "close" in required_data:
                        del required_data["close"]
                converted_params = mapped_params

        # デバッグ: all_argsの内容を確認
        all_args = {**required_data, **converted_params}

        # 関数シグネチャを動的に検査して呼び出し方を決定
        import inspect

        sig = inspect.signature(adapter_function)
        valid_params = set(sig.parameters.keys())

        # 位置引数を必要とする関数
        # 先頭のパラメータはデータ系列、残りはキーワード引数
        if indicator_type.lower() in POSITIONAL_DATA_FUNCTIONS:
            # 位置引数を必要とする関数の場合
            # 必要なデータを順序通りに位置引数として渡す
            positional_args = []
            keyword_args = {}

            # required_dataの順序で位置引数を構築
            for data_key in config.required_data:
                if data_key in all_args:
                    positional_args.append(all_args[data_key])
                    del all_args[data_key]

            # 残りのパラメータをキーワード引数として渡す
            # ただし、すでに位置引数として渡されたパラメータは除外
            # 関数シグネチャのチェックを追加して予期しないパラメータを除外
            # valid_params は既に上で定義済み

            keyword_args = {}
            for k, v in all_args.items():
                if k not in config.parameters.keys():
                    if k in valid_params:
                        keyword_args[k] = v
                    # 無効なパラメータは除外（何もしない）
                # config.parameters に含まれるパラメータは除外（何もしない）

            result = adapter_function(*positional_args, **keyword_args)
            if isinstance(result, pd.Series):
                return result.values
            elif (
                isinstance(result, pd.DataFrame)
                and config.result_type == IndicatorResultType.SINGLE
            ):
                return result.iloc[:, 0].values
            else:
                return result


        # 通常のキーワード引数呼び出し（フィルタリングを追加）
        # 有効なパラメータのみフィルタリング
        filtered_args = {}
        for k, v in all_args.items():
            if k in valid_params:
                filtered_args[k] = v
            elif k.lower() in valid_params:
                filtered_args[k.lower()] = v

        # MAVP指標の特別処理: periodsパラメータが不足している場合、デフォルト値を生成
        if indicator_type == "MAVP" and "periods" not in filtered_args:
            # periodsが提供されていない場合、デフォルト期間を使用
            default_period = params.get("default_period", 14)
            data_length = len(required_data.get("data", df))
            periods_series = pd.Series([default_period] * data_length, index=df.index)
            filtered_args["periods"] = periods_series
            logger.debug(f"MAVP: 生成したデフォルトperiods: {default_period}")

        result = adapter_function(**filtered_args)
        if isinstance(result, pd.Series):
            return result.values
        elif (
            isinstance(result, pd.DataFrame)
            and config.result_type == IndicatorResultType.SINGLE
        ):
            return result.iloc[:, 0].values
        else:
            return result

    def _resolve_column_name(self, df: pd.DataFrame, data_key: str) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決
        """
        # 優先順位: 元の名前 > 大文字 > 小文字 > Capitalized
        candidates = [data_key, data_key.upper(), data_key.lower(), data_key.capitalize()]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        return None

    def get_supported_indicators(self) -> Dict[str, Any]:
        """
        サポートされている指標の情報を取得

        Returns:
            サポート指標の情報
        """
        infos = {}
        for name, config in self.registry._configs.items():
            if not config.adapter_function:
                continue
            infos[name] = {
                "parameters": config.get_parameter_ranges(),
                "result_type": config.result_type.value,
                "required_data": config.required_data,
                "scale_type": config.scale_type.value if config.scale_type else None,
            }
        return infos
