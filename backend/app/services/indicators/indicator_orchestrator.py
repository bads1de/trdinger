"""
テクニカル指標統合サービス

pandas-taを直接活用し、冗長なラッパーを削除した効率的な実装。
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import IndicatorConfig, indicator_registry

logger = logging.getLogger(__name__)

# Constants
# pandas-ta動的処理設定
PANDAS_TA_CONFIG = {
    "RSI": {
        "function": "rsi",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "WMA": {
        "function": "wma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "MACD": {
        "function": "macd",
        "params": {"fast": ["fast"], "slow": ["slow"], "signal": ["signal"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["MACD", "Signal", "Histogram"],
        "default_values": {"fast": 12, "slow": 26, "signal": 9},
    },
}

POSITIONAL_DATA_FUNCTIONS = {
    "rsi",
    "wma",
    "sar",
    "roc",
    "stoch",
    "bbands",
    "macd",
    "dpo",
    "rmi",
    "kama",
    "trima",
    "wma",
    "ma",
    "midpoint",
    "midprice",
    "ht_trendline",
    "adosc",
    "correl",
    "linearreg",
    "stddev",
    "tsf",
    "var",
    "linearreg_angle",
    "linearreg_intercept",
    "linearreg_slope",
    "hma",
    "zlma",
    "swma",
    "alma",
    "rma",
    "tsi",
    "pvo",
    "cfo",
    "cti",
    "sma_slope",
    "price_ema_ratio",
    "beta",
    "belta",
    "qqe",
    "smi",
    "trix",
    "apo",
    "macdext",
    "macdfix",
    "WMA",
    "TRIMA",
    "MA",
    "chop",
    "vortex",
    "BBANDS",
    "hilo",  # HILO指標を位置引数関数に追加
    # Volume indicators that need positional arguments
    "ad",  # Accumulation/Distribution
    "eom",  # Ease of Movement
    "kvo",  # Klinger Volume Oscillator
    "cmf",  # Chaikin Money Flow
}


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
    ) -> Union[np.ndarray, tuple]:
        """
        指定された指標を計算

        pandas-taを動的に使用し、設定ベースの効率的な実装。
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
                data_series = df[config["data_column"]]
                result = func(data_series, **normalized_params)

            # 戻り値処理
            if config["returns"] == "single":
                return result.values
            else:  # multiple
                if result is None or result.empty:
                    raise ValueError(
                        f"pandas-ta function returned empty result for {indicator_type}"
                    )
                return tuple(
                    result.iloc[:, i].values
                    for i in range(min(len(config["return_cols"]), result.shape[1]))
                )

        except Exception as e:
            logger.warning(f"pandas-ta計算失敗 {indicator_type}: {e}")
            return None  # フォールバックさせる

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

        from .parameter_manager import normalize_params

        converted_params = normalize_params(indicator_type, params, config)

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

            return adapter_function(*positional_args, **keyword_args)

        # dataパラメータが含まれているがcloseが含まれていない場合
        # dataを最初の位置引数として渡す（一部の関数で期待される形式）
        elif "data" in all_args and "close" not in all_args:
            # dataを位置引数として渡し、他のパラメータをキーワード引数として渡す
            data_arg = all_args.pop("data")
            return adapter_function(data_arg, **all_args)

        # Pass data as first positional argument (format expected by some functions)
        elif "data" in all_args and "close" not in all_args:
            # Pass data as positional argument and others as keyword arguments
            data_arg = all_args.pop("data")
            return adapter_function(data_arg, **all_args)

        # 通常のキーワード引数呼び出し
        return adapter_function(**all_args)

    def _resolve_column_name(self, df: pd.DataFrame, data_key: str) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決（簡素化版）
        """
        # 特別なマッピング
        key_mappings = {
            "open_data": "open",
            "open_": "open",
            "data0": "high",
            "data1": "low",
        }

        actual_key = key_mappings.get(data_key, data_key)

        # 大文字小文字のバリエーションをチェック
        for variant in [
            actual_key.capitalize(),
            actual_key.upper(),
            actual_key.lower(),
        ]:
            if variant in df.columns:
                return variant

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
