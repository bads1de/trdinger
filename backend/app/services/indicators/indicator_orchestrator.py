"""
テクニカル指標統合サービス（簡素化版）

pandas-taの利点を最大限に活用しつつ、実装の複雑さを大幅に削減した効率的なサービス。
82個のテクニカル指標を統一的に管理し、高い保守性と拡張性を確保。

主な特徴:
- 5つの簡素メソッドによる処理分割
- pandas-ta直接呼び出しによる高効率
- 統一されたエラーハンドリング
- 後方互換性の確保（アダプター方式統合）

サポート指標数: 90個
- Momentum: ADX, AO, APO, AROON, BOP, CCI, CG, CMO, COPPOCK, CTI, FISHER, KST, MACD, MASSI, MOM, PGO, PPO, PSL, PSY, QQE, ROC, RSI, SQUEEZE, STC, STOCH, STOCHRSI, TRIX, TSI, UO, WILLR, WPR (31個)
- Trend: ALMA, AMAT, DEMA, DPO, EMA, FRAMA, HMA, KAMA, LINREG, LINREGSLOPE, RMA, SAR, SMA, SUPER_SMOOTHER, T3, TEMA, TRIMA, VORTEX, VWMA, WMA, ZLMA (21個)
- Volatility: ACCBANDS, ATR, BB, CHOP, DONCHIAN, KELTNER, NATR, RVI, SUPERTREND, UI, VHF (11個)
- Volume: AD, ADOSC, CMF, EFI, EOM, KVO, MFI, NVI, OBV, PVO, PVT, VWAP (12個)
- Price Transform: KAMA (1個)
- Original: ADAPTIVE_ENTROPY, CHAOS_FRACTAL_DIM, ELDER_RAY, FIBO_CYCLE, FRAMA, GRI, HARMONIC_RESONANCE, PRIME_OSC, QUANTUM_FLOW, SUPER_SMOOTHER, MCGINLEY_DYNAMIC, KAUFMAN_EFFICIENCY_RATIO, CHANDE_KROLL_STOP, TREND_INTENSITY_INDEX, CONNORS_RSI (15個)

※ 50個はpandas-ta直接対応、35個は独自実装によるアダプター方式

"""

import inspect
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from cachetools import LRUCache

from .config import IndicatorConfig, IndicatorResultType, indicator_registry
from .config.indicator_config import POSITIONAL_DATA_FUNCTIONS
from .data_validation import create_nan_result, validate_data_length_with_fallback

logger = logging.getLogger(__name__)


class TechnicalIndicatorService:
    """テクニカル指標統合サービス"""

    # 計算結果キャッシュ（クラスレベルで保持し、プロセス内で共有）
    # GA実行時に同一データの同一指標計算を回避する
    # メモリ消費を抑えるためサイズを調整 (5000 -> 1000)
    _calculation_cache = LRUCache(maxsize=1000)

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
        指定された指標を計算（簡素化版）

        5つの簡素ステップで処理:
        1. pandas-ta設定取得 → 2. 基本検証 → 3. パラメータ正規化
        4. pandas-ta直接呼び出し → 5. アダプター方式フォールバック

        Args:
            df: OHLCV価格データ
            indicator_type: 指標タイプ（SMA, RSI, MACDなど）
            params: パラメータ辞書（length, fast, slowなど）

        Returns:
            計算結果（numpy配列またはタプル）

        Raises:
            ValueError: サポートされていない指標タイプの場合
        """
        # キャッシュチェック
        try:
            # パラメータをハッシュ可能な形式に変換（辞書の順序を無視）
            # 値は念のため文字列化して比較（型違いによる重複計算を防ぐ意図だが、厳密な型区別はされない）
            # リストなどのミュータブルな値が含まれる場合に対応するため str() を使用
            cache_params = frozenset(sorted([(k, str(v)) for k, v in params.items()]))
            
            # DataFrameのIDをキーに含める（バックテスト中は同じオブジェクトが使い回される前提）
            cache_key = (indicator_type, cache_params, id(df))
            
            if cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key]
        except Exception:
            # キャッシュキー生成失敗時はキャッシュスキップ
            pass

        try:
            # 1. pandas-ta設定を取得
            config = self._get_config(indicator_type)
            result = None

            if config:
                # pandas-ta方式で処理
                # 3. パラメータ正規化（基本検証前に必要）
                normalized_params = self._normalize_params(params, config)

                # 2. 基本検証
                if not self._basic_validation(df, config, normalized_params):
                    result = self._create_nan_result(df, config)
                else:
                    # 4. pandas-ta直接呼び出し
                    raw_result = self._call_pandas_ta(df, config, normalized_params)
                    if raw_result is not None:
                        result = self._post_process(raw_result, config, df)

            # 5. アダプター方式にフォールバック（pandas-taにない場合または失敗した場合）
            if result is None:
                try:
                    config_obj = self._get_indicator_config(indicator_type)
                    if config_obj.adapter_function:
                        result = self._calculate_with_adapter(
                            df, indicator_type, params, config_obj
                        )
                    else:
                        raise ValueError(f"指標 {indicator_type} の実装が見つかりません")
                except ValueError:
                    # レジストリにもない場合
                    if config:
                        # pandas-ta設定はあるが呼び出し失敗の場合
                        result = self._create_nan_result(df, config)
                    else:
                        raise ValueError(f"指標 {indicator_type} の実装が見つかりません")
            
            # 結果をキャッシュに保存
            try:
                if result is not None:
                    # 前述のcache_keyロジックと同じ
                    cache_params = frozenset(sorted([(k, str(v)) for k, v in params.items()]))
                    cache_key = (indicator_type, cache_params, id(df))
                    self._calculation_cache[cache_key] = result
            except Exception:
                pass

            return result

        except Exception as e:
            logger.error(f"指標計算エラー {indicator_type}: {e}")
            raise

    def _get_config(self, indicator_type: str) -> Optional[Dict[str, Any]]:
        """設定を取得"""
        config = indicator_registry.get_indicator_config(indicator_type.upper())
        if config and config.pandas_function:
            # param_map から params 構造を構築
            # params = { "target_param_name": ["alias1", "alias2", ...] }
            params_mapping = {}
            if config.param_map:
                for alias, target in config.param_map.items():
                    # data mapping (e.g. close->data) excludes from params unless data is also a param?
                    # target None means ignore.
                    if target and target != "data":
                        if target not in params_mapping:
                            params_mapping[target] = []
                        if alias not in params_mapping[target]:
                            params_mapping[target].append(alias)

            # IndicatorConfigからPANDAS_TA_CONFIG形式に変換
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
        # データ長検証 - data_validation.pyの関数を直接使用
        is_valid, _ = validate_data_length_with_fallback(df, config["function"], params)
        if not is_valid:
            return False

        # カラム検証
        if config.get("multi_column", False):
            required_columns = config.get("data_columns", [])
            for req_col in required_columns:
                if not self._resolve_column_name(df, req_col):
                    return False
        else:
            if not self._resolve_column_name(df, config["data_column"]):
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
        """パラメータ正規化 - エイリアスマッピングとガード"""
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
                                f"パラメータ {param_name}={value} が最小値 {min_length} 未満のため調整します"
                            )
                            logger.debug(
                                f"TA_SMA パラメータ調整: {param_name} {value} -> {min_length}"
                            )
                            value = min_length
                    elif (
                        isinstance(min_length_func, (int, float))
                        and isinstance(value, (int, float))
                        and value < min_length_func
                    ):
                        logger.warning(
                            f"パラメータ {param_name}={value} が最小値 {min_length_func} 未満のため調整します"
                        )
                        logger.debug(
                            f"TA_SMA パラメータ調整: {param_name} {value} -> {min_length_func}"
                        )
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
                # 複数カラム処理
                required_columns = config.get("data_columns", [])
                positional_args = []
                for req_col in required_columns:
                    col_name = self._resolve_column_name(df, req_col)
                    if col_name is None:
                        logger.error(f"必須カラム '{req_col}' が存在しません")
                        return None
                    positional_args.append(df[col_name])

                return func(*positional_args, **params)
            else:
                # 単一カラム処理
                col_name = self._resolve_column_name(df, config["data_column"])
                if col_name is None:
                    logger.error(
                        f"TA_SMA エラー: 必須カラム '{config['data_column']}' が存在しません"
                    )
                    return None

                # データ長チェック
                if len(df) < params.get("length", 0):
                    logger.error(
                        f"TA_SMA エラー: データ長({len(df)})がlength({params.get('length', 'N/A')})未満"
                    )
                    return None

                return func(df[col_name], **params)

        except Exception as e:
            logger.error(
                f"pandas-ta呼び出し失敗: {config['function']}, エラー: {e}, パラメータ: {params}"
            )
            return None

    def _post_process(
        self, result: Any, config: Dict[str, Any], df: Optional[pd.DataFrame] = None
    ) -> Union[np.ndarray, tuple]:
        """後処理 - 戻り値の統一"""
        # pandas-taの一部関数(ichimokuなど)はDataFrameのタプルを返すため、最初の要素を使用
        if (
            isinstance(result, tuple)
            and len(result) > 0
            and isinstance(result[0], (pd.DataFrame, pd.Series))
        ):
            result = result[0]

        # NaN処理
        if isinstance(result, (pd.Series, pd.DataFrame)):
            # 入力DataFrameと結果の長さが異なる場合（一部の指標で行が削除される場合など）、indexに合わせて再編成
            if df is not None and len(result) != len(df):
                try:
                    result = result.reindex(df.index)
                except Exception:
                    # インデックスの互換性がない場合はそのまま
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
        else:  # multiple
            if isinstance(result, pd.DataFrame):
                # return_cols が指定されている場合は指定された列のみを返す
                if "return_cols" in config and config["return_cols"]:
                    selected_cols = []
                    for col in config["return_cols"]:
                        if col in result.columns:
                            selected_cols.append(result[col].values)
                        else:
                            # 部分一致で検索（例: BBL -> BBL_20_2.0）
                            matching_cols = [
                                c
                                for c in result.columns
                                if col in c or c.startswith(col + "_")
                            ]
                            if matching_cols:
                                selected_cols.append(result[matching_cols[0]].values)
                            else:
                                # NaN配列を追加
                                selected_cols.append(np.full(len(result), np.nan))
                    return tuple(selected_cols)
                else:
                    return tuple(
                        result.iloc[:, i].values for i in range(result.shape[1])
                    )
            else:
                return tuple(np.asarray(arr) for arr in result)

    def _prepare_adapter_data(
        self, df: pd.DataFrame, config: IndicatorConfig
    ) -> Dict[str, pd.Series]:
        """
        アダプター関数用のデータを準備

        Args:
            df: OHLCV価格データ
            config: 指標設定

        Returns:
            準備されたデータ辞書
        """
        required_data = {}

        # param_mapを処理（data_keyがNoneでない場合）
        if (
            hasattr(config, "param_map")
            and config.param_map is not None
            and isinstance(config.param_map, dict)
        ):
            for param_key, data_key in config.param_map.items():
                if data_key is None:
                    continue  # パラメータマッピング用なのでスキップ
                column_name = self._resolve_column_name(df, param_key)
                if column_name:
                    required_data[data_key] = df[column_name]

        # 通常のrequired_data処理
        for data_key in config.required_data:
            # param_mapで既に処理済みの場合はスキップ
            if (
                hasattr(config, "param_map")
                and config.param_map is not None
                and isinstance(config.param_map, dict)
                and data_key in config.param_map.keys()
            ):
                continue

            column_name = self._resolve_column_name(df, data_key)
            if column_name:
                required_data[data_key] = df[column_name]

        return required_data

    def _map_adapter_params(
        self,
        params: Dict[str, Any],
        config: IndicatorConfig,
        required_data: Dict[str, pd.Series],
    ) -> Tuple[Dict[str, Any], Dict[str, pd.Series]]:
        """
        アダプター関数のパラメータをマッピング

        Args:
            params: 入力パラメータ
            config: 指標設定
            required_data: 必要なデータ（変更される可能性あり）

        Returns:
            (マッピング済みパラメータ, 更新されたrequired_data)
        """
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
            if "data" in config.param_map and config.param_map["data"] == "close":
                # required_data から close データを data として使用
                if "close" in required_data:
                    converted_params["data"] = required_data["close"]
                    # close を required_data から削除して重複を防ぐ
                    del required_data["close"]
                converted_params = mapped_params

        return converted_params, required_data

    def _call_adapter_function(
        self,
        adapter_function: Any,
        all_args: Dict[str, Any],
        indicator_type: str,
        config: IndicatorConfig,
    ) -> Any:
        """
        アダプター関数を呼び出し

        Args:
            adapter_function: 呼び出すアダプター関数
            all_args: 関数に渡す全ての引数
            indicator_type: 指標タイプ
            config: 指標設定

        Returns:
            関数呼び出し結果
        """
        # 関数シグネチャを動的に検査して呼び出し方を決定
        sig = inspect.signature(adapter_function)
        valid_params = set(sig.parameters.keys())

        # 位置引数を必要とする関数
        if indicator_type.lower() in POSITIONAL_DATA_FUNCTIONS:
            # 位置引数を必要とする関数の場合
            positional_args = []
            keyword_args = {}

            # required_dataの順序で位置引数を構築
            for data_key in config.required_data:
                # param_map がある場合はマッピングされたキーを使用
                search_key = data_key
                if (
                    hasattr(config, "param_map")
                    and config.param_map
                    and data_key in config.param_map
                ):
                    search_key = config.param_map[data_key]

                if search_key in all_args:
                    positional_args.append(all_args[search_key])
                    del all_args[search_key]

            # 残りのパラメータをキーワード引数として渡す
            for k, v in all_args.items():
                if k not in config.parameters.keys():
                    if k in valid_params:
                        keyword_args[k] = v
                # config.parameters に含まれるパラメータは除外

            result = adapter_function(*positional_args, **keyword_args)
        else:
            # 通常のキーワード引数呼び出し
            # パラメータの順序を考慮してpositional argsとkeyword argsを分ける
            # sig は既に上部で取得済み
            positional_args = []
            keyword_args = {}

            # パラメータの順序で処理
            for param_name in sig.parameters:
                if param_name in all_args:
                    param = sig.parameters[param_name]
                    if param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        positional_args.append(all_args[param_name])
                    else:
                        keyword_args[param_name] = all_args[param_name]

            result = adapter_function(*positional_args, **keyword_args)

        # 結果の後処理
        if isinstance(result, pd.Series):
            return result.values
        elif (
            isinstance(result, pd.DataFrame)
            and config.result_type == IndicatorResultType.SINGLE
        ):
            return result.iloc[:, 0].values
        else:
            return result

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

        # データ準備を分離したメソッドに委譲
        required_data = self._prepare_adapter_data(df, config)

        # パラメータマッピングを分離したメソッドに委譲
        converted_params, required_data = self._map_adapter_params(
            params, config, required_data
        )

        # 引数の統合
        all_args = {**required_data, **converted_params}

        # 関数呼び出しを分離したメソッドに委譲
        return self._call_adapter_function(
            adapter_function, all_args, indicator_type, config
        )

    def _resolve_column_name(
        self, df: pd.DataFrame, data_key: Optional[str]
    ) -> Optional[str]:
        """
        データフレームから適切なカラム名を解決
        """
        if data_key is None:
            return None

        # 優先順位: 元の名前 > 大文字 > 小文字 > Capitalized
        candidates = [
            data_key,
            data_key.upper(),
            data_key.lower(),
            data_key.capitalize(),
        ]

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


