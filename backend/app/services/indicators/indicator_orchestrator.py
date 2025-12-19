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


# 指標名のエイリアスマッピング（外部からも参照可能）
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

    def clear_cache(self) -> None:
        """計算キャッシュをクリアする"""
        self._calculation_cache.clear()
        logger.info("Indicator calculation cache cleared.")

    def _resolve_indicator_name(self, indicator_type: str) -> str:
        """指標名のエイリアスを解決"""
        upper_name = indicator_type.upper()
        return INDICATOR_ALIASES.get(upper_name, upper_name)

    def _get_config(self, indicator_type: str) -> Optional[IndicatorConfig]:
        """
        指定された指標の pandas-ta 設定を取得
        """
        resolved_name = self._resolve_indicator_name(indicator_type)
        return self.registry.get_indicator_config(resolved_name)

    def calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str, params: Dict[str, Any]
    ) -> Union[np.ndarray, pd.Series, tuple, tuple[pd.Series, ...]]:
        """
        OHLCVデータから指定されたテクニカル指標を計算。
        """
        # 指標名の解決
        indicator_type = self._resolve_indicator_name(indicator_type)
        
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
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )
                except ValueError:
                    # レジストリにもない場合
                    if config:
                        # pandas-ta設定はあるが呼び出し失敗の場合
                        result = self._create_nan_result(df, config)
                    else:
                        raise ValueError(
                            f"指標 {indicator_type} の実装が見つかりません"
                        )

            # 結果をキャッシュに保存
            try:
                if result is not None:
                    # 前述のcache_keyロジックと同じ
                    cache_params = frozenset(
                        sorted([(k, str(v)) for k, v in params.items()])
                    )
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
        resolved_name = self._resolve_indicator_name(indicator_type)
        config = self.registry.get_indicator_config(resolved_name)
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
            else:
                # param_mapがない場合はparametersから直接マッピングを作成
                for param_name in config.parameters.keys():
                    params_mapping[param_name] = [param_name]

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
        """
        入力パラメータを pandas-ta 等が期待する内部名称と値に正規化

        エイリアスの解決（例: 'length' -> 'n'）、デフォルト値の補完、
        およびデータ長等に基づいた最小値ガードの適用を行います。

        Args:
            params: ユーザー入力パラメータ
            config: 指標設定辞書

        Returns:
            正規化済みパラメータ辞書
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
                
                # pandas-taの関数シグネチャを確認してキーワード引数を準備
                # 例: chop(high, low, close, length=14, ...)
                # 多くの関数は high, low, close, volume をキーワード引数としても受け取る
                ta_args = {}
                positional_args = []
                
                for req_col in required_columns:
                    col_name = self._resolve_column_name(df, req_col)
                    if col_name is None:
                        logger.error(f"必須カラム '{req_col}' が存在しません")
                        return None
                    
                    # カラム名に基づいて引数名を決定
                    col_lower = req_col.lower()
                    if col_lower in ["open", "high", "low", "close", "volume"]:
                        # chopなどはキーワード引数での指定が必要な場合がある
                        ta_args[col_lower] = df[col_name]
                    else:
                        positional_args.append(df[col_name])

                # 引数の統合（paramsよりも価格データを優先、またはマージ）
                combined_params = {**params, **ta_args}
                
                return func(*positional_args, **combined_params)
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
                                if col in c or col.lower() in c.lower()
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
        # 利用可能な標準的なデータソース
        standard_keys = ["open", "high", "low", "close", "volume"]
        required_data = {}
        
        # 1. 明示的に要求されているデータを収集
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
                    # その他のカスタムカラム
                    col_name = self._resolve_column_name(df, key)
                    if col_name:
                        required_data[key] = df[col_name]
        
        # 2. 補足：基本的なデータが不足している場合の自動追加
        if not required_data:
            # デフォルトとしてCloseと全体を渡せるようにしておく
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
        """
        アダプター関数のパラメータをマッピング
        """
        # パラメータ正規化
        converted_params = config.normalize_params(params)
        mapped_params = converted_params.copy()

        # param_map を使用してデータ引数をマッピング
        if (
            hasattr(config, "param_map")
            and config.param_map is not None
            and isinstance(config.param_map, dict)
        ):
            for source_name, arg_name in config.param_map.items():
                # source_name (close, high等) が required_data にあれば、arg_name として mapped_params に追加
                if source_name in required_data:
                    mapped_params[arg_name] = required_data[source_name]
                    # 元の required_data から削除して重複を防ぐ
                    del required_data[source_name]

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
        """
        sig = inspect.signature(adapter_function)
        positional_args = []
        keyword_args = {}
        
        # 利用可能なデータを準備
        series_data = {k: v for k, v in all_args.items() if isinstance(v, (pd.Series, pd.DataFrame))}
        scalar_params = {k: v for k, v in all_args.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
        
        assigned_params = {}

        # シグネチャに基づいて引数を割り当て
        for param_name, param in sig.parameters.items():
            param_lower = param_name.lower()
            val = None
            
            # 1. 直接一致
            if param_name in all_args:
                val = all_args[param_name]
            # 2. データ引数の曖昧解決
            elif param_lower in ["data", "df", "ohlcv", "close", "open", "high", "low", "volume"]:
                # 名前に基づく解決
                target_key = param_lower.rstrip("_")
                val = series_data.get(target_key) or series_data.get("data")
            # 3. スカラー引数の曖昧解決
            elif param_lower in ["length", "period", "window", "n"]:
                for k in ["length", "period", "window", "n"]:
                    if k in scalar_params:
                        val = scalar_params[k]
                        break
            
            if val is not None:
                assigned_params[param_name] = val

        # まだ割り当てられていないスカラー引数があれば追加（キーワード引数として）
        for k, v in all_args.items():
            if k not in assigned_params and k in sig.parameters:
                assigned_params[k] = v

        try:
            # 全てをキーワード引数として渡す（ほとんどのテクニカル指標関数で安全）
            # シグネチャに含まれるキーのみを抽出して渡す
            valid_params = {k: v for k, v in assigned_params.items() if k in sig.parameters}
            result = adapter_function(**valid_params)
        except TypeError:
            # キーワード引数がサポートされていない場合（稀なケース）は位置引数にフォールバック
            try:
                result = adapter_function(*assigned_params.values())
            except Exception as e:
                logger.error(f"{indicator_type} 計算エラー (位置引数フォールバック): {e}")
                result = None
        except Exception as e:
            logger.error(f"{indicator_type} 計算エラー: {e}")
            result = None
        except Exception as e:
            logger.error(f"{indicator_type} 計算エラー: {e}")
            result = None

        # 結果がNoneの場合のフォールバック
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
            # NaN処理を追加
            return result.bfill().fillna(0).values
        elif isinstance(result, pd.DataFrame):
            # NaN処理を追加
            result = result.bfill().fillna(0)
            if config.result_type == IndicatorResultType.SINGLE:
                return result.iloc[:, 0].values
            else:
                return tuple(result[col].values for col in result.columns)
        elif isinstance(result, tuple):
            # タプル内の各要素をnumpy配列に変換し、NaN処理を行う
            return tuple(
                (arr.bfill().fillna(0).values if isinstance(arr, pd.Series) else 
                 (arr.bfill().fillna(0).values if isinstance(arr, pd.DataFrame) else 
                  np.nan_to_num(np.asarray(arr))))
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
        # また、末尾のアンダースコアを除去したものも候補に入れる (例: open_ -> open)
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
