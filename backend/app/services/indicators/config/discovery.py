"""
動的インジケーター検出モジュール

pandas-ta および独自実装のテクニカル指標を動的にスキャンし、
設定(IndicatorConfig)を自動生成します。
これにより、手動でのマニフェスト管理を不要にします。
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Set, Type

import pandas as pd
import pandas_ta as ta
from pandas_ta.core import Strategy

from ..technical_indicators import (
    advanced_features,
    momentum,
    original,
    trend,
    volatility,
)
from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
)

logger = logging.getLogger(__name__)


class DynamicIndicatorDiscovery:
    """インジケーター動的検出クラス"""

    # 除外する関数名や内部メソッド
    EXCLUDED_FUNCTIONS = {
        "cdl_pattern", "cdl_z", "ha", "tsignals", "above", "below",
        "cross", "cross_value", "fibonacci", "verify_series",
        "progress_bar", "imports", "category", "utils",
        "strategy", "log", "datetime", "version"
    }

    # データカラムとみなす引数名（小文字）
    DATA_ARGUMENTS = {
        "open", "high", "low", "close", "volume", 
        "open_", "high_", "low_", "close_", "volume_",
        "openinterest", "fundingrate"
    }

    # パラメータ名に基づくデフォルト範囲ルール (Heuristics)
    PARAMETER_RULES = {
        "length": {"min": 2, "max": 200, "default": 14},
        "period": {"min": 2, "max": 200, "default": 14},
        "fast": {"min": 2, "max": 50, "default": 12},
        "slow": {"min": 10, "max": 200, "default": 26},
        "signal": {"min": 2, "max": 50, "default": 9},
        "std": {"min": 0.1, "max": 5.0, "default": 2.0},
        "factor": {"min": 0.1, "max": 5.0, "default": 1.0},
        "scalar": {"min": 1.0, "max": 200.0, "default": 100.0},
        "multiplier": {"min": 0.5, "max": 5.0, "default": 2.0},
        "drift": {"min": 1, "max": 20, "default": 1},
        "offset": {"min": -50, "max": 50, "default": 0},
        "k": {"min": 2, "max": 100, "default": 14},
        "d": {"min": 2, "max": 100, "default": 3},
        "smooth_k": {"min": 1, "max": 30, "default": 3},
        "rsi_length": {"min": 2, "max": 100, "default": 14},
        "stoch_length": {"min": 2, "max": 100, "default": 14},
    }

    @classmethod
    def discover_all(cls) -> List[IndicatorConfig]:
        """全ての指標を検出して設定リストを返す"""
        configs = []
        discovered_names = set()

        # 1. pandas-ta の指標を検出
        ta_configs = cls._discover_pandas_ta()
        for config in ta_configs:
            if config.indicator_name not in discovered_names:
                cls._apply_special_overrides(config)
                configs.append(config)
                discovered_names.add(config.indicator_name)

        # 2. 独自実装の指標を検出
        from ..technical_indicators.volume import VolumeIndicators
        custom_modules = [
            advanced_features.AdvancedFeatures,
            momentum.MomentumIndicators,
            original.OriginalIndicators,
            trend.TrendIndicators,
            volatility.VolatilityIndicators,
            VolumeIndicators
        ]

        for module_class in custom_modules:
            custom_configs = cls._discover_custom_class(module_class)
            for config in custom_configs:
                cls._apply_special_overrides(config)
                configs = [c for c in configs if c.indicator_name != config.indicator_name]
                configs.append(config)
                discovered_names.add(config.indicator_name)

        logger.info(f"合計 {len(configs)} 個のインジケーターを動的検出しました")
        return configs

    @classmethod
    def _apply_special_overrides(cls, config: IndicatorConfig):
        """特定の指標に対する特別な設定の上書き"""
        name = config.indicator_name.upper()
        
        if name == "STOCH":
            config.min_length_func = lambda p: (p.get("k") or p.get("k_length") or 14) + (p.get("d") or p.get("d_length") or 3) + (p.get("smooth_k") or 3)
            config.return_cols = ["STOCHk_14_3_3", "STOCHd_14_3_3"]
        elif name == "MACD":
            config.min_length_func = lambda p: (p.get("slow") or 26) + (p.get("signal") or 9)
            config.return_cols = ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]
        elif name == "BBANDS":
            config.return_cols = ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]
        elif name == "RSI":
            config.min_length_func = lambda p: (p.get("length") or p.get("period") or 14) + 1
        elif name == "ATR":
            config.min_length_func = lambda p: (p.get("length") or 14) + 1
        elif name == "SUPERTREND":
            config.return_cols = ["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"]
        elif name == "VORTEX":
            config.return_cols = ["VTXP_14", "VTXM_14"]
        elif name == "FISHER":
            config.return_cols = ["FISHERT_9_1", "FISHERTs_9_1"]
        elif name == "KST":
            config.return_cols = ["KST_10_15_20_30_10_10_10_15", "KSTs_9"]
        elif name == "PVO":
            config.return_cols = ["PVO_12_26_9", "PVOs_12_26_9", "PVOh_12_26_9"]
        elif name == "TSI":
            config.return_cols = ["TSI_13_25_13", "TSIs_13_25_13"]

    @classmethod
    def _discover_pandas_ta(cls) -> List[IndicatorConfig]:
        """pandas-taの関数をスキャン"""
        configs = []
        
        # カテゴリマッピング (pandas-ta category -> system category)
        category_map = {
            "momentum": "momentum",
            "trend": "trend",
            "volatility": "volatility",
            "volume": "volume",
            "overlap": "trend",
            "cycles": "cycle",
            "statistics": "statistic",
            "performance": "statistic"
        }
        
        discovered_functions = set()

        try:
            # 主要な pandas-ta 指標の明示的なリスト（スキャン失敗時の保険）
            core_ta_funcs = [
                "rsi", "sma", "ema", "macd", "bbands", "atr", "adx", "cci", "mfi", "obv",
                "roc", "trix", "stoch", "willr", "uo", "ao", "apo", "bop", "chop", "cmf",
                "cmo", "coppock", "dema", "donchian", "efi", "eom", "fisher", "hma", "kama",
                "kst", "pgo", "ppo", "psl", "pvo", "pvt", "qqe", "rma", "rvi", "sar",
                "stc", "supertrend", "t3", "tema", "trima", "vortex", "vwap", "vwma", "zlma",
                "ad", "adosc", "stochrsi", "massi", "nvi", "squeeze", "tsi", "cg", "mom"
            ]

            # 1. 既知の主要関数を優先的にチェック
            for name in core_ta_funcs:
                func = getattr(ta, name, None)
                if func and callable(func):
                    if name in discovered_functions:
                        continue
                    
                    # カテゴリの取得 (モジュール名から抽出)
                    # pandas_ta.momentum.rsi -> momentum
                    module_parts = func.__module__.split(".")
                    ta_cat = "technical"
                    if len(module_parts) >= 2 and module_parts[0] == "pandas_ta":
                        ta_cat = module_parts[1]
                    
                    sys_cat = category_map.get(ta_cat, "technical")
                    
                    config = cls._analyze_function(name, func, sys_cat)
                    if config:
                        config.pandas_function = name
                        configs.append(config)
                        discovered_functions.add(name)

            # 2. 残りの全属性をスキャン
            for name in dir(ta):
                if name.startswith("_") or name in cls.EXCLUDED_FUNCTIONS or name in discovered_functions:
                    continue
                
                func = getattr(ta, name, None)
                if func and callable(func):
                    # 同様にカテゴリ抽出
                    module_parts = func.__module__.split(".")
                    if len(module_parts) >= 2 and module_parts[0] == "pandas_ta":
                        ta_cat = module_parts[1]
                        if ta_cat in category_map:
                            config = cls._analyze_function(name, func, category_map[ta_cat])
                            if config:
                                config.pandas_function = name
                                configs.append(config)
                                discovered_functions.add(name)

        except Exception as e:
            logger.error(f"pandas-ta スキャンエラー: {e}")

        return configs

    @classmethod
    def _discover_custom_class(cls, klass: Type) -> List[IndicatorConfig]:
        """独自実装クラスをスキャン"""
        configs = []
        for name, method in inspect.getmembers(klass, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            
            # staticmethodの場合はラップを解除
            if isinstance(method, staticmethod):
                method = method.__func__

            config = cls._analyze_function(name, method, category="custom")
            if config:
                # 独自実装なので adapter_function を設定
                # 文字列パスとして保存 ("module.Class.method")
                module_path = klass.__module__
                class_name = klass.__name__
                config.adapter_function = getattr(klass, name) # 関数オブジェクトそのものを渡すか、パスにするか
                # IndicatorConfigは adapter_function に callable を期待する設計に見えるが
                # Manifestでは文字列パスだった。Orchestratorは callable を受け取るように見える。
                # ここでは callable を直接設定する。
                
                # Orchestrator は config.adapter_function を直接呼び出す
                configs.append(config)

        return configs

    @classmethod
    def _analyze_function(cls, name: str, func: Any, category: str) -> Optional[IndicatorConfig]:
        """関数を解析してIndicatorConfigを生成"""
        try:
            sig = inspect.signature(func)
            
            # 1. 必要なデータカラムとパラメータを分離
            required_data = []
            parameters = {}
            default_values = {}
            param_map = {}
            
            for param_name, param in sig.parameters.items():
                if param_name in ["kwargs", "args"]:
                    continue

                # データ引数かパラメータか
                param_lower = param_name.lower()
                if param_lower in cls.DATA_ARGUMENTS or param_lower in ["data", "series"]:
                    # データ引数
                    source = param_lower
                    if param_lower in ["data", "series"]:
                        source = "close" # デフォルトはclose
                    
                    required_data.append(source)
                    # 引数名 -> ソース名のマッピングを記録（アダプター用）
                    param_map[source] = param_name
                else:
                    # パラメータ
                    default_val = param.default
                    if default_val == inspect.Parameter.empty:
                        default_val = None
                    
                    default_values[param_name] = default_val
                    
                    # GA用パラメータ設定を作成
                    param_config = cls._create_parameter_config(param_name, default_val)
                    if param_config:
                        parameters[param_name] = param_config

            # 2. 戻り値情報の推測
            result_type = IndicatorResultType.SINGLE
            
            # 一般的な複合指標名 (pandas-taでDataFrameを返すもの)
            complex_indicators = {
                "bbands", "macd", "stoch", "aroon", "supertrend", "accbands", "donchian", "kc",
                "ppo", "trix", "tsi", "vortex", "fisher", "kst", "adosc", "stochrsi", "squeeze"
            }
            if name.lower() in complex_indicators:
                result_type = IndicatorResultType.COMPLEX

            # 3. スケールタイプの推測
            scale_type = IndicatorScaleType.PRICE_ABSOLUTE
            if name.lower() in ["rsi", "stoch", "cci", "mfi", "adx", "chop", "aroon"]:
                scale_type = IndicatorScaleType.OSCILLATOR_0_100
            elif name.lower() in ["macd", "mom", "roc", "trix"]:
                scale_type = IndicatorScaleType.MOMENTUM_ZERO_CENTERED

            # 4. Configオブジェクト生成
            config = IndicatorConfig(
                indicator_name=name.upper(),
                category=category,
                required_data=required_data,
                param_map=param_map,
                parameters=parameters,
                default_values=default_values,
                result_type=result_type,
                returns="multiple" if result_type == IndicatorResultType.COMPLEX else "single",
                scale_type=scale_type,
                multi_column=len(required_data) > 1,
                data_column=required_data[0].capitalize() if required_data else "Close",
                data_columns=[col.capitalize() for col in required_data] if len(required_data) > 1 else None
            )
            
            return config

        except Exception as e:
            logger.warning(f"関数解析エラー {name}: {e}")
            return None

    @classmethod
    def _create_parameter_config(cls, name: str, default_val: Any) -> Optional[ParameterConfig]:
        """パラメータ名から設定を生成"""
        # ルールベースでマッチング
        rule = None
        for key in cls.PARAMETER_RULES:
            if key in name.lower():
                rule = cls.PARAMETER_RULES[key]
                break
        
        if rule:
            # デフォルト値が関数のデフォルトにあれば優先
            val = default_val if default_val is not None and isinstance(default_val, (int, float)) else rule["default"]
            
            return ParameterConfig(
                name=name,
                default_value=val,
                min_value=rule["min"],
                max_value=rule["max"]
            )
        
        # ルールにない数値パラメータ
        if isinstance(default_val, (int, float)) and not isinstance(default_val, bool):
             return ParameterConfig(
                name=name,
                default_value=default_val,
                min_value=default_val * 0.1,
                max_value=default_val * 5.0
            )
            
        return None
