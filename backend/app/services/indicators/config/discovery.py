"""
動的インジケーター検出モジュール

pandas-ta および独自実装のテクニカル指標を動的にスキャンし、
設定(IndicatorConfig)を自動生成します。
これにより、手動でのマニフェスト管理を不要にします。
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List, Optional, Type, cast

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
)
from .pandas_ta_introspection import (
    _build_sample_ohlcv_frame,
    _run_indicator_on_sample_frame,
    calculate_min_length,
    extract_default_parameters,
    get_all_pandas_ta_indicators,
    get_indicator_category,
    get_return_column_names,
    is_multi_column_indicator,
)

logger = logging.getLogger(__name__)


class DynamicIndicatorDiscovery:
    """インジケーター動的検出クラス"""

    _EXCLUDED_DISCOVERY_NAMES = {
        "above",
        "above_value",
        "below",
        "below_value",
        "cross",
        "cross_value",
        "df_dates",
        "df_error_analysis",
        "df_month_to_date",
        "df_quarter_to_date",
        "df_year_to_date",
        "downside_deviation",
        "is_datetime_ordered",
        "jensens_alpha",
        "linear_regression",
        "mtd",
        "qtd",
        "total_time",
        "to_utc",
        "ytd",
        "verify_series",
        "short_run",
        "long_run",
        "recent_maximum_index",
        "recent_minimum_index",
        "signals",
        "tsignals",
        "xsignals",
    }

    # プロジェクト固有のデータカラム（pandas-ta標準以外）
    _PROJECT_DATA_COLUMNS = {
        "openinterest",
        "fundingrate",
        "open_interest",
        "funding_rate",
        "market_cap",
    }

    # pandas-ta標準のデータ引数（動的検出用）
    # fast, slow, signal はパラメータと競合するためここには入れない
    _STANDARD_DATA_ARGS = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_",
        "trend",
        "series",
    }

    _SAMPLE_PROBE_DATA_COLUMNS = {"open", "open_", "high", "low", "close", "volume"}

    _ALIAS_OVERRIDES = {
        "BBANDS": ["BB", "BOLLINGER"],
        "MOM": ["MOMENTUM"],
        "EMA": ["EXP_MA"],
        "SMA": ["SIMPLE_MA"],
    }

    _SPECIAL_CONFIG_OVERRIDES = {
        "DEMARKER": {
            "scale_type": IndicatorScaleType.OSCILLATOR_0_100,
            "use_default_thresholds": True,
        },
        "RMI": {
            "scale_type": IndicatorScaleType.OSCILLATOR_0_100,
            "use_default_thresholds": True,
        },
        "MMI": {
            "scale_type": IndicatorScaleType.OSCILLATOR_0_100,
            "use_default_thresholds": True,
        },
        "TTF": {
            "scale_type": IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
            "thresholds": {
                "aggressive": {"long_gt": 80, "short_lt": -80},
                "normal": {"long_gt": 100, "short_lt": -100},
                "conservative": {"long_gt": 120, "short_lt": -120},
            },
            "min_length_func": lambda p: max(int(p.get("length", 15)) * 2, 2),
        },
        "RWI": {
            "result_type": IndicatorResultType.COMPLEX,
            "returns": "multiple",
            "return_cols": ["RWI_HIGH", "RWI_LOW"],
            "output_names": ["RWI_HIGH", "RWI_LOW"],
            "default_output": "RWI_HIGH",
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "thresholds": {
                "aggressive": {"long_gt": 0.8, "short_lt": 0.8},
                "normal": {"long_gt": 1.0, "short_lt": 1.0},
                "conservative": {"long_gt": 1.2, "short_lt": 1.2},
            },
            "min_length_func": lambda p: max(int(p.get("length", 14)) + 1, 2),
        },
        "PFE": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "CRYPTO_LEVERAGE_INDEX": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "LIQUIDATION_CASCADE_SCORE": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "TREND_QUALITY": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "OI_WEIGHTED_FUNDING_RATE": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "OI_PRICE_CONFIRMATION": {
            "scale_type": IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            "use_default_thresholds": True,
        },
        "SQUEEZE_PROBABILITY": {
            "scale_type": IndicatorScaleType.FUNDING_RATE,
            "thresholds": {
                "aggressive": {"long_gt": 0.0, "short_lt": 0.001},
                "normal": {"long_gt": 0.0, "short_lt": 0.01},
                "conservative": {"long_gt": 0.0, "short_lt": 0.05},
            },
        },
        "WHALE_DIVERGENCE": {
            "scale_type": IndicatorScaleType.PRICE_RATIO,
            "thresholds": {
                "aggressive": {"long_gt": 1.02, "short_lt": 0.98},
                "normal": {"long_gt": 1.05, "short_lt": 0.95},
                "conservative": {"long_gt": 1.1, "short_lt": 0.9},
            },
        },
    }

    @classmethod
    def _calculate_param_range(
        cls, param_name: str, default_val: Any
    ) -> tuple[float, float]:
        """パラメータの min/max 範囲を動的に計算

        Args:
            param_name: パラメータ名
            default_val: デフォルト値

        Returns:
            (min_value, max_value) のタプル
        """
        if not isinstance(default_val, (int, float)) or isinstance(default_val, bool):
            return (1, 100)  # フォールバック

        name_lower = param_name.lower()

        # 特殊なパラメータタイプごとの処理
        if "distribution_offset" in name_lower:
            return (0.0, 1.0)
        if "std" in name_lower or "factor" in name_lower:
            # 標準偏差・係数: 0.1 ~ 5.0倍
            return (max(0.1, default_val * 0.1), default_val * 3.0)
        elif "multiplier" in name_lower:
            # 乗数: 0.5 ~ 5.0
            return (0.5, 5.0)
        elif name_lower == "offset":
            # オフセット: 負の値も許可
            return (-50, 50)
        elif "drift" in name_lower:
            # ドリフト: 1 ~ 20
            return (1, 20)
        elif isinstance(default_val, float) and (
            not default_val.is_integer()
            or abs(default_val) <= 1
            or any(
                token in name_lower
                for token in (
                    "scalar",
                    "alpha",
                    "beta",
                    "gamma",
                    "coef",
                    "quantile",
                    "prob",
                    "weight",
                    "ratio",
                )
            )
        ):
            if default_val > 0:
                min_val = max(0.01, default_val * 0.5)
                max_val = max(default_val * 2.0, default_val + 0.1)
                if default_val <= 1:
                    max_val = min(1.0, max_val)
                return (min_val, max_val)

            if default_val < 0:
                min_val = min(default_val * 2.0, default_val - 0.1)
                max_val = min(-0.01, default_val * 0.5)
                return (min_val, max_val)

            return (-1.0, 1.0)
        else:
            # 一般的な期間パラメータ: 最小2、最大は5倍
            min_val = max(2, int(default_val * 0.2))
            max_val = max(int(default_val * 5), 50)
            return (min_val, max_val)

    @classmethod
    def _is_indicator_function(cls, func: Any) -> bool:
        """指標関数かユーティリティ関数かを判定

        シグネチャに価格データ引数（close/high/low/open/volume）が
        含まれていれば指標関数とみなす
        """
        try:
            sig = inspect.signature(func)
            for param in sig.parameters.values():
                if cls._is_data_argument(param):
                    return True
            return False
        except (ValueError, TypeError):
            return False

    @classmethod
    def _is_data_argument(cls, param: inspect.Parameter) -> bool:
        """引数がデータ引数かどうかを判定"""
        # 1. 型ヒントによる判定 (最優先)
        if param.annotation != inspect.Parameter.empty:
            type_str = str(param.annotation)
            # 文字列でSeriesやDataFrameを含むかチェック
            if "Series" in type_str or "DataFrame" in type_str:
                return True
            # 数値型ならFalse
            if "int" in type_str or "float" in type_str:
                return False

        # 2. デフォルト値による判定
        if param.default != inspect.Parameter.empty and param.default is not None:
            # デフォルト値が数値ならデータ引数ではない
            if isinstance(param.default, (int, float)):
                return False

        # 3. 名前による判定
        param_lower = param.name.lower()
        if (
            param_lower in cls._STANDARD_DATA_ARGS
            or param_lower in cls._PROJECT_DATA_COLUMNS
            or param_lower in ["data", "series"]
        ):
            return True

        return False

    # pandas-ta カテゴリ -> システムカテゴリのマッピング
    # overlap は移動平均系などを含むため、独自カテゴリとして維持
    _TA_CATEGORY_MAP = {
        "cycles": "cycle",
        "statistics": "statistic",
        "performance": "statistic",
    }

    @classmethod
    def _guess_scale_type(
        cls, name: str, func: Any, default_params: dict
    ) -> IndicatorScaleType:
        """サンプルデータを用いて指標のスケールタイプを推測"""
        try:
            # 実行（FutureWarningを抑制）
            result = _run_indicator_on_sample_frame(
                func,
                rows=200,
                default_params=default_params,
                walk_close=True,
                with_datetime_index=True,
            )

            # 結果の取り出し
            if isinstance(result, pd.DataFrame):
                val = result.iloc[:, 0].dropna()
            elif isinstance(result, pd.Series):
                val = result.dropna()
            else:
                return IndicatorScaleType.PRICE_ABSOLUTE

            if val.empty:
                return IndicatorScaleType.PRICE_ABSOLUTE

            v_min, v_max = val.min(), val.max()
            mean = val.mean()

            # 判定ロジック
            # 1. 0-100 オシレーター
            if 0 <= v_min <= 20 and 80 <= v_max <= 100:
                return IndicatorScaleType.OSCILLATOR_0_100

            # 2. 0 中心モメンタム
            if v_min < 0 < v_max and abs(mean) < (v_max - v_min) * 0.5:
                # 0付近で変動している
                return IndicatorScaleType.MOMENTUM_ZERO_CENTERED

            # 3. 価格絶対値
            if v_min > 50 and abs(mean - 100) < 50:
                return IndicatorScaleType.PRICE_ABSOLUTE

            return IndicatorScaleType.PRICE_RATIO

        except Exception:
            # 判定失敗時は安全なデフォルトを返す
            return IndicatorScaleType.PRICE_ABSOLUTE

    @classmethod
    def _supports_timeseries_output(
        cls,
        indicator_name: str,
        func: Any,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """エンジンで扱える時系列出力を返す関数だけを対象にする。"""
        expected_length = 120
        sample_frame = _build_sample_ohlcv_frame(
            expected_length,
            walk_close=True,
            with_datetime_index=True,
        )
        try:
            result = _run_indicator_on_sample_frame(
                func,
                rows=expected_length,
                default_params=default_params,
                walk_close=True,
                with_datetime_index=True,
            )
        except Exception as exc:
            logger.debug(
                "時系列互換性チェックに失敗: %s (%s)", indicator_name, exc
            )
            return False

        return cls._is_timeseries_compatible_result(result, sample_frame.index)

    @classmethod
    def _is_timeseries_compatible_result(cls, result: Any, expected_index: Any) -> bool:
        """出力がサンプル入力長と整合する時系列結果かどうかを判定する。"""
        if isinstance(result, pd.Series):
            return cls._has_compatible_timeseries_index(result.index, expected_index)

        if isinstance(result, pd.DataFrame):
            return result.shape[1] > 0 and cls._has_compatible_timeseries_index(
                result.index, expected_index
            )

        if isinstance(result, np.ndarray):
            return cls._has_compatible_array_shape(result, expected_index)

        if isinstance(result, tuple):
            return bool(result) and all(
                cls._is_timeseries_compatible_result(item, expected_index)
                for item in result
            )

        return False

    @classmethod
    def _has_compatible_timeseries_index(cls, result_index: Any, expected_index: Any) -> bool:
        """出力 index が入力時系列と整合しているかを判定する。"""
        if len(result_index) == 0:
            return False

        if len(result_index) == len(expected_index):
            return True

        try:
            if not getattr(result_index, "is_monotonic_increasing", True):
                return False
            return bool(result_index.isin(expected_index).all())
        except Exception:
            return False

    @classmethod
    def _has_compatible_array_shape(cls, result: np.ndarray, expected_index: Any) -> bool:
        """index を持たない ndarray は入力長と一致する場合のみ許容する。"""
        if result.ndim == 0 or result.size == 0:
            return False

        return result.shape[0] == len(expected_index)

    @classmethod
    def _can_probe_with_sample(cls, required_data: List[str]) -> bool:
        """標準 OHLCV サンプルだけで関数を実行できるかを判定する。"""
        return all(
            data_key.lower() in cls._SAMPLE_PROBE_DATA_COLUMNS
            for data_key in required_data
        )

    @classmethod
    def _probe_indicator_output(
        cls,
        indicator_name: str,
        func: Any,
        required_data: List[str],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """標準 OHLCV だけで評価できる指標はサンプル実行結果を返す。"""
        if not cls._can_probe_with_sample(required_data):
            return None

        try:
            return _run_indicator_on_sample_frame(
                func,
                rows=120,
                default_params=default_params,
                walk_close=True,
                with_datetime_index=True,
            )
        except Exception as exc:
            logger.debug("出力プローブに失敗: %s (%s)", indicator_name, exc)
            return None

    @classmethod
    def _infer_result_type_from_output(
        cls, result: Any
    ) -> Optional[IndicatorResultType]:
        """サンプル出力から result_type を推定する。"""
        if isinstance(result, pd.DataFrame):
            return (
                IndicatorResultType.COMPLEX
                if result.shape[1] > 1
                else IndicatorResultType.SINGLE
            )

        if isinstance(result, tuple):
            return (
                IndicatorResultType.COMPLEX
                if len(result) > 1
                else IndicatorResultType.SINGLE
            )

        if isinstance(result, pd.Series):
            return IndicatorResultType.SINGLE

        return None

    @classmethod
    def _should_skip_indicator_name(cls, indicator_name: str) -> bool:
        """discovery 対象外の関数名かどうかを軽量に判定する。"""
        name_lower = indicator_name.lower()
        if "cdl" in name_lower or "candle" in name_lower:
            return True
        return name_lower in cls._EXCLUDED_DISCOVERY_NAMES

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

        # 2. 独自実装の指標を自動検出
        custom_modules = []
        original_modules = []
        from .. import technical_indicators

        # technical_indicators パッケージ内の全モジュールをスキャン
        # pandas_ta サブパッケージへ移した互換ラッパーは除外する。
        skipped_wrapper_modules = {
            "momentum",
            "overlap",
            "trend",
            "volatility",
            "volume",
        }
        for loader, module_name, is_pkg in pkgutil.iter_modules(
            technical_indicators.__path__
        ):
            if module_name in skipped_wrapper_modules:
                continue
            try:
                # 相対インポートでモジュールをロード
                full_module_name = f"..technical_indicators.{module_name}"
                module = importlib.import_module(full_module_name, __package__)

                # モジュール内の全クラスをチェック
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 指標クラスを検出:
                    # - 末尾が Indicators
                    # - 高度特徴量クラス AdvancedFeatures
                    if (
                        name.endswith("Indicators") and name != "Indicators"
                    ) or name == "AdvancedFeatures":
                        if module_name == "original":
                            original_modules.append(obj)
                        else:
                            custom_modules.append(obj)
            except Exception as e:
                logger.warning(f"モジュール {module_name} のスキャンに失敗: {e}")

        for module_class in custom_modules + original_modules:
            custom_configs = cls._discover_custom_class(module_class)
            for config in custom_configs:
                cls._apply_special_overrides(config)
                configs = [
                    c for c in configs if c.indicator_name != config.indicator_name
                ]
                configs.append(config)
                discovered_names.add(config.indicator_name)

        logger.info(f"合計 {len(configs)} 個のインジケーターを動的検出しました")
        return configs

    @classmethod
    def _apply_special_overrides(cls, config: IndicatorConfig):
        """特定の指標に対する特別な設定の上書き

        min_length_func と return_cols は pandas_ta_introspection を使用して
        動的に取得します。エイリアスもルールベースで自動付与します。
        """
        name_upper = config.indicator_name.upper()
        name_lower = config.indicator_name.lower()

        # 1. 最小データ長と戻り値カラムの動的取得
        if config.pandas_function or hasattr(ta, name_lower):
            config.min_length_func = lambda p, ind=name_lower: calculate_min_length(
                ind, p
            )  # type: ignore[misc]

            if not config.return_cols:
                cols = get_return_column_names(name_lower)
                if cols:
                    config.return_cols = cols

        # 2. ルールベースのエイリアス自動付与
        aliases = set(cls._ALIAS_OVERRIDES.get(name_upper, []))

        if aliases:
            if config.aliases:
                aliases.update(config.aliases)
            config.aliases = list(aliases)

        overrides = cls._SPECIAL_CONFIG_OVERRIDES.get(name_upper, {})
        for attr_name in (
            "result_type",
            "returns",
            "return_cols",
            "output_names",
            "default_output",
            "scale_type",
            "thresholds",
            "min_length_func",
        ):
            if attr_name in overrides:
                setattr(config, attr_name, overrides[attr_name])

        if overrides.get("use_default_thresholds"):
            config.thresholds = config._get_default_thresholds()

        # 3. 特殊なパラメータ制約の付与
        if name_upper == "FRAMA":
            if "length" in config.parameters:
                config.parameters["length"].even_only = True
            elif "len" in config.parameters:
                config.parameters["len"].even_only = True

    @classmethod
    def _discover_pandas_ta(cls) -> List[IndicatorConfig]:
        """pandas-taの関数をスキャン"""
        configs = []

        try:
            for name in get_all_pandas_ta_indicators():
                func = getattr(ta, name, None)
                if func is None or not callable(func):
                    continue

                if not cls._is_indicator_function(func):
                    continue

                if cls._should_skip_indicator_name(name):
                    continue

                default_params = extract_default_parameters(name)
                if not cls._supports_timeseries_output(
                    name, func, default_params=default_params
                ):
                    continue

                # カテゴリ抽出 (イントロスペクションを使用)
                ta_cat = get_indicator_category(name) or "technical"
                # 例外マップになければそのままの名前を使用
                sys_cat = cls._TA_CATEGORY_MAP.get(ta_cat, ta_cat)

                config = cls._analyze_function(name, func, sys_cat)
                if config:
                    config.pandas_function = name
                    configs.append(config)

        except Exception as e:
            logger.error(f"pandas-ta スキャンエラー: {e}")

        return configs

    @classmethod
    def _discover_custom_class(cls, klass: Type) -> List[IndicatorConfig]:
        """独自実装クラスをスキャン"""
        configs = []
        for name, method in inspect.getmembers(klass, predicate=inspect.isfunction):
            if name.startswith("_") or name.startswith("calculate_"):
                continue
            if not cls._is_indicator_function(method):
                continue

            config = cls._analyze_function(name, method, category="custom")
            if config:
                if cls._can_probe_with_sample(config.required_data) and not cls._supports_timeseries_output(
                    name,
                    getattr(klass, name),
                    default_params=config.default_values,
                ):
                    continue

                # 独自実装なので adapter_function を設定
                # Orchestrator は config.adapter_function を直接呼び出すため、
                # callable を直接設定する
                config.adapter_function = getattr(klass, name)
                configs.append(config)

        return configs

    @classmethod
    def _analyze_function(
        cls, name: str, func: Any, category: str
    ) -> Optional[IndicatorConfig]:
        """関数を解析してIndicatorConfigを生成"""
        try:
            name_lower = name.lower()

            # 除外対象のフィルタリング
            if cls._should_skip_indicator_name(name_lower):
                return None

            sig = inspect.signature(func)

            # 1. 必要なデータカラムとパラメータを分離
            required_data = []
            parameters = {}
            default_values = {}
            param_map = {}
            inferred_defaults = extract_default_parameters(name_lower)

            for param_name, param in sig.parameters.items():
                if param_name in ["kwargs", "args"]:
                    continue

                # データ引数かパラメータかを動的に判定
                param_lower = param_name.lower()
                if cls._is_data_argument(param):
                    # データ引数
                    source = param_lower
                    if param_lower in ["data", "series"]:
                        source = "close"  # デフォルトはclose

                    required_data.append(source)
                    # 引数名 -> ソース名のマッピングを記録（アダプター用）
                    param_map[source] = param_name  # type: ignore[assignment]
                else:
                    # パラメータ
                    default_val = param.default
                    if default_val == inspect.Parameter.empty:
                        default_val = inferred_defaults.get(param_name)

                    default_values[param_name] = default_val

                    # GA用パラメータ設定を作成
                    param_config = cls._create_parameter_config(param_name, default_val)
                    if param_config:
                        parameters[param_name] = param_config

            # 2. 戻り値情報の推測
            result_type = IndicatorResultType.SINGLE
            sample_result = cls._probe_indicator_output(
                name,
                func,
                required_data,
                default_params=default_values,
            )
            inferred_result_type = cls._infer_result_type_from_output(sample_result)
            if inferred_result_type is not None:
                result_type = inferred_result_type
            elif is_multi_column_indicator(name.lower()):
                result_type = IndicatorResultType.COMPLEX

            # 3. スケールタイプの推測 (動的判定)
            scale_type = cls._guess_scale_type(name, func, default_values)

            # 4. Configオブジェクト生成
            config = IndicatorConfig(
                indicator_name=name.upper(),
                category=category,
                required_data=required_data,
                param_map=cast(Dict[str, Optional[str]], param_map),
                parameters=parameters,
                default_values=default_values,
                result_type=result_type,
                scale_type=scale_type,
            )

            return config

        except Exception as e:
            logger.warning(f"関数解析エラー {name}: {e}")
            return None

    @classmethod
    def _create_parameter_config(
        cls, name: str, default_val: Any
    ) -> Optional[ParameterConfig]:
        """パラメータ名から設定を生成（動的範囲計算）"""
        # 数値パラメータのみ対象
        if not isinstance(default_val, (int, float)) or isinstance(default_val, bool):
            return None

        # 動的に範囲を計算
        min_val, max_val = cls._calculate_param_range(name, default_val)

        return ParameterConfig(
            name=name,
            default_value=default_val,
            min_value=min_val,
            max_value=max_val,
        )
