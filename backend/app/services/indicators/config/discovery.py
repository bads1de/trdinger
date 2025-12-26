"""
動的インジケーター検出モジュール

pandas-ta および独自実装のテクニカル指標を動的にスキャンし、
設定(IndicatorConfig)を自動生成します。
これにより、手動でのマニフェスト管理を不要にします。
"""

import inspect
import logging
from typing import Any, List, Optional, Type

import pandas_ta as ta
import numpy as np
import pandas as pd
import pkgutil
import importlib


from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
)
from .pandas_ta_introspection import (
    calculate_min_length,
    get_indicator_category,
    get_return_column_names,
    is_multi_column_indicator,
)

logger = logging.getLogger(__name__)


class DynamicIndicatorDiscovery:
    """インジケーター動的検出クラス"""

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
        if "std" in name_lower or "factor" in name_lower:
            # 標準偏差・係数: 0.1 ~ 5.0倍
            return (max(0.1, default_val * 0.1), default_val * 3.0)
        elif "multiplier" in name_lower:
            # 乗数: 0.5 ~ 5.0
            return (0.5, 5.0)
        elif "offset" in name_lower:
            # オフセット: 負の値も許可
            return (-50, 50)
        elif "drift" in name_lower:
            # ドリフト: 1 ~ 20
            return (1, 20)
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
            # 判定用のサンプルデータを生成 (200本)
            np.random.seed(42)
            close = np.random.randn(200).cumsum() + 100
            df = pd.DataFrame(
                {
                    "open": close - 1,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": np.random.randint(100, 1000, 200),
                }
            )

            # パラメータの準備（データ引数を注入）
            sig = inspect.signature(func)
            call_params = {}
            for p_name in sig.parameters:
                p_lower = p_name.lower()
                if p_lower == "close":
                    call_params[p_name] = df["close"]
                elif p_lower == "high":
                    call_params[p_name] = df["high"]
                elif p_lower == "low":
                    call_params[p_name] = df["low"]
                elif p_lower == "open":
                    call_params[p_name] = df["open"]
                elif p_lower == "volume":
                    call_params[p_name] = df["volume"]
                elif p_lower in ["data", "series"]:
                    call_params[p_name] = df["close"]
                elif p_name in default_params:
                    call_params[p_name] = default_params[p_name]

            # 実行
            result = func(**call_params)

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
        from .. import technical_indicators

        # technical_indicators パッケージ内の全モジュールをスキャン
        for loader, module_name, is_pkg in pkgutil.iter_modules(
            technical_indicators.__path__
        ):
            try:
                # 相対インポートでモジュールをロード
                full_module_name = f"..technical_indicators.{module_name}"
                module = importlib.import_module(full_module_name, __package__)

                # モジュール内の全クラスをチェック
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 名前の末尾が Indicators で、かつ Indicators 自体ではないクラス
                    if name.endswith("Indicators") and name != "Indicators":
                        custom_modules.append(obj)
            except Exception as e:
                logger.warning(f"モジュール {module_name} のスキャンに失敗: {e}")

        for module_class in custom_modules:
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
            )

            if not config.return_cols:
                cols = get_return_column_names(name_lower)
                if cols:
                    config.return_cols = cols

        # 2. ルールベースのエイリアス自動付与
        aliases = set()
        if name_upper == "BBANDS":
            aliases.update(["BB", "BOLLINGER"])
        elif name_upper == "MOM":
            aliases.add("MOMENTUM")
        elif name_upper == "EMA":
            aliases.add("EXP_MA")
        elif name_upper == "SMA":
            aliases.add("SIMPLE_MA")

        if aliases:
            if config.aliases:
                aliases.update(config.aliases)
            config.aliases = list(aliases)

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

        discovered_functions = set()

        try:
            # pandas-ta の全指標を動的にスキャン
            for name in dir(ta):
                if name.startswith("_") or name in discovered_functions:
                    continue

                func = getattr(ta, name, None)
                if func is None or not callable(func):
                    continue

                # 動的にユーティリティ関数を判定
                if not cls._is_indicator_function(func):
                    continue

                # カテゴリ抽出 (イントロスペクションを使用)
                ta_cat = get_indicator_category(name) or "technical"
                # 例外マップになければそのままの名前を使用
                sys_cat = cls._TA_CATEGORY_MAP.get(ta_cat, ta_cat)

                config = cls._analyze_function(name, func, sys_cat)
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
            # 1. キャンドル系 (TA-Lib依存やオートストラテジー不適合)
            if "cdl" in name_lower or "candle" in name_lower:
                return None

            # 2. シグナル生成・複雑な戻り値の関数
            if name_lower in ["tsignals", "xsignals"]:
                return None

            # 3. ユーティリティ関数およびデータ整合性チェック関数の除外
            utility_names = {
                "above", "above_value", "below", "below_value", "cross", "cross_value",
                "df_dates", "df_error_analysis", "df_month_to_date", "df_quarter_to_date",
                "df_year_to_date", "downside_deviation", "is_datetime_ordered",
                "jensens_alpha", "linear_regression", "mtd", "qtd", "total_time",
                "to_utc", "ytd", "verify_series", "short_run", "long_run"
            }
            if name_lower in utility_names:
                return None

            sig = inspect.signature(func)

            # 1. 必要なデータカラムとパラメータを分離
            required_data = []
            parameters = {}
            default_values = {}
            param_map = {}

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
            if is_multi_column_indicator(name.lower()):
                result_type = IndicatorResultType.COMPLEX

            # 3. スケールタイプの推測 (動的判定)
            scale_type = cls._guess_scale_type(name, func, default_values)

            # 4. Configオブジェクト生成
            config = IndicatorConfig(
                indicator_name=name.upper(),
                category=category,
                required_data=required_data,
                param_map=param_map,
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
