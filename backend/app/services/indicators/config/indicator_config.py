"""
インジケーター設定管理モジュール

動的に検出されたインジケーター設定を管理するためのクラスを提供します。
オートストラテジー（GA）との連携を前提とした設計になっています。
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeGuard, cast

from app.types import SerializablePrimitive
from app.utils.registry import Registry

logger = logging.getLogger(__name__)


class IndicatorScaleType(Enum):
    """指標のスケールタイプ"""

    OSCILLATOR_0_100 = "oscillator_0_100"  # 0-100スケール（RSI, STOCH等）
    OSCILLATOR_PLUS_MINUS_100 = "oscillator_plus_minus_100"  # ±100スケール（CCI等）
    MOMENTUM_ZERO_CENTERED = "momentum_zero_centered"  # ゼロ近辺変動（TRIX, PPO等）
    PRICE_RATIO = "price_ratio"  # 価格比率（SMA, EMA等）
    FUNDING_RATE = "funding_rate"  # ファンディングレート
    OPEN_INTEREST = "open_interest"  # オープンインタレスト
    VOLUME = "volume"  # 出来高
    PRICE_ABSOLUTE = "price_absolute"  # 絶対価格


class IndicatorResultType(Enum):
    """インジケーター結果タイプ"""

    SINGLE = "single"  # 単一値（例：RSI、SMA）
    COMPLEX = "complex"  # 複数値（例：MACD、Bollinger Bands）


@dataclass
class ParameterConfig:
    """パラメータ設定

    インジケーターの各パラメータの制約と探索範囲を定義します。
    GAによるパラメータ最適化で使用されます。
    """

    name: str  # パラメータ名（例：period, fast_period）
    default_value: SerializablePrimitive  # デフォルト値
    min_value: int | float | None = None  # 最小値
    max_value: int | float | None = None  # 最大値
    description: str | None = None  # パラメータの説明
    even_only: bool = False  # 偶数のみを許可するか（FRAMA等用）
    # 探索プリセット: 用途に応じた探索範囲（例: short_term, mid_term, long_term）
    presets: dict[str, tuple] | None = None

    def get_range_for_preset(self, preset_name: str) -> tuple:
        """
        指定されたプリセット名に対応する探索範囲を取得

        Args:
            preset_name: プリセット名（例: "short_term", "mid_term", "long_term"）

        Returns:
            (min_value, max_value) のタプル
        """
        if self.presets and preset_name in self.presets:
            return self.presets[preset_name]

        # フォールバック：デフォルトの min/max 範囲を使用
        return (self.min_value, self.max_value)

    def _apply_even_constraint(self, value: int | float) -> int | float:
        """even_only 制約を満たすように値を調整する。"""
        if not self.even_only or int(value) % 2 == 0:
            return value

        upper_bound = self.max_value if self.max_value is not None else float("inf")
        return value + 1 if value < upper_bound else value - 1

    def _normalize_numeric_value(self, value: int | float) -> int | float:
        """min/max と even_only をまとめて適用する。"""
        normalized = value

        if self.min_value is not None and normalized < self.min_value:
            normalized = self.min_value
        if self.max_value is not None and normalized > self.max_value:
            normalized = self.max_value

        return self._apply_even_constraint(normalized)


def _is_numeric(value: Any) -> TypeGuard[int | float]:
    """値が数値（bool を除く）かどうか。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _constraint_param_names(
    constraint: dict[str, Any],
) -> tuple[str, str] | None:
    """制約の param1/param2 名を取得する（文字列でない場合は None）。"""
    param1_name = constraint.get("param1")
    param2_name = constraint.get("param2")
    if isinstance(param1_name, str) and isinstance(param2_name, str):
        return param1_name, param2_name
    return None


def _constraint_min_diff(constraint: dict[str, Any]) -> int | float:
    """制約の min_diff を数値で取得する（未指定・不正値は 0）。"""
    min_diff = constraint.get("min_diff", 0)
    if isinstance(min_diff, (int, float)) and not isinstance(min_diff, bool):
        return min_diff
    return 0


def _has_int_default(param_config: ParameterConfig) -> bool:
    """パラメータのデフォルト値が整数型かどうか（GAの整数パラメータ判定）。"""
    default_value = param_config.default_value
    return isinstance(default_value, int) and not isinstance(default_value, bool)


@dataclass
class IndicatorConfig:
    """インジケーター設定

    テクニカル指標の計算に必要なすべての設定を保持します。
    pandas-taとの連携設定および独自実装のアダプター設定を含みます。
    """

    indicator_name: str
    adapter_function: Callable | None = None
    required_data: list[str] = field(default_factory=list)
    result_type: IndicatorResultType = IndicatorResultType.SINGLE
    scale_type: IndicatorScaleType = IndicatorScaleType.PRICE_RATIO
    category: str | None = None
    output_names: list[str] | None = None
    default_output: str | None = None
    aliases: list[str] | None = None
    param_map: dict[str, str | None] = field(default_factory=dict)
    parameters: dict[str, ParameterConfig] = field(default_factory=dict)

    # pandas-ta連携設定
    pandas_function: str | None = None
    data_column: str | None = None
    data_columns: list[str] | None = None
    returns: str = "single"
    return_cols: list[str] | None = None
    multi_column: bool = False
    default_values: dict[str, Any] = field(default_factory=dict)
    min_length_func: Callable[[dict[str, Any]], int] | None = None

    # パラメータ依存関係制約
    parameter_constraints: list[dict[str, Any]] | None = None

    # GA用閾値設定
    thresholds: dict[str, Any] = field(default_factory=dict)

    # 絶対的最小データ長
    absolute_min_length: int = 1

    def __post_init__(self) -> None:
        """後処理でパラメータおよび派生属性を構築"""
        if not self.parameters:
            self.parameters = self._build_parameters_from_defaults()

        # thresholdsのデフォルト設定
        if not self.thresholds:
            self.thresholds = self._get_default_thresholds()

        # 派生属性の判定と自動設定
        # returns (single / multiple)
        if hasattr(self, "returns") and (not self.returns or self.returns == "single"):
            if self.result_type == IndicatorResultType.COMPLEX:
                self.returns = "multiple"
            else:
                self.returns = "single"

        # multi_column
        self.multi_column = len(self.required_data) > 1

        # data_column / data_columns
        if not self.data_column and self.required_data:
            self.data_column = self.required_data[0].capitalize()

        if not self.data_columns and len(self.required_data) > 1:
            self.data_columns = [col.capitalize() for col in self.required_data]

    def validate_constraints(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        パラメータ依存関係制約を検証

        Args:
            params: 検証するパラメータ辞書

        Returns:
            (is_valid, error_messages) のタプル
        """
        if not self.parameter_constraints:
            return True, []

        errors = []

        for constraint in self.parameter_constraints:
            constraint_type = constraint.get("type")
            param1_name = constraint.get("param1")
            param2_name = constraint.get("param2")

            if param1_name not in params or param2_name not in params:
                continue

            param1_value = params[param1_name]
            param2_value = params[param2_name]

            if not isinstance(param1_value, (int, float)) or not isinstance(
                param2_value, (int, float)
            ):
                continue

            if constraint_type == "less_than":
                if param1_value >= param2_value:
                    errors.append(
                        f"{param1_name}({param1_value}) は "
                        f"{param2_name}({param2_value}) より小さくなければなりません"
                    )

            elif constraint_type == "greater_than":
                if param1_value <= param2_value:
                    errors.append(
                        f"{param1_name}({param1_value}) は "
                        f"{param2_name}({param2_value}) より大きくなければなりません"
                    )

            elif constraint_type == "min_difference":
                min_diff = constraint.get("min_diff", 0)
                actual_diff = param1_value - param2_value
                if actual_diff < min_diff:
                    errors.append(
                        f"{param1_name}({param1_value}) と {param2_name}({param2_value}) の差は "
                        f"最低 {min_diff} 必要ですが、{actual_diff} です"
                    )

        return len(errors) == 0, errors

    def repair_parameters(
        self, params: dict[str, Any], *, clamp_ranges: bool = False
    ) -> int:
        """
        パラメータの整数性と依存関係制約を修復する（GA生成・変異・交叉後用）。

        GAは各パラメータを独立に生成・変異するため、fast >= slow のような
        依存関係違反や小数化した期間パラメータが生じ得る。本メソッドは

        1. ParameterConfig.default_value が int のパラメータを整数へ丸める
        2. 依存関係制約に違反する値ペアをスワップ（遺伝子材料を温存）し、
           各パラメータの探索範囲へクランプ、それでも違反する場合は近傍値へ調整

        を in-place で適用する。

        Args:
            params: 修復対象のパラメータ辞書（in-place で更新される）
            clamp_ranges: True の場合、制約対象外のパラメータも探索範囲
                （min/max）へクランプする。プリセット探索範囲は min/max と
                一致しない場合があるため、デフォルトは False。

        Returns:
            値を変更したパラメータの数。
        """
        before = dict(params)

        # 1) 整数パラメータの丸め込み（と clamp_ranges 時の範囲クランプ）
        for name, param_config in self.parameters.items():
            value = params.get(name)
            if not _is_numeric(value):
                continue
            repaired_value = value
            if _has_int_default(param_config):
                repaired_value = int(repaired_value)
            if clamp_ranges:
                repaired_value = param_config._normalize_numeric_value(repaired_value)
            params[name] = repaired_value

        # 2) 依存関係制約の修復（連鎖制約は複数パスで収束させる）
        if self.parameter_constraints:
            constraints = [c for c in self.parameter_constraints if isinstance(c, dict)]
            for _ in range(len(constraints) + 1):
                self._repair_dependency_constraints_pass(params, constraints)
                is_valid, _ = self.validate_constraints(params)
                if is_valid:
                    break

        return sum(
            1 for name, old_value in before.items() if old_value != params.get(name)
        )

    def _repair_dependency_constraints_pass(
        self, params: dict[str, Any], constraints: list[dict[str, Any]]
    ) -> None:
        """依存関係制約を1パス分修復する。連鎖（fast<medium<slow等）は
        個別修復が他制約を壊し得るため、呼び出し側で複数パス回す。"""
        for constraint in constraints:
            constraint_type = constraint.get("type")
            names = _constraint_param_names(constraint)
            if names is None:
                continue
            param1_name, param2_name = names

            if param1_name not in params or param2_name not in params:
                continue

            value1 = params[param1_name]
            value2 = params[param2_name]
            if not _is_numeric(value1) or not _is_numeric(value2):
                continue

            if self._is_pair_satisfied(constraint, value1, value2):
                continue

            if constraint_type in ("less_than", "greater_than"):
                # スワップで遺伝子材料を温存し、各探索範囲へクランプする
                params[param1_name], params[param2_name] = value2, value1
                for name in (param1_name, param2_name):
                    param_config = self.parameters.get(name)
                    if param_config is not None:
                        params[name] = param_config._normalize_numeric_value(
                            params[name]
                        )
                if self._is_pair_satisfied(
                    constraint, params[param1_name], params[param2_name]
                ):
                    continue

            # スワップで解消できない場合（同値等）は近傍値へ調整する
            self._nudge_constraint_params(constraint, params)

    @staticmethod
    def _is_pair_satisfied(
        constraint: dict[str, Any], value1: float, value2: float
    ) -> bool:
        """単一の依存関係制約を値ペアに対して評価する（未知の型は満た扱い）。"""
        constraint_type = constraint.get("type")
        if constraint_type == "less_than":
            return value1 < value2
        if constraint_type == "greater_than":
            return value1 > value2
        if constraint_type == "min_difference":
            return (value1 - value2) >= _constraint_min_diff(constraint)
        return True

    def _nudge_constraint_params(
        self, constraint: dict[str, Any], params: dict[str, Any]
    ) -> None:
        """スワップで解消できない違反を、範囲内の近傍値へ調整する。"""
        constraint_type = constraint.get("type")
        names = _constraint_param_names(constraint)
        if names is None:
            return
        param1_name, param2_name = names

        value1 = params.get(param1_name)
        value2 = params.get(param2_name)
        if not _is_numeric(value1) or not _is_numeric(value2):
            return

        if constraint_type == "less_than":
            step = self._integer_step(param2_name)
            raised = value1 + step
            if self._is_within_range(param2_name, raised):
                params[param2_name] = raised
            elif self._is_within_range(param1_name, value2 - step):
                params[param1_name] = value2 - step
        elif constraint_type == "greater_than":
            step = self._integer_step(param1_name)
            lowered = value2 - step
            if self._is_within_range(param1_name, lowered):
                params[param1_name] = lowered
            elif self._is_within_range(param2_name, value1 + step):
                params[param2_name] = value1 + step
        elif constraint_type == "min_difference":
            needed = _constraint_min_diff(constraint) - (value1 - value2)
            raised = value1 + needed
            if self._is_within_range(param1_name, raised):
                params[param1_name] = raised
            elif self._is_within_range(param2_name, value2 - needed):
                params[param2_name] = value2 - needed

    def _integer_step(self, param_name: str) -> int | float:
        """整数パラメータには 1、実数パラメータには 1.0 を調整幅とする。"""
        param_config = self.parameters.get(param_name)
        if param_config is not None and _has_int_default(param_config):
            return 1
        return 1.0

    def _is_within_range(self, param_name: str, value: int | float) -> bool:
        """値がパラメータの探索範囲内か（範囲未定義なら無制限とみなす）。"""
        param_config = self.parameters.get(param_name)
        if param_config is None:
            return True
        min_value = param_config.min_value
        max_value = param_config.max_value
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    def _get_default_thresholds(self) -> dict[str, Any]:
        """スケールタイプに基づくデフォルトの閾値設定を生成"""
        if self.scale_type == IndicatorScaleType.OSCILLATOR_0_100:
            return {
                "aggressive": {"long_lt": 35, "short_gt": 65},
                "normal": {"long_lt": 30, "short_gt": 70},
                "conservative": {"long_lt": 25, "short_gt": 75},
            }
        elif self.scale_type == IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100:
            return {"normal": {"long_lt": -80, "short_gt": 80}}
        elif self.scale_type == IndicatorScaleType.MOMENTUM_ZERO_CENTERED:
            return {"normal": {"long_gt": 0, "short_lt": 0}}
        elif self.scale_type == IndicatorScaleType.PRICE_RATIO:
            # SMA/EMAなどの場合はCloseとの比較（ConditionEvaluatorで処理されるため閾値自体は不要な場合も多い）
            return {}
        return {}

    def _build_parameters_from_defaults(self) -> dict[str, ParameterConfig]:
        """デフォルト値からパラメータを構築"""
        params: dict[str, ParameterConfig] = {}
        required_data_keys = {name.lower() for name in self.required_data}

        # param_mapからパラメータ名を取得
        param_names = set()
        if self.param_map:
            for param_name in self.param_map:
                if (
                    param_name
                    and param_name != "data"
                    and param_name.lower() not in required_data_keys
                ):
                    param_names.add(param_name)

        # default_valuesからもパラメータ名を取得
        for param_name in self.default_values:
            param_names.add(param_name)

        # 各パラメータのParameterConfigを作成
        for param_name in param_names:
            default_value = self.default_values.get(param_name, 14)

            # length/periodパラメータの特別処理
            if any(word in param_name for word in ["length", "period"]):
                min_val: int | float | None = 2
                max_val: int | float | None = 200
            else:
                min_val = None
                max_val = None

            param_config = ParameterConfig(
                name=param_name,
                default_value=default_value,
                min_value=min_val,
                max_value=max_val,
            )
            params[param_name] = param_config

        return params

    def get_parameter_ranges(self) -> dict[str, dict[str, Any]]:
        """パラメータ範囲を取得（GA用）"""
        ranges = {}

        for param_name, param_config in self.parameters.items():
            ranges[param_name] = {
                "min": param_config.min_value,
                "max": param_config.max_value,
                "default": param_config.default_value,
            }

        return ranges

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """パラメータ正規化

        エイリアスを解決し、範囲内に値を制限します。
        """
        normalized = params.copy()

        # param_mapに基づいてエイリアスマッピング
        if self.param_map:
            for param_key, mapped_param in self.param_map.items():
                if param_key in params and mapped_param and mapped_param != param_key:
                    value = params[param_key]
                    normalized[mapped_param] = value
                    if param_key in normalized:
                        del normalized[param_key]

        # パラメータ範囲チェック
        for param_name, param_config in self.parameters.items():
            if param_name in normalized and isinstance(param_config, ParameterConfig):
                value = normalized[param_name]
                if isinstance(value, (int, float)):
                    normalized[param_name] = param_config._normalize_numeric_value(
                        value
                    )

        return normalized

    def generate_random_parameters(self, preset: str | None = None) -> dict[str, Any]:
        """
        ランダムなパラメータを生成（GA用）

        Args:
            preset: 探索プリセット名（例：short_term, mid_term, long_term）
                None の場合はデフォルト範囲を使用

        Returns:
            生成されたパラメータ辞書
        """
        import random

        params = {}
        for param_name, param_config in self.parameters.items():
            if not isinstance(param_config, ParameterConfig):
                continue  # type: ignore[unreachable]

            # プリセットが指定されている場合はプリセット範囲を使用
            if preset:
                min_val, max_val = param_config.get_range_for_preset(preset)
            else:
                min_val = param_config.min_value
                max_val = param_config.max_value

            if min_val is not None and max_val is not None:
                # リストの場合は最初の要素を使用
                if isinstance(min_val, list):
                    min_val = min_val[0]
                if isinstance(max_val, list):
                    max_val = max_val[0]

                if isinstance(param_config.default_value, int):
                    val = random.randint(int(min_val), int(max_val))
                    params[param_name] = param_config._apply_even_constraint(val)
                else:
                    params[param_name] = random.uniform(float(min_val), float(max_val))
            else:
                params[param_name] = param_config.default_value  # type: ignore[assignment]

        # 各パラメータが独立に抽選されるため、fast >= slow のような依存関係違反を修復する
        self.repair_parameters(params)

        return params


class IndicatorConfigRegistry(Registry[str, IndicatorConfig]):
    """インジケーター設定レジストリ

    すべてのインジケーター設定を一元管理し、
    名前やエイリアスによる検索を提供します。
    """

    def __init__(self) -> None:
        super().__init__()
        self._aliases: dict[str, IndicatorConfig] = {}
        self._initialized: bool = False
        self._auto_initialize_on_access: bool = True

    @staticmethod
    def _normalize_key(name: str) -> str:
        """検索・登録用のキーを正規化する。"""
        return name.upper()

    def _ensure_ready(self) -> None:
        """グローバルレジストリの遅延初期化を吸収する。"""
        if (
            self is indicator_registry
            and not self._initialized
            and self._auto_initialize_on_access
        ):
            self.ensure_initialized()

    def register(self, config: IndicatorConfig) -> None:
        """設定を登録"""
        primary_key = self._normalize_key(config.indicator_name)
        self.set(primary_key, config)

        # 同一主指標の再登録時に古い alias が残らないようにする
        self._aliases = {
            alias_key: alias_config
            for alias_key, alias_config in self._aliases.items()
            if self._normalize_key(alias_config.indicator_name) != primary_key
        }

        # エイリアスは lookup 専用マップに登録
        if config.aliases:
            for alias in config.aliases:
                alias_key = self._normalize_key(alias)
                if alias_key != primary_key:
                    self._aliases[alias_key] = config

    def get_indicator_config(self, indicator_name: str) -> IndicatorConfig | None:
        """設定を取得"""
        self._ensure_ready()
        normalized_name = self._normalize_key(indicator_name)
        return self._items.get(normalized_name) or self._aliases.get(normalized_name)

    def list_indicators(self) -> list[str]:
        """登録されているインジケーター名のリストを取得"""
        self._ensure_ready()
        return list(self._items.keys())

    def get_all_indicators(self) -> dict[str, IndicatorConfig]:
        """登録済みインジケーターの辞書を取得"""
        self._ensure_ready()
        return dict(self._items)

    def generate_parameters_for_indicator(
        self, indicator_type: str, preset: str | None = None
    ) -> dict[str, Any]:
        """
        指標タイプに応じたランダムパラメータを生成（GA用）

        Args:
            indicator_type: 指標タイプ（例：RSI, MACD）
            preset: 探索プリセット名

        Returns:
            生成されたパラメータ辞書
        """
        config = self.get_indicator_config(indicator_type)
        if config:
            return config.generate_random_parameters(preset)
        else:
            logger.warning(f"指標 {indicator_type} の設定が見つかりません")
            return {}

    def ensure_initialized(self) -> None:
        """レジストリが初期化されていることを確認"""
        if not self._initialized:
            _initialize_registry(self)
            self._initialized = True
        self._auto_initialize_on_access = True


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()

# 初期化キャッシュ（プロセス間共有）
_indicator_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def get_cached_indicators() -> list[str]:
    """キャッシュされたインジケーターリストを取得"""
    with _cache_lock:
        if "indicators" not in _indicator_cache:
            _indicator_cache["indicators"] = indicator_registry.list_indicators()
        return cast(list[str], _indicator_cache["indicators"])


def _initialize_registry(registry: IndicatorConfigRegistry) -> None:
    """レジストリを初期化（動的検出）"""
    try:
        from .discovery import DynamicIndicatorDiscovery

        indicators = DynamicIndicatorDiscovery.discover_all()
        for config in indicators:
            registry.register(config)

        logger.info(f"インジケーター初期化完了: {len(indicators)} 個登録")
        registry._initialized = True

    except ImportError as e:
        logger.warning(f"DynamicIndicatorDiscoveryのインポートに失敗しました: {e}")
    except Exception as e:
        logger.error(f"インジケーター初期化エラー: {e}")


def initialize_all_indicators() -> None:
    """全インジケーターの設定を初期化"""
    indicator_registry.ensure_initialized()
