"""
インジケーター設定管理モジュール

動的に検出されたインジケーター設定を管理するためのクラスを提供します。
オートストラテジー（GA）との連携を前提とした設計になっています。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

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
    default_value: Union[int, float, str, bool]  # デフォルト値
    min_value: Optional[Union[int, float]] = None  # 最小値
    max_value: Optional[Union[int, float]] = None  # 最大値
    description: Optional[str] = None  # パラメータの説明
    # 探索プリセット: 用途に応じた探索範囲（例: short_term, mid_term, long_term）
    presets: Optional[Dict[str, tuple]] = None

    def validate_value(self, value: Any) -> bool:
        """与えられた値がこのパラメータの制約（範囲など）を満たすか検証する"""
        if not isinstance(value, (int, float)):
            # 数値でない場合は検証スキップ
            return True

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        return True

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


@dataclass
class IndicatorConfig:
    """インジケーター設定

    テクニカル指標の計算に必要なすべての設定を保持します。
    pandas-taとの連携設定および独自実装のアダプター設定を含みます。
    """

    indicator_name: str
    adapter_function: Optional[Callable] = None
    required_data: List[str] = field(default_factory=list)
    result_type: IndicatorResultType = IndicatorResultType.SINGLE
    scale_type: IndicatorScaleType = IndicatorScaleType.PRICE_RATIO
    category: Optional[str] = None
    output_names: Optional[List[str]] = None
    default_output: Optional[str] = None
    aliases: Optional[List[str]] = None
    param_map: Dict[str, Optional[str]] = field(default_factory=dict)
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)

    # pandas-ta連携設定
    pandas_function: Optional[str] = None
    data_column: Optional[str] = None
    data_columns: Optional[List[str]] = None
    returns: str = "single"
    return_cols: Optional[List[str]] = None
    multi_column: bool = False
    default_values: Dict[str, Any] = field(default_factory=dict)
    min_length_func: Optional[Callable[[Dict[str, Any]], int]] = None

    # パラメータ依存関係制約
    parameter_constraints: Optional[List[Dict[str, Any]]] = None

    # 絶対的最小データ長
    absolute_min_length: int = 1

    def __post_init__(self):
        """後処理でパラメータおよび派生属性を構築"""
        if not self.parameters:
            self.parameters = self._build_parameters_from_defaults()

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

    def validate_constraints(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
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

    def _build_parameters_from_defaults(self) -> Dict[str, ParameterConfig]:
        """デフォルト値からパラメータを構築"""
        params: Dict[str, ParameterConfig] = {}

        # param_mapからパラメータ名を取得
        param_names = set()
        if self.param_map:
            for param_name in self.param_map.keys():
                if param_name and param_name != "data":
                    param_names.add(param_name)

        # default_valuesからもパラメータ名を取得
        for param_name in self.default_values.keys():
            param_names.add(param_name)

        # 各パラメータのParameterConfigを作成
        for param_name in param_names:
            default_value = self.default_values.get(param_name, 14)

            # length/periodパラメータの特別処理
            if any(word in param_name for word in ["length", "period"]):
                min_val: Optional[Union[int, float]] = 2
                max_val: Optional[Union[int, float]] = 200
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

    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """パラメータ範囲を取得（GA用）"""
        ranges = {}

        for param_name, param_config in self.parameters.items():
            if isinstance(param_config, ParameterConfig):
                ranges[param_name] = {
                    "min": param_config.min_value,
                    "max": param_config.max_value,
                    "default": param_config.default_value,
                }
            else:
                # 後方互換性のための処理
                default_value = self.default_values.get(param_name, 14)
                if any(word in param_name for word in ["length", "period"]):
                    min_val = 2
                    max_val = 200
                else:
                    min_val = None
                    max_val = 100
                ranges[param_name] = {
                    "min": min_val,
                    "max": max_val,
                    "default": default_value,
                }

        return ranges

    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
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

                if param_config.min_value is not None and isinstance(
                    value, (int, float)
                ):
                    if value < param_config.min_value:
                        normalized[param_name] = param_config.min_value
                if param_config.max_value is not None and isinstance(
                    value, (int, float)
                ):
                    if value > param_config.max_value:
                        normalized[param_name] = param_config.max_value

        return normalized

    def generate_random_parameters(self, preset: str | None = None) -> Dict[str, Any]:
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
                continue

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
                    params[param_name] = random.randint(int(min_val), int(max_val))
                else:
                    params[param_name] = random.uniform(float(min_val), float(max_val))
            else:
                params[param_name] = param_config.default_value

        return params


class IndicatorConfigRegistry:
    """インジケーター設定レジストリ

    すべてのインジケーター設定を一元管理し、
    名前やエイリアスによる検索を提供します。
    """

    def __init__(self):
        self._configs: Dict[str, IndicatorConfig] = {}
        self._initialized: bool = False

    def register(self, config: IndicatorConfig) -> None:
        """設定を登録"""
        self._configs[config.indicator_name] = config
        # エイリアスも登録
        if config.aliases:
            for alias in config.aliases:
                self._configs[alias] = config

    def reset(self) -> None:
        """レジストリをクリア（テスト用）"""
        self._configs.clear()
        self._initialized = False

    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """設定を取得"""
        return self._configs.get(indicator_name.upper()) or self._configs.get(
            indicator_name
        )

    def list_indicators(self) -> List[str]:
        """登録されているインジケーター名のリストを取得"""
        return list(self._configs.keys())

    def get_all_indicators(self) -> Dict[str, IndicatorConfig]:
        """登録済みインジケーターの辞書を取得"""
        return dict(self._configs)

    def generate_parameters_for_indicator(
        self, indicator_type: str, preset: str | None = None
    ) -> Dict[str, Any]:
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


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()


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
    """全インジケーターの設定を初期化（後方互換性用）"""
    indicator_registry.ensure_initialized()


# モジュールロード時に自動初期化
initialize_all_indicators()
