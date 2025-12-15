"""
インジケーター設定管理クラス

JSON形式でのインジケーター設定を管理し、
パラメータ埋め込み文字列からの移行をサポートします。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


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


logger = logging.getLogger(__name__)


class IndicatorResultType(Enum):
    """インジケーター結果タイプ"""

    SINGLE = "single"  # 単一値（例：RSI、SMA）
    COMPLEX = "complex"  # 複数値（例：MACD、Bollinger Bands）


@dataclass
class ParameterConfig:
    """パラメータ設定"""

    name: str  # パラメータ名（例：period, fast_period）
    default_value: Union[int, float, str, bool]  # デフォルト値
    min_value: Optional[Union[int, float, str, bool]] = None  # 最小値
    max_value: Optional[Union[int, float, str, bool]] = None  # 最大値
    description: Optional[str] = None  # パラメータの説明
    # 探索プリセット: 用途に応じた探索範囲（例: short_term, mid_term, long_term）
    presets: Optional[Dict[str, tuple]] = None

    def validate_value(self, value: Any) -> bool:
        """与えられた値がこのパラメータの制約（範囲など）を満たすか検証する"""
        if not isinstance(value, (int, float)):
            # 数値でない場合は検証スキップ（または型チェックを厳密に行う）
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
        # プリセットが定義されていて、該当するプリセット名がある場合
        if self.presets and preset_name in self.presets:
            return self.presets[preset_name]

        # フォールバック：デフォルトの min/max 範囲を使用
        return (self.min_value, self.max_value)


@dataclass
class IndicatorConfig:
    """インジケーター設定のメタ情報を保持するシンプルなデータクラス

    pandas-taとの連携設定を含む完全なインジケーター定義を提供します。
    """

    indicator_name: str
    adapter_function: Optional[Any] = None
    required_data: List[str] = field(default_factory=list)
    result_type: IndicatorResultType = IndicatorResultType.SINGLE
    scale_type: IndicatorScaleType = IndicatorScaleType.PRICE_RATIO
    category: Optional[str] = None
    output_names: Optional[List[str]] = None
    default_output: Optional[str] = None
    aliases: Optional[List[str]] = None
    param_map: Dict[str, Optional[str]] = field(default_factory=dict)
    # 後方互換性のための追加
    parameters: Dict[str, Any] = field(default_factory=dict)

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
    # 例: [{"type": "less_than", "param1": "fast", "param2": "slow"}]
    # type: "less_than" (param1 < param2), "greater_than" (param1 > param2),
    #       "min_difference" (param1 - param2 >= min_diff)
    parameter_constraints: Optional[List[Dict[str, Any]]] = None

    # 絶対的最小データ長（パラメータに関係なく必要な最低限のデータポイント数）
    absolute_min_length: int = 1

    def __post_init__(self):
        """後処理でパラメータを構築"""
        if not self.parameters:
            self.parameters = self._build_parameters_from_defaults()

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

            # 必要なパラメータが存在しない場合はスキップ
            if param1_name not in params or param2_name not in params:
                continue

            param1_value = params[param1_name]
            param2_value = params[param2_name]

            # 数値でない場合はスキップ
            if not isinstance(param1_value, (int, float)) or not isinstance(
                param2_value, (int, float)
            ):
                continue

            if constraint_type == "less_than":
                # param1 < param2 でなければならない
                if param1_value >= param2_value:
                    errors.append(
                        f"{param1_name}({param1_value}) は "
                        f"{param2_name}({param2_value}) より小さくなければなりません"
                    )

            elif constraint_type == "greater_than":
                # param1 > param2 でなければならない
                if param1_value <= param2_value:
                    errors.append(
                        f"{param1_name}({param1_value}) は "
                        f"{param2_name}({param2_value}) より大きくなければなりません"
                    )

            elif constraint_type == "min_difference":
                # param1 - param2 >= min_diff でなければならない
                min_diff = constraint.get("min_diff", 0)
                actual_diff = param1_value - param2_value
                if actual_diff < min_diff:
                    errors.append(
                        f"{param1_name}({param1_value}) と {param2_name}({param2_value}) の差は "
                        f"最低 {min_diff} 必要ですが、{actual_diff} です"
                    )

        return len(errors) == 0, errors

    def _build_parameters_from_defaults(self) -> Dict[str, Any]:
        """デフォルト値からパラメータを構築"""
        params = {}

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
                min_val = 2
                max_val = 200
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
        """パラメータ範囲を取得"""
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
        """パラメータ正規化"""
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

                # 範囲チェック
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


class IndicatorConfigRegistry:
    """インジケーター設定レジストリ"""

    def __init__(self):
        self._configs: Dict[str, IndicatorConfig] = {}

    def register(self, config: IndicatorConfig) -> None:
        """設定を登録"""
        self._configs[config.indicator_name] = config
        # エイリアスも登録
        if config.aliases:
            for alias in config.aliases:
                self._configs[alias] = config

    def reset(self) -> None:
        """レジストリをクリア（テスト用ユーティリティ）"""
        self._configs.clear()

    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """設定を取得 (get をリネーム)"""
        return self._configs.get(indicator_name)

    def list_indicators(self) -> List[str]:
        """登録されているインジケーター名のリストを取得"""
        return list(self._configs.keys())

    def get_all_indicators(self) -> Dict[str, IndicatorConfig]:
        """登録済みインジケーターのディクショナリを取得

        Returns:
            Dict[str, IndicatorConfig]: name/alias -> IndicatorConfig のマップ
        """
        return dict(self._configs)

    def generate_parameters_for_indicator(
        self, indicator_type: str, preset: str | None = None
    ) -> Dict[str, Any]:
        """
        指標タイプに応じたパラメータを生成

        Args:
            indicator_type: 指標タイプ（例：RSI, MACD）
            preset: 探索プリセット名（例：short_term, mid_term, long_term）
                None の場合はデフォルト範囲を使用

        Returns:
            生成されたパラメータ辞書
        """
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        try:
            config = self.get_indicator_config(indicator_type)
            if config:
                manager = IndicatorParameterManager()
                return manager.generate_parameters(
                    indicator_type, config, preset=preset
                )
            else:
                logger.warning(f"指標 {indicator_type} の設定が見つかりません")
                return {}
        except Exception as e:
            logger.error(f"指標 {indicator_type} のパラメータ生成に失敗: {e}")
            return {}


# グローバルレジストリインスタンス
indicator_registry = IndicatorConfigRegistry()


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    from app.services.indicators.manifests.registry import register_indicator_manifest

    register_indicator_manifest(indicator_registry)


def generate_positional_functions() -> set:
    """レジストリからPOSITIONAL_DATA_FUNCTIONSを動的生成

    Returns:
        位置パラメータを使用する関数名のセット
    """
    functions = set()
    all_indicators = indicator_registry.get_all_indicators()

    for name, indicator_config in all_indicators.items():
        # エイリアスではなく本名のみを使用
        if not indicator_config.aliases or name not in indicator_config.aliases:
            functions.add(indicator_config.indicator_name.lower())

    return functions


# 全インジケーター初期化
initialize_all_indicators()

# 動的設定初期化（レジストリから生成）
POSITIONAL_DATA_FUNCTIONS = generate_positional_functions()


