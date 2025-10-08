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

    def __init__(
        self,
        name: str,
        default_value: Union[int, float, str, bool],
        min_value: Optional[Union[int, float, str, bool]] = None,
        max_value: Optional[Union[int, float, str, bool]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.description = description

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

    def __post_init__(self):
        """後処理でパラメータを構築"""
        if not self.parameters:
            self.parameters = self._build_parameters_from_defaults()

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

    def add_parameter(self, param: ParameterConfig) -> None:
        """互換性のためのダミー実装: パラメータを param_map に追加しないが呼び出しを許可する。"""
        # 実際の実装ではパラメータリストを保持するが、ここでは最小限にとどめる
        return None

    def generate_json_name(self) -> str:
        """JSON用の名称を生成（現状は indicator_name を返す）"""
        return self.indicator_name

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
                if param_config.min_value is not None and isinstance(value, (int, float)):
                    if value < param_config.min_value:
                        normalized[param_name] = param_config.min_value
                if param_config.max_value is not None and isinstance(value, (int, float)):
                    if value > param_config.max_value:
                        normalized[param_name] = param_config.max_value

        return normalized


class IndicatorConfigRegistry:
    """インジケーター設定レジストリ"""

    def __init__(self):
        self._configs: Dict[str, IndicatorConfig] = {}
        # 実験的インジケータ集合（ジェネレーターから参照）
        self.experimental_indicators = {
            "CMF",
        }

        # フォールバックマッピングをレジストリ内に定義
        self._fallback_indicators = {}

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

    def get_supported_indicator_names(self) -> List[str]:
        """サポートされている指標の名前のリストを取得 (新規追加)"""
        return list(self._configs.keys())

    def get_all_indicators(self) -> Dict[str, IndicatorConfig]:
        """登録済みインジケーターのディクショナリを取得

        Returns:
            Dict[str, IndicatorConfig]: name/alias -> IndicatorConfig のマップ
        """
        return dict(self._configs)

    def generate_parameters_for_indicator(self, indicator_type: str) -> Dict[str, Any]:
        """
        指標タイプに応じたパラメータを生成

        IndicatorParameterManagerシステムを使用した統一されたパラメータ生成。
        """
        from app.services.indicators.parameter_manager import IndicatorParameterManager

        try:
            config = self.get_indicator_config(indicator_type)
            if config:
                manager = IndicatorParameterManager()
                return manager.generate_parameters(indicator_type, config)
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
    from app.services.indicators.manifest import register_indicator_manifest

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
