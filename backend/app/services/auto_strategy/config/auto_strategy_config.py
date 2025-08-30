"""
オートストラテジー統合設定クラス

このモジュールはオートストラテジーの全ての設定を構造化して一元管理します。
constants.pyに統合された定数を基に、構造化された設定クラスを提供します。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import (
    # 取引設定
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    # 指標設定
    VALID_INDICATOR_TYPES,
    INDICATOR_CHARACTERISTICS,
    ML_INDICATOR_TYPES,
    # GA設定
    GA_DEFAULT_CONFIG,
    DEFAULT_FITNESS_WEIGHTS,
    DEFAULT_FITNESS_CONSTRAINTS,
    GA_PARAMETER_RANGES,
    GA_THRESHOLD_RANGES,
    GA_DEFAULT_FITNESS_SHARING,
    # TPSL設定
    TPSL_METHODS,
    TPSL_LIMITS,
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
    # ポジションサイジング設定
    POSITION_SIZING_METHODS,
    POSITION_SIZING_LIMITS,
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_POSITION_SIZING_LOOKBACK_RANGE,
    GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_ATR_PERIOD_RANGE,
    GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
    GA_POSITION_SIZING_FIXED_RATIO_RANGE,
    GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
    GA_POSITION_SIZING_MIN_SIZE_RANGE,
    GA_POSITION_SIZING_MAX_SIZE_RANGE,
    GA_POSITION_SIZING_PRIORITY_RANGE,
    # その他の設定
    OPERATORS,
    DATA_SOURCES,
    ERROR_CODES,
    THRESHOLD_RANGES,
    CONSTRAINTS,
    DEFAULT_GA_OBJECTIVES,
    DEFAULT_GA_OBJECTIVE_WEIGHTS,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(ABC):
    """設定クラスの基底クラス"""

    enabled: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        pass

    def validate(self) -> Tuple[bool, List[str]]:
        """共通検証ロジック"""
        errors = []

        try:
            # 必須フィールドチェック
            required_fields = self.validation_rules.get("required_fields", [])
            for field_name in required_fields:
                if not hasattr(self, field_name) or getattr(self, field_name) is None:
                    errors.append(f"必須フィールド '{field_name}' が設定されていません")

            # 範囲チェック
            range_rules = self.validation_rules.get("ranges", {})
            for field_name, (min_val, max_val) in range_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if isinstance(value, (int, float)) and not (
                        min_val <= value <= max_val
                    ):
                        errors.append(
                            f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください"
                        )

            # 型チェック
            type_rules = self.validation_rules.get("types", {})
            for field_name, expected_type in type_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if value is not None and not isinstance(value, expected_type):
                        errors.append(
                            f"'{field_name}' は {expected_type.__name__} 型である必要があります"
                        )

            # カスタム検証
            custom_errors = self._custom_validation()
            errors.extend(custom_errors)

        except Exception as e:
            logger.error(f"設定検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return len(errors) == 0, errors

    def _custom_validation(self) -> List[str]:
        """サブクラスでオーバーライド可能なカスタム検証"""
        return []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """辞書から設定オブジェクトを作成"""
        try:
            # まずデフォルトインスタンスを作成
            instance = cls()

            # データで更新
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            return instance
        except Exception as e:
            logger.error(f"設定オブジェクト作成エラー: {e}", exc_info=True)
            raise ValueError(f"設定の作成に失敗しました: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        try:
            result = {}
            for field_info in fields(self):
                value = getattr(self, field_info.name)
                # 複雑なオブジェクトの場合は文字列化
                if hasattr(value, "__dict__"):
                    result[field_info.name] = str(value)
                else:
                    result[field_info.name] = value
            return result
        except Exception as e:
            logger.error(f"設定辞書変換エラー: {e}", exc_info=True)
            return {}

    def to_json(self) -> str:
        """設定オブジェクトをJSON文字列に変換"""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSON変換エラー: {e}", exc_info=True)
            return "{}"

    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """JSON文字列から設定オブジェクトを復元"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")


@dataclass
class TradingSettings(BaseConfig):
    """取引基本設定"""

    # 基本取引設定
    default_symbol: str = DEFAULT_SYMBOL
    default_timeframe: str = DEFAULT_TIMEFRAME
    supported_symbols: List[str] = field(
        default_factory=lambda: SUPPORTED_SYMBOLS.copy()
    )
    supported_timeframes: List[str] = field(
        default_factory=lambda: SUPPORTED_TIMEFRAMES.copy()
    )

    # 運用制約
    min_trades: int = CONSTRAINTS["min_trades"]
    max_drawdown_limit: float = CONSTRAINTS["max_drawdown_limit"]
    max_position_size: float = CONSTRAINTS["max_position_size"]
    min_position_size: float = CONSTRAINTS["min_position_size"]

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "default_symbol": DEFAULT_SYMBOL,
            "default_timeframe": DEFAULT_TIMEFRAME,
            "supported_symbols": SUPPORTED_SYMBOLS.copy(),
            "supported_timeframes": SUPPORTED_TIMEFRAMES.copy(),
            "min_trades": CONSTRAINTS["min_trades"],
            "max_drawdown_limit": CONSTRAINTS["max_drawdown_limit"],
            "max_position_size": CONSTRAINTS["max_position_size"],
            "min_position_size": CONSTRAINTS["min_position_size"],
        }

    def _custom_validation(self) -> List[str]:
        """カスタム検証"""
        errors = []

        if self.default_symbol not in self.supported_symbols:
            errors.append(
                f"デフォルトシンボル '{self.default_symbol}' はサポート対象外です"
            )

        if self.default_timeframe not in self.supported_timeframes:
            errors.append(
                f"デフォルト時間軸 '{self.default_timeframe}' はサポート対象外です"
            )

        if self.min_position_size >= self.max_position_size:
            errors.append(
                "最小ポジションサイズは最大ポジションサイズより小さく設定してください"
            )

        return errors

    # from_dictメソッドを削除 - BaseConfigの統一実装を使用


@dataclass
class IndicatorSettings(BaseConfig):
    """テクニカル指標設定"""

    # 利用可能な指標
    valid_indicator_types: List[str] = field(
        default_factory=lambda: VALID_INDICATOR_TYPES.copy()
    )
    ml_indicator_types: List[str] = field(
        default_factory=lambda: ML_INDICATOR_TYPES.copy()
    )

    # 指標特性データベース
    indicator_characteristics: Dict[str, Any] = field(
        default_factory=lambda: INDICATOR_CHARACTERISTICS.copy()
    )

    # 演算子とデータソース
    operators: List[str] = field(default_factory=lambda: OPERATORS.copy())
    data_sources: List[str] = field(default_factory=lambda: DATA_SOURCES.copy())

    # 指標解決支援
    multi_output_mappings: Dict[str, str] = field(
        default_factory=lambda: {
            "AROON": "AROON_0",
            "MACD": "MACD_0",
            "STOCH": "STOCH_0",
            "BBANDS": "BBANDS_1",
        }
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "valid_indicator_types": VALID_INDICATOR_TYPES.copy(),
            "ml_indicator_types": ML_INDICATOR_TYPES.copy(),
            "indicator_characteristics": INDICATOR_CHARACTERISTICS.copy(),
            "operators": OPERATORS.copy(),
            "data_sources": DATA_SOURCES.copy(),
            "multi_output_mappings": {
                "AROON": "AROON_0",
                "MACD": "MACD_0",
                "STOCH": "STOCH_0",
                "BBANDS": "BBANDS_1",
            },
        }

    def get_all_indicators(self) -> List[str]:
        """全指標タイプを取得"""
        return self.valid_indicator_types + self.ml_indicator_types

    def get_indicator_characteristics(self, indicator: str) -> Optional[Dict[str, Any]]:
        """特定の指標の特性を取得"""
        return self.indicator_characteristics.get(indicator)

    # from_dictメソッドを削除 - BaseConfigの統一実装を使用


@dataclass
class GASettings(BaseConfig):
    """遺伝的アルゴリズム設定"""

    # 基本GA設定
    population_size: int = GA_DEFAULT_CONFIG["population_size"]
    generations: int = GA_DEFAULT_CONFIG["generations"]
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = GA_DEFAULT_CONFIG["elite_size"]
    max_indicators: int = GA_DEFAULT_CONFIG["max_indicators"]

    # 戦略生成制約
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3

    # パラメータ範囲
    parameter_ranges: Dict[str, List] = field(
        default_factory=lambda: GA_PARAMETER_RANGES.copy()
    )
    threshold_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
    )

    # フィットネス設定
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_WEIGHTS.copy()
    )
    fitness_constraints: Dict[str, Any] = field(
        default_factory=lambda: DEFAULT_FITNESS_CONSTRAINTS.copy()
    )

    # フィットネス共有設定
    fitness_sharing: Dict[str, Any] = field(
        default_factory=lambda: GA_DEFAULT_FITNESS_SHARING.copy()
    )

    # 多目的最適化設定
    enable_multi_objective: bool = False
    ga_objectives: List[str] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy()
    )
    ga_objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "population_size": GA_DEFAULT_CONFIG["population_size"],
            "generations": GA_DEFAULT_CONFIG["generations"],
            "crossover_rate": GA_DEFAULT_CONFIG["crossover_rate"],
            "mutation_rate": GA_DEFAULT_CONFIG["mutation_rate"],
            "elite_size": GA_DEFAULT_CONFIG["elite_size"],
            "max_indicators": GA_DEFAULT_CONFIG["max_indicators"],
            "min_indicators": 1,
            "min_conditions": 1,
            "max_conditions": 3,
            "parameter_ranges": GA_PARAMETER_RANGES.copy(),
            "threshold_ranges": GA_THRESHOLD_RANGES.copy(),
            "fitness_weights": DEFAULT_FITNESS_WEIGHTS.copy(),
            "fitness_constraints": DEFAULT_FITNESS_CONSTRAINTS.copy(),
            "fitness_sharing": GA_DEFAULT_FITNESS_SHARING.copy(),
            "enable_multi_objective": False,
            "ga_objectives": DEFAULT_GA_OBJECTIVES.copy(),
            "ga_objective_weights": DEFAULT_GA_OBJECTIVE_WEIGHTS.copy(),
        }

    def _custom_validation(self) -> List[str]:
        """カスタム検証"""
        errors = []

        if self.population_size <= 0:
            errors.append("人口サイズは正の整数である必要があります")

        if not (0 <= self.crossover_rate <= 1):
            errors.append("交叉率は0から1の範囲である必要があります")

        if not (0 <= self.mutation_rate <= 1):
            errors.append("突然変異率は0から1の範囲である必要があります")

        if self.elite_size >= self.population_size:
            errors.append("エリートサイズは人口サイズより小さく設定してください")

        if self.min_indicators > self.max_indicators:
            errors.append("最小指標数は最大指標数以下である必要があります")

        return errors

    # from_dictメソッドを削除 - BaseConfigの統一実装を使用


@dataclass
class TPSLSettings(BaseConfig):
    """TP/SL設定"""

    # TPSL方法
    methods: List[str] = field(default_factory=lambda: TPSL_METHODS.copy())
    default_tpsl_methods: List[str] = field(
        default_factory=lambda: GA_DEFAULT_TPSL_METHOD_CONSTRAINTS.copy()
    )

    # パラメータ範囲
    sl_range: List[float] = field(default_factory=lambda: GA_TPSL_SL_RANGE.copy())
    tp_range: List[float] = field(default_factory=lambda: GA_TPSL_TP_RANGE.copy())
    rr_range: List[float] = field(default_factory=lambda: GA_TPSL_RR_RANGE.copy())
    atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_TPSL_ATR_MULTIPLIER_RANGE.copy()
    )

    # 制限設定
    limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: TPSL_LIMITS.copy()
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "methods": TPSL_METHODS.copy(),
            "default_tpsl_methods": GA_DEFAULT_TPSL_METHOD_CONSTRAINTS.copy(),
            "sl_range": GA_TPSL_SL_RANGE.copy(),
            "tp_range": GA_TPSL_TP_RANGE.copy(),
            "rr_range": GA_TPSL_RR_RANGE.copy(),
            "atr_multiplier_range": GA_TPSL_ATR_MULTIPLIER_RANGE.copy(),
            "limits": TPSL_LIMITS.copy(),
        }

    def get_limits_for_param(self, param_name: str) -> Tuple[float, float]:
        """指定されたパラメータの制限を取得"""
        if param_name in self.limits:
            return self.limits[param_name]
        raise ValueError(f"不明なパラメータ: {param_name}")

    # from_dictメソッドを削除 - BaseConfigの統一実装を使用


@dataclass
class PositionSizingSettings(BaseConfig):
    """ポジションサイジング設定"""

    # サイジング方法
    methods: List[str] = field(default_factory=lambda: POSITION_SIZING_METHODS.copy())
    default_methods: List[str] = field(
        default_factory=lambda: GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS.copy()
    )

    # パラメータ範囲
    lookback_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_LOOKBACK_RANGE.copy()
    )
    optimal_f_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE.copy()
    )
    atr_period_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_PERIOD_RANGE.copy()
    )
    atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE.copy()
    )
    risk_per_trade_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_RISK_PER_TRADE_RANGE.copy()
    )
    fixed_ratio_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_RATIO_RANGE.copy()
    )
    fixed_quantity_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_QUANTITY_RANGE.copy()
    )
    min_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MIN_SIZE_RANGE.copy()
    )
    max_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MAX_SIZE_RANGE.copy()
    )
    priority_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_PRIORITY_RANGE.copy()
    )

    # 制限設定
    limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: POSITION_SIZING_LIMITS.copy()
    )

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "methods": POSITION_SIZING_METHODS.copy(),
            "default_methods": GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS.copy(),
            "lookback_range": GA_POSITION_SIZING_LOOKBACK_RANGE.copy(),
            "optimal_f_multiplier_range": GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE.copy(),
            "atr_period_range": GA_POSITION_SIZING_ATR_PERIOD_RANGE.copy(),
            "atr_multiplier_range": GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE.copy(),
            "risk_per_trade_range": GA_POSITION_SIZING_RISK_PER_TRADE_RANGE.copy(),
            "fixed_ratio_range": GA_POSITION_SIZING_FIXED_RATIO_RANGE.copy(),
            "fixed_quantity_range": GA_POSITION_SIZING_FIXED_QUANTITY_RANGE.copy(),
            "min_size_range": GA_POSITION_SIZING_MIN_SIZE_RANGE.copy(),
            "max_size_range": GA_POSITION_SIZING_MAX_SIZE_RANGE.copy(),
            "priority_range": GA_POSITION_SIZING_PRIORITY_RANGE.copy(),
            "limits": POSITION_SIZING_LIMITS.copy(),
        }

    # from_dictメソッドを削除 - BaseConfigの統一実装を使用


@dataclass
class AutoStrategyConfig:
    """オートストラテジー統合設定

    このクラスはオートストラテジーの全ての設定を一元管理します。
    """

    # 設定グループ
    trading: TradingSettings = field(default_factory=TradingSettings)
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    ga: GASettings = field(default_factory=GASettings)
    tpsl: TPSLSettings = field(default_factory=TPSLSettings)
    position_sizing: PositionSizingSettings = field(
        default_factory=PositionSizingSettings
    )

    # 共通設定
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_async_processing: bool = False
    log_level: str = "WARNING"

    # エラーハンドリング設定
    error_codes: Dict[str, str] = field(default_factory=lambda: ERROR_CODES.copy())
    threshold_ranges: Dict[str, List] = field(
        default_factory=lambda: THRESHOLD_RANGES.copy()
    )

    # 設定検証ルール
    validation_rules: Dict[str, Any] = field(default_factory=lambda: {
        "required_fields": [],
        "ranges": {
            "cache_ttl_hours": [1, 168],  # 1時間から1週間
        },
        "types": {
            "enable_caching": bool,
            "enable_async_processing": bool,
            "log_level": str,
        },
    })

    def get_default_values(self) -> Dict[str, Any]:
        """デフォルト値を取得"""
        return {
            "trading": TradingSettings().get_default_values(),
            "indicators": IndicatorSettings().get_default_values(),
            "ga": GASettings().get_default_values(),
            "tpsl": TPSLSettings().get_default_values(),
            "position_sizing": PositionSizingSettings().get_default_values(),
            "enable_caching": True,
            "cache_ttl_hours": 24,
            "enable_async_processing": False,
            "log_level": "WARNING",
            "error_codes": ERROR_CODES.copy(),
            "threshold_ranges": THRESHOLD_RANGES.copy(),
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """設定の妥当性を検証"""
        errors = []

        try:
            # 必須フィールドチェック
            required_fields = self.validation_rules.get("required_fields", [])
            for field_name in required_fields:
                if not hasattr(self, field_name) or getattr(self, field_name) is None:
                    errors.append(f"必須フィールド '{field_name}' が設定されていません")

            # 範囲チェック
            range_rules = self.validation_rules.get("ranges", {})
            for field_name, (min_val, max_val) in range_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                        errors.append(f"'{field_name}' は {min_val} から {max_val} の範囲で設定してください")

            # 型チェック
            type_rules = self.validation_rules.get("types", {})
            for field_name, expected_type in type_rules.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    if value is not None and not isinstance(value, expected_type):
                        errors.append(f"'{field_name}' は {expected_type.__name__} 型である必要があります")

            # カスタム検証
            custom_errors = self._custom_validation()
            errors.extend(custom_errors)

        except Exception as e:
            logger.error(f"AutoStrategyConfig検証中にエラーが発生: {e}", exc_info=True)
            errors.append(f"検証処理エラー: {e}")

        return len(errors) == 0, errors

    def _custom_validation(self) -> List[str]:
        """カスタム検証（サブクラスでオーバーライド可能）"""
        errors = []

        # cache_ttl_hoursの検証
        if self.cache_ttl_hours < 0:
            errors.append("キャッシュTTLは正の数である必要があります")

        # log_levelの検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            errors.append(f"無効なログレベル: {self.log_level}")

        return errors

    def validate_all(self) -> Tuple[bool, Dict[str, List[str]]]:
        """全ての設定グループを検証"""
        all_errors = {}
        is_valid = True

        # 各設定グループの検証
        settings_groups = {
            "trading": self.trading,
            "indicators": self.indicators,
            "ga": self.ga,
            "tpsl": self.tpsl,
            "position_sizing": self.position_sizing,
        }

        for group_name, group_config in settings_groups.items():
            valid, errors = group_config.validate()
            if not valid:
                all_errors[group_name] = errors
                is_valid = False

        # メイン設定の検証
        main_valid, main_errors = self.validate()
        if not main_valid:
            all_errors["main"] = main_errors
            is_valid = False

        return is_valid, all_errors

    def to_nested_dict(self) -> Dict[str, Any]:
        """ネストされた辞書形式に変換"""
        try:
            result = {}

            # 各設定グループを辞書化
            result["trading"] = self.trading.to_dict()
            result["indicators"] = self.indicators.to_dict()
            result["ga"] = self.ga.to_dict()
            result["tpsl"] = self.tpsl.to_dict()
            result["position_sizing"] = self.position_sizing.to_dict()

            # 共通設定
            result["enable_caching"] = self.enable_caching
            result["cache_ttl_hours"] = self.cache_ttl_hours
            result["enable_async_processing"] = self.enable_async_processing
            result["log_level"] = self.log_level
            result["error_codes"] = self.error_codes
            result["threshold_ranges"] = self.threshold_ranges

            return result
        except Exception as e:
            logger.error(f"設定の辞書変換エラー: {e}", exc_info=True)
            return {}

    @classmethod
    def from_nested_dict(cls, data: Dict[str, Any]) -> "AutoStrategyConfig":
        """ネストされた辞書から設定オブジェクトを作成"""
        try:
            # 設定グループの作成
            trading_data = data.get("trading", {})
            indicators_data = data.get("indicators", {})
            ga_data = data.get("ga", {})
            tpsl_data = data.get("tpsl", {})
            position_sizing_data = data.get("position_sizing", {})

            # 各設定グループのインスタンス化
            trading = TradingSettings.from_dict(trading_data)
            indicators = IndicatorSettings.from_dict(indicators_data)
            ga = GASettings.from_dict(ga_data)
            tpsl = TPSLSettings.from_dict(tpsl_data)
            position_sizing = PositionSizingSettings.from_dict(position_sizing_data)

            # メイン設定の作成
            instance = cls(
                trading=trading,
                indicators=indicators,
                ga=ga,
                tpsl=tpsl,
                position_sizing=position_sizing,
                enable_caching=data.get("enable_caching", True),
                cache_ttl_hours=data.get("cache_ttl_hours", 24),
                enable_async_processing=data.get("enable_async_processing", False),
                log_level=data.get("log_level", "WARNING"),
                error_codes=data.get("error_codes", ERROR_CODES.copy()),
                threshold_ranges=data.get("threshold_ranges", THRESHOLD_RANGES.copy()),
            )

            return instance
        except Exception as e:
            logger.error(f"設定オブジェクト作成エラー: {e}", exc_info=True)
            raise ValueError(f"設定の作成に失敗しました: {e}")

    def save_to_json(self, filepath: str) -> bool:
        """設定をJSONファイルに保存"""
        try:
            data = self.to_nested_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"設定のJSON保存エラー: {e}", exc_info=True)
            return False

    @classmethod
    def load_from_json(cls, filepath: str) -> "AutoStrategyConfig":
        """JSONファイルから設定を読み込み"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_nested_dict(data)
        except Exception as e:
            logger.error(f"設定のJSON読み込みエラー: {e}", exc_info=True)
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")


# デフォルト設定インスタンス
DEFAULT_AUTO_STRATEGY_CONFIG = AutoStrategyConfig()


def get_default_config() -> AutoStrategyConfig:
    """デフォルト設定を取得"""
    return DEFAULT_AUTO_STRATEGY_CONFIG


def create_config_from_file(filepath: str) -> AutoStrategyConfig:
    """設定ファイルを読み込んでAutoStrategyConfigを作成"""
    return AutoStrategyConfig.load_from_json(filepath)


def validate_config_file(filepath: str) -> Tuple[bool, Dict[str, List[str]]]:
    """設定ファイルの妥当性を検証"""
    try:
        config = AutoStrategyConfig.load_from_json(filepath)
        return config.validate_all()
    except Exception as e:
        return False, {"file_error": [f"設定ファイル読み込みエラー: {e}"]}


@dataclass
class GAConfig(BaseConfig):
    """
    遺伝的アルゴリズム設定

    GA実行時の全パラメータをフラットな構造で管理します。

    Args:
        auto_strategy_config: AutoStrategyConfigインスタンス（オプション）
                              指定された場合、このGAConfigはAutoStrategyConfig.gaから設定を継承します。
    """

    # 参照設定（AutoStrategyConfig統合用）
    auto_strategy_config: Optional[AutoStrategyConfig] = None

    # 進化アルゴリズム設定
    population_size: int = GA_DEFAULT_CONFIG["population_size"]
    generations: int = GA_DEFAULT_CONFIG["generations"]
    crossover_rate: float = GA_DEFAULT_CONFIG["crossover_rate"]
    mutation_rate: float = GA_DEFAULT_CONFIG["mutation_rate"]
    elite_size: int = GA_DEFAULT_CONFIG["elite_size"]
    max_indicators: int = GA_DEFAULT_CONFIG["max_indicators"]

    # 評価設定（単一目的最適化用、後方互換性のため保持）
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_WEIGHTS.copy()
    )
    primary_metric: str = "sharpe_ratio"
    fitness_constraints: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FITNESS_CONSTRAINTS.copy()
    )

    # 多目的最適化設定
    enable_multi_objective: bool = False  # 多目的最適化を有効にするかどうか
    objectives: List[str] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVES.copy()  # デフォルトは単一目的
    )  # 最適化する目的のリスト
    objective_weights: List[float] = field(
        default_factory=lambda: DEFAULT_GA_OBJECTIVE_WEIGHTS.copy()  # デフォルトは最大化
    )  # 各目的の重み（1.0=最大化、-1.0=最小化）

    # フィットネス共有設定
    enable_fitness_sharing: bool = GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"]
    sharing_radius: float = GA_DEFAULT_FITNESS_SHARING["sharing_radius"]
    sharing_alpha: float = GA_DEFAULT_FITNESS_SHARING["sharing_alpha"]

    # 指標設定
    allowed_indicators: List[str] = field(default_factory=list)
    indicator_mode: str = "technical_only"  # "technical_only", "ml_only"
    enable_ml_indicators: bool = True  # 後方互換性のため保持

    # パラメータ範囲設定
    parameter_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: GA_PARAMETER_RANGES.copy()
    )
    threshold_ranges: Dict[str, List] = field(
        default_factory=lambda: GA_THRESHOLD_RANGES.copy()
    )

    # 遺伝子生成設定
    min_indicators: int = 1
    min_conditions: int = 1
    max_conditions: int = 3
    price_data_weight: int = 3
    volume_data_weight: int = 1
    oi_fr_data_weight: int = 1
    numeric_threshold_probability: float = 0.8
    min_compatibility_score: float = 0.8
    strict_compatibility_score: float = 0.9

    # TP/SL GA最適化範囲設定（ユーザー設定ではなくGA制約）
    tpsl_method_constraints: List[str] = field(
        default_factory=lambda: GA_DEFAULT_TPSL_METHOD_CONSTRAINTS.copy()
    )  # GA最適化で使用可能なTP/SLメソッド
    tpsl_sl_range: List[float] = field(
        default_factory=lambda: GA_TPSL_SL_RANGE.copy()
    )  # SL範囲（1%-8%）
    tpsl_tp_range: List[float] = field(
        default_factory=lambda: GA_TPSL_TP_RANGE.copy()
    )  # TP範囲（2%-20%）
    tpsl_rr_range: List[float] = field(
        default_factory=lambda: GA_TPSL_RR_RANGE.copy()
    )  # リスクリワード比範囲
    tpsl_atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_TPSL_ATR_MULTIPLIER_RANGE.copy()
    )  # ATR倍率範囲

    # ポジションサイジング GA最適化範囲設定
    position_sizing_method_constraints: List[str] = field(
        default_factory=lambda: GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS.copy()
    )  # GA最適化で使用可能なポジションサイジングメソッド
    position_sizing_lookback_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_LOOKBACK_RANGE.copy()
    )  # ハーフオプティマルF用ルックバック期間
    position_sizing_optimal_f_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE.copy()
    )  # オプティマルF倍率範囲
    position_sizing_atr_period_range: List[int] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_PERIOD_RANGE.copy()
    )  # ATR計算期間範囲
    position_sizing_atr_multiplier_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE.copy()
    )  # ポジションサイジング用ATR倍率範囲
    position_sizing_risk_per_trade_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_RISK_PER_TRADE_RANGE.copy()
    )  # 1取引あたりのリスク範囲（1%-5%）
    position_sizing_fixed_ratio_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_RATIO_RANGE.copy()
    )  # 固定比率範囲（5%-30%）
    position_sizing_fixed_quantity_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_FIXED_QUANTITY_RANGE.copy()
    )  # 固定枚数範囲
    position_sizing_min_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MIN_SIZE_RANGE.copy()
    )  # 最小ポジションサイズ範囲
    position_sizing_max_size_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_MAX_SIZE_RANGE.copy()
    )  # 最大ポジションサイズ範囲（BTCトレードに適した範囲に拡大）
    position_sizing_priority_range: List[float] = field(
        default_factory=lambda: GA_POSITION_SIZING_PRIORITY_RANGE.copy()
    )  # 優先度範囲

    # 実行設定
    parallel_processes: Optional[int] = None
    random_state: Optional[int] = None
    log_level: str = "ERROR"  # エラーログのみ出力
    save_intermediate_results: bool = True
    progress_callback: Optional[Callable[["GAProgress"], None]] = None

    def validate(self) -> tuple[bool, List[str]]:
        """
        設定の妥当性を検証

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 進化設定の検証
        if self.population_size <= 0:
            errors.append("個体数は正の整数である必要があります")
        elif self.population_size > 1000:
            errors.append(
                "個体数は1000以下である必要があります（パフォーマンス上の制約）"
            )

        if self.generations <= 0:
            errors.append("世代数は正の整数である必要があります")
        elif self.generations > 500:
            errors.append(
                "世代数は500以下である必要があります（パフォーマンス上の制約）"
            )

        if not 0 <= self.crossover_rate <= 1:
            errors.append("交叉率は0-1の範囲である必要があります")

        if not 0 <= self.mutation_rate <= 1:
            errors.append("突然変異率は0-1の範囲である必要があります")

        if self.elite_size < 0 or self.elite_size >= self.population_size:
            errors.append("エリート保存数は0以上、個体数未満である必要があります")

        # 評価設定の検証
        if abs(sum(self.fitness_weights.values()) - 1.0) > 0.01:
            errors.append("フィットネス重みの合計は1.0である必要があります")

        required_metrics = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}
        missing_metrics = required_metrics - set(self.fitness_weights.keys())
        if missing_metrics:
            errors.append(f"必要なメトリクスが不足しています: {missing_metrics}")

        if self.primary_metric not in self.fitness_weights:
            errors.append(
                f"プライマリメトリクス '{self.primary_metric}' がフィットネス重みに含まれていません"
            )

        # 指標設定の検証
        if self.max_indicators <= 0:
            errors.append("最大指標数は正の整数である必要があります")
        elif self.max_indicators > 10:
            errors.append(
                "最大指標数は10以下である必要があります（パフォーマンス上の制約）"
            )

        if not self.allowed_indicators:
            errors.append("許可された指標リストが空です")
        else:
            try:
                from app.services.indicators import TechnicalIndicatorService

                valid_indicators = set(
                    TechnicalIndicatorService().get_supported_indicators().keys()
                )
                invalid_indicators = set(self.allowed_indicators) - valid_indicators
                if invalid_indicators:
                    errors.append(f"無効な指標が含まれています: {invalid_indicators}")
            except Exception:
                # インポートできない場合は検証スキップ
                logger.warning("指標検証がスキップされました")

        # パラメータ設定の検証
        for param_name, range_values in self.parameter_ranges.items():
            if not isinstance(range_values, list) or len(range_values) != 2:
                errors.append(
                    f"パラメータ '{param_name}' の範囲は [min, max] の形式である必要があります"
                )
            elif range_values[0] >= range_values[1]:
                errors.append(
                    f"パラメータ '{param_name}' の最小値は最大値より小さい必要があります"
                )

        # 実行設定の検証
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            errors.append(
                f"無効なログレベル: {self.log_level}. 有効な値: {valid_log_levels}"
            )

        if self.parallel_processes is not None:
            if self.parallel_processes <= 0:
                errors.append("並列プロセス数は正の整数である必要があります")
            elif self.parallel_processes > 32:
                errors.append("並列プロセス数は32以下である必要があります")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "fitness_weights": self.fitness_weights,
            "primary_metric": self.primary_metric,
            "max_indicators": self.max_indicators,
            # 多目的最適化設定
            "enable_multi_objective": self.enable_multi_objective,
            "objectives": self.objectives,
            "objective_weights": self.objective_weights,
            "allowed_indicators": self.allowed_indicators,
            "parameter_ranges": self.parameter_ranges,
            "threshold_ranges": self.threshold_ranges,
            "fitness_constraints": self.fitness_constraints,
            "min_indicators": self.min_indicators,
            "min_conditions": self.min_conditions,
            "max_conditions": self.max_conditions,
            "parallel_processes": self.parallel_processes,
            "random_state": self.random_state,
            "log_level": self.log_level,
            "save_intermediate_results": self.save_intermediate_results,
            # フィットネス共有設定
            "enable_fitness_sharing": self.enable_fitness_sharing,
            "sharing_radius": self.sharing_radius,
            "sharing_alpha": self.sharing_alpha,
            # 指標モード設定
            "indicator_mode": self.indicator_mode,
            "enable_ml_indicators": self.enable_ml_indicators,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        """辞書から復元（BaseConfig統一化バージョン）"""
        # GA特有のデータ前処理
        processed_data = cls._preprocess_ga_dict(data)

        # BaseConfigの標準from_dict処理を使用
        return super().from_dict(processed_data)

    @classmethod
    def _preprocess_ga_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """GAConfig特有のデータの前処理"""
        # allowed_indicatorsが空の場合はデフォルトの指標リストを使用
        if not data.get("allowed_indicators"):
            try:
                from app.services.indicators import TechnicalIndicatorService

                indicator_service = TechnicalIndicatorService()
                data["allowed_indicators"] = list(
                    indicator_service.get_supported_indicators().keys()
                )
            except Exception:
                # インポートできない場合はデフォルトを使用
                logger.warning("指標サービスの取得が失敗しました")
                data["allowed_indicators"] = []

        # fitness_weightsが指定されていない場合はデフォルト値を使用
        if not data.get("fitness_weights"):  # 空の辞書やNoneの場合
            data["fitness_weights"] = DEFAULT_FITNESS_WEIGHTS

        # 他のデフォルト値も設定（既存のget()ロジックを維持）
        defaults = {
            "population_size": GA_DEFAULT_CONFIG["population_size"],
            "generations": GA_DEFAULT_CONFIG["generations"],
            "crossover_rate": GA_DEFAULT_CONFIG["crossover_rate"],
            "mutation_rate": GA_DEFAULT_CONFIG["mutation_rate"],
            "elite_size": GA_DEFAULT_CONFIG.get("elite_size", 10),
            "primary_metric": "sharpe_ratio",
            "fitness_constraints": DEFAULT_FITNESS_CONSTRAINTS,
            "max_indicators": GA_DEFAULT_CONFIG["max_indicators"],
            "parameter_ranges": GA_PARAMETER_RANGES,
            "threshold_ranges": GA_THRESHOLD_RANGES,
            "min_indicators": 1,
            "min_conditions": 1,
            "max_conditions": 3,
            "log_level": "ERROR",
            "save_intermediate_results": True,
            # フィットネス共有設定
            "enable_fitness_sharing": GA_DEFAULT_FITNESS_SHARING["enable_fitness_sharing"],
            "sharing_radius": GA_DEFAULT_FITNESS_SHARING["sharing_radius"],
            "sharing_alpha": GA_DEFAULT_FITNESS_SHARING["sharing_alpha"],
            # 指標モード設定
            "indicator_mode": "mixed",
            "enable_ml_indicators": True,
            # 多目的最適化設定
            "enable_multi_objective": False,
            "objectives": DEFAULT_GA_OBJECTIVES,
            "objective_weights": DEFAULT_GA_OBJECTIVE_WEIGHTS,
            # 実行設定
            "parallel_processes": None,
            "random_state": None,
        }

        # デフォルト値をマージ
        for key, default_value in defaults.items():
            if data.get(key) is None:  # 明示的にNoneが設定されている場合のみ
                data[key] = default_value

        return data

    def apply_auto_strategy_config(self, config: AutoStrategyConfig) -> None:
        """
        AutoStrategyConfigから設定を適用

        Args:
            config: AutoStrategyConfigインスタンス
        """
        # GA設定をAutoStrategyConfigから継承
        ga_config = config.ga
        self.auto_strategy_config = config

        # 基本GAパラメータ
        self.population_size = ga_config.population_size
        self.generations = ga_config.generations
        self.crossover_rate = ga_config.crossover_rate
        self.mutation_rate = ga_config.mutation_rate
        self.elite_size = ga_config.elite_size
        self.max_indicators = ga_config.max_indicators
        self.min_indicators = ga_config.min_indicators
        self.min_conditions = ga_config.min_conditions
        self.max_conditions = ga_config.max_conditions

        # 評価関連設定
        self.fitness_weights = ga_config.fitness_weights.copy()
        self.fitness_constraints = ga_config.fitness_constraints.copy()
        self.enable_multi_objective = ga_config.enable_multi_objective
        self.objectives = ga_config.ga_objectives.copy()
        self.objective_weights = ga_config.ga_objective_weights.copy()

        # フィットネス共有設定
        self.enable_fitness_sharing = ga_config.fitness_sharing[
            "enable_fitness_sharing"
        ]
        self.sharing_radius = ga_config.fitness_sharing["sharing_radius"]
        self.sharing_alpha = ga_config.fitness_sharing["sharing_alpha"]

        # パラメータ範囲
        self.parameter_ranges = ga_config.parameter_ranges.copy()
        self.threshold_ranges = ga_config.threshold_ranges.copy()

        # 許可指標リスト
        if not self.allowed_indicators:
            try:
                from app.services.indicators import TechnicalIndicatorService

                indicator_service = TechnicalIndicatorService()
                self.allowed_indicators = list(
                    indicator_service.get_supported_indicators().keys()
                )
            except Exception:
                # Fallback: 設定から取得
                self.allowed_indicators = config.indicators.valid_indicator_types[:]

    @classmethod
    def from_auto_strategy_config(cls, config: AutoStrategyConfig) -> "GAConfig":
        """
        AutoStrategyConfigからGAConfigを作成

        Args:
            config: AutoStrategyConfigインスタンス

        Returns:
            GAConfigインスタンス
        """
        instance = cls()
        instance.apply_auto_strategy_config(config)
        return instance

    def get_default_values(self) -> Dict[str, Any]:
        """BaseConfig用のデフォルト値を取得"""
        return {
            "population_size": self.__class__.population_size,
            "generations": self.__class__.generations,
            "crossover_rate": self.__class__.crossover_rate,
            "mutation_rate": self.__class__.mutation_rate,
            "elite_size": self.__class__.elite_size,
            "fitness_weights": self.__class__.fitness_weights,
            "primary_metric": self.__class__.primary_metric,
            "max_indicators": self.__class__.max_indicators,
            "min_indicators": 1,
            "min_conditions": 1,
            "max_conditions": 3,
            "enabled": True,
        }

    # BaseConfigのメソッドをオーバーライド（既存機能を保持）
    def to_json(self) -> str:
        """JSON文字列に変換（BaseConfigの機能を活用）"""
        return super().to_json()

    @classmethod
    def from_json(cls, json_str: str) -> "GAConfig":
        """JSON文字列から復元（BaseConfigの機能を活用）"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"JSON復元エラー: {e}", exc_info=True)
            raise ValueError(f"JSON からの復元に失敗しました: {e}")

    @classmethod
    def create_default(cls) -> "GAConfig":
        """デフォルト設定を作成"""
        return cls()

    @classmethod
    def create_fast(cls) -> "GAConfig":
        """高速実行用設定を作成（オートストラテジー用デフォルト）"""
        return cls(
            population_size=10,
            generations=5,
            elite_size=2,
            max_indicators=3,
        )

    @classmethod
    def create_thorough(cls) -> "GAConfig":
        """徹底的な探索用設定を作成"""
        return cls(
            population_size=200,
            generations=100,
            crossover_rate=0.85,
            mutation_rate=0.05,
            elite_size=20,
            max_indicators=5,
            log_level="INFO",
            save_intermediate_results=True,
        )

    @classmethod
    def create_multi_objective(
        cls,
        objectives: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> "GAConfig":
        """
        多目的最適化用設定を作成

        Args:
            objectives: 最適化する目的のリスト（デフォルト: ["total_return", "max_drawdown"]）
            weights: 各目的の重み（デフォルト: [1.0, -1.0] = [最大化, 最小化]）
        """
        if objectives is None:
            objectives = ["total_return", "max_drawdown"]
        if weights is None:
            weights = [1.0, -1.0]  # total_return最大化、max_drawdown最小化

        return cls(
            population_size=50,  # 多目的最適化では少し大きめの個体数
            generations=30,
            enable_multi_objective=True,
            objectives=objectives,
            objective_weights=weights,
            max_indicators=3,
            log_level="INFO",
            save_intermediate_results=True,
        )


@dataclass
class GAProgress:
    """
    GA実行進捗情報

    リアルタイム進捗表示用のデータ構造
    """

    experiment_id: str
    current_generation: int
    total_generations: int
    best_fitness: float
    average_fitness: float
    execution_time: float
    estimated_remaining_time: float
    status: str = "running"  # "running", "completed", "error"
    best_strategy_preview: Optional[Dict[str, Any]] = None

    @property
    def progress_percentage(self) -> float:
        """進捗率（0-100）"""
        if self.total_generations == 0:
            return 0.0
        return (self.current_generation / self.total_generations) * 100

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "experiment_id": self.experiment_id,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "execution_time": self.execution_time,
            "estimated_remaining_time": self.estimated_remaining_time,
            "progress_percentage": self.progress_percentage,
            "status": self.status,
            "best_strategy_preview": self.best_strategy_preview,
        }
