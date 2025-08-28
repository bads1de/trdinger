"""
統合された戦略モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を統合的に定義します。
以前は複数のファイルに分散していた機能を1つのファイルに統合し、
循環依存を解消し、理解しやすさを向上させました。
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class PositionSizingMethod(Enum):
    """ポジションサイジング決定方式"""

    HALF_OPTIMAL_F = "half_optimal_f"
    VOLATILITY_BASED = "volatility_based"
    FIXED_RATIO = "fixed_ratio"
    FIXED_QUANTITY = "fixed_quantity"


class TPSLMethod(Enum):
    """TP/SL決定方式"""

    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    VOLATILITY_BASED = "volatility_based"
    STATISTICAL = "statistical"
    ADAPTIVE = "adaptive"


# =============================================================================
# Base Classes
# =============================================================================


@dataclass
class Condition:
    """
    条件

    エントリー・イグジット条件を表現します。
    """

    left_operand: Union[Dict[str, Any], str, float]
    operator: str
    right_operand: Union[Dict[str, Any], str, float]

    def __post_init__(self):
        """型の正規化: 数値はfloatへ（テストの型要件に合わせる）"""
        try:
            if isinstance(self.left_operand, int):
                self.left_operand = float(self.left_operand)
            if isinstance(self.right_operand, int):
                self.right_operand = float(self.right_operand)
        except Exception:
            pass

    def validate(self) -> bool:
        """条件の妥当性を検証"""
        validator = GeneValidator()
        return validator.validate_condition(self)[0]


@dataclass
class ConditionGroup:
    """
    OR条件グループ

    conditions のいずれかが True なら True。
    ANDで使う側の配列にこのグループを1要素として混在させることで、
    A AND (B OR C) を表現できる。
    """

    conditions: List[Condition] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.conditions) == 0

    def validate(self) -> bool:
        validator = GeneValidator()
        for c in self.conditions:
            ok, _ = validator.validate_condition(c)
            if not ok:
                return False
        return True


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。
    """

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    json_config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証"""
        validator = GeneValidator()
        return validator.validate_indicator_gene(self)

    def get_json_config(self) -> Dict[str, Any]:
        """JSON形式の設定を取得"""
        try:
            from app.services.indicators.config import indicator_registry

            config = indicator_registry.get_indicator_config(self.type)
            if config:
                resolved_params = {}
                for param_name, param_config in config.parameters.items():
                    resolved_params[param_name] = self.parameters.get(
                        param_name, param_config.default_value
                    )
                return {"indicator": self.type, "parameters": resolved_params}
            return {"indicator": self.type, "parameters": self.parameters}
        except ImportError:
            return {"indicator": self.type, "parameters": self.parameters}


# =============================================================================
# Gene Classes
# =============================================================================


@dataclass
class PositionSizingGene:
    """
    ポジションサイジング遺伝子

    GA最適化対象としてのポジションサイジング設定を表現します。
    """

    method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_BASED
    lookback_period: int = 100
    optimal_f_multiplier: float = 0.5
    atr_period: int = 14
    atr_multiplier: float = 2.0
    risk_per_trade: float = 0.02
    fixed_ratio: float = 0.1
    fixed_quantity: float = 1.0
    min_position_size: float = 0.01
    max_position_size: float = 9999.0
    enabled: bool = True
    priority: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "method": self.method.value,
            "lookback_period": self.lookback_period,
            "optimal_f_multiplier": self.optimal_f_multiplier,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "risk_per_trade": self.risk_per_trade,
            "fixed_ratio": self.fixed_ratio,
            "fixed_quantity": self.fixed_quantity,
            "min_position_size": self.min_position_size,
            "max_position_size": self.max_position_size,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSizingGene":
        """辞書形式から遺伝子を復元"""
        return cls(
            method=PositionSizingMethod(data.get("method", "fixed_ratio")),
            lookback_period=data.get("lookback_period", 100),
            optimal_f_multiplier=data.get("optimal_f_multiplier", 0.5),
            atr_period=data.get("atr_period", 14),
            atr_multiplier=data.get("atr_multiplier", 2.0),
            risk_per_trade=data.get("risk_per_trade", 0.02),
            fixed_ratio=data.get("fixed_ratio", 0.1),
            fixed_quantity=data.get("fixed_quantity", 1.0),
            min_position_size=data.get("min_position_size", 0.01),
            max_position_size=data.get("max_position_size", 50.0),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []
        try:
            from ..config.constants import POSITION_SIZING_LIMITS

            lb_min, lb_max = POSITION_SIZING_LIMITS["lookback_period"]
            if not (lb_min <= self.lookback_period <= lb_max):
                errors.append(
                    f"lookback_periodは{lb_min}-{lb_max}の範囲である必要があります"
                )

            # 他の検証も同様に実装...

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (50 <= self.lookback_period <= 200):
                errors.append("lookback_periodは50-200の範囲である必要があります")

        return len(errors) == 0, errors


@dataclass
class TPSLGene:
    """
    TP/SL遺伝子

    GA最適化対象としてのTP/SL設定を表現します。
    """

    method: TPSLMethod = TPSLMethod.RISK_REWARD_RATIO
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    risk_reward_ratio: float = 2.0
    base_stop_loss: float = 0.03
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    atr_period: int = 14
    lookback_period: int = 100
    confidence_threshold: float = 0.7
    method_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "fixed": 0.25,
            "risk_reward": 0.35,
            "volatility": 0.25,
            "statistical": 0.15,
        }
    )
    enabled: bool = True
    priority: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """遺伝子を辞書形式に変換"""
        return {
            "method": self.method.value,
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 3),
            "base_stop_loss": round(self.base_stop_loss, 4),
            "atr_multiplier_sl": round(self.atr_multiplier_sl, 3),
            "atr_multiplier_tp": round(self.atr_multiplier_tp, 3),
            "atr_period": self.atr_period,
            "lookback_period": self.lookback_period,
            "confidence_threshold": round(self.confidence_threshold, 3),
            "method_weights": {k: round(v, 3) for k, v in self.method_weights.items()},
            "enabled": self.enabled,
            "priority": round(self.priority, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TPSLGene":
        """辞書形式から遺伝子を復元"""
        return cls(
            method=TPSLMethod(data.get("method", "risk_reward_ratio")),
            stop_loss_pct=data.get("stop_loss_pct", 0.03),
            take_profit_pct=data.get("take_profit_pct", 0.06),
            risk_reward_ratio=data.get("risk_reward_ratio", 2.0),
            base_stop_loss=data.get("base_stop_loss", 0.03),
            atr_multiplier_sl=data.get("atr_multiplier_sl", 2.0),
            atr_multiplier_tp=data.get("atr_multiplier_tp", 3.0),
            atr_period=data.get("atr_period", 14),
            lookback_period=data.get("lookback_period", 100),
            confidence_threshold=data.get("confidence_threshold", 0.7),
            method_weights=data.get(
                "method_weights",
                {
                    "fixed": 0.25,
                    "risk_reward": 0.35,
                    "volatility": 0.25,
                    "statistical": 0.15,
                },
            ),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """遺伝子の妥当性を検証"""
        errors = []
        try:
            from ..config.constants import TPSL_LIMITS

            sl_min, sl_max = TPSL_LIMITS["stop_loss_pct"]
            if not (sl_min <= self.stop_loss_pct <= sl_max):
                errors.append(
                    f"stop_loss_pct must be between {sl_min*100:.1f}% and {sl_max*100:.0f}%"
                )

            # 他の検証も同様に実装...

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (0.005 <= self.stop_loss_pct <= 0.15):
                errors.append("stop_loss_pct must be between 0.5% and 15%")

        return len(errors) == 0, errors


@dataclass
class StrategyGene:
    """
    戦略遺伝子

    完全な取引戦略を表現する遺伝子です。
    """

    MAX_INDICATORS = 5

    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    long_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    short_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    risk_management: Dict[str, Any] = field(default_factory=dict)
    tpsl_gene: Optional[TPSLGene] = None
    position_sizing_gene: Optional[PositionSizingGene] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_long_conditions(self) -> List[Union[Condition, ConditionGroup]]:
        """有効なロング条件を取得（後方互換性を考慮）"""
        if self.long_entry_conditions:
            return self.long_entry_conditions
        elif self.entry_conditions:
            return list(self.entry_conditions)
        else:
            return []

    def get_effective_short_conditions(self) -> List[Union[Condition, ConditionGroup]]:
        """有効なショート条件を取得（後方互換性を考慮）"""
        if self.short_entry_conditions:
            return self.short_entry_conditions
        elif self.entry_conditions and not self.long_entry_conditions:
            return list(self.entry_conditions)
        else:
            return []

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック"""
        return (
            self.long_entry_conditions is not None
            and len(self.long_entry_conditions) > 0
        ) or (
            self.short_entry_conditions is not None
            and len(self.short_entry_conditions) > 0
        )

    @property
    def method(self):
        """ポジションサイジングメソッドを取得（後方互換性のため）"""
        if self.position_sizing_gene and hasattr(self.position_sizing_gene, "method"):
            return self.position_sizing_gene.method
        else:
            return PositionSizingMethod.FIXED_RATIO

    def validate(self):
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す"""
        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors


# =============================================================================
# Validator Class
# =============================================================================


class GeneValidator:
    """
    遺伝子バリデーター

    戦略遺伝子の妥当性検証を担当します。
    """

    def __init__(self):
        """初期化"""
        try:
            from ..config.constants import (
                OPERATORS,
                DATA_SOURCES,
                VALID_INDICATOR_TYPES,
            )

            self.valid_indicator_types = VALID_INDICATOR_TYPES
            self.valid_operators = OPERATORS
            self.valid_data_sources = DATA_SOURCES
        except ImportError:
            # フォールバック値
            self.valid_indicator_types = [
                "SMA",
                "EMA",
                "RSI",
                "MACD",
                "BB",
                "STOCH",
                "ADX",
                "CCI",
                "MFI",
                "WILLR",
                "ROC",
                "TSI",
                "UO",
                "AO",
                "KAMA",
                "TEMA",
                "TRIMA",
            ]
            self.valid_operators = [">", "<", ">=", "<=", "==", "!=", "above", "below"]
            self.valid_data_sources = ["open", "high", "low", "close", "volume"]

    def validate_indicator_gene(self, indicator_gene) -> bool:
        """指標遺伝子の妥当性を検証"""
        try:
            if not indicator_gene.type or not isinstance(indicator_gene.type, str):
                return False
            if not isinstance(indicator_gene.parameters, dict):
                return False

            # タイポ修正
            if indicator_gene.type.upper() == "UI":
                indicator_gene.type = "UO"
                logger.warning("指標タイプ 'UI' を 'UO' に修正しました")

            if indicator_gene.type not in self.valid_indicator_types:
                return False

            if "period" in indicator_gene.parameters:
                period = indicator_gene.parameters["period"]
                if not isinstance(period, (int, float)) or period <= 0:
                    return False

            return True
        except Exception as e:
            logger.error(f"指標遺伝子バリデーションエラー: {e}")
            return False

    def validate_condition(self, condition) -> Tuple[bool, str]:
        """条件の妥当性を検証"""
        try:
            if not all(
                hasattr(condition, attr)
                for attr in ["operator", "left_operand", "right_operand"]
            ):
                return False, "条件オブジェクトに必要な属性がありません"

            if condition.operator not in self.valid_operators:
                return False, f"無効な演算子: {condition.operator}"

            left_valid, left_error = self._is_valid_operand_detailed(
                condition.left_operand
            )
            if not left_valid:
                return False, f"無効な左オペランド: {left_error}"

            right_valid, right_error = self._is_valid_operand_detailed(
                condition.right_operand
            )
            if not right_valid:
                return False, f"無効な右オペランド: {right_error}"

            return True, ""
        except Exception as e:
            return False, f"条件バリデーションエラー: {e}"

    def _is_valid_operand_detailed(self, operand) -> Tuple[bool, str]:
        """オペランドの妥当性を詳細に検証"""
        try:
            if operand is None:
                return False, "オペランドがNoneです"

            if isinstance(operand, (int, float)):
                return True, ""

            if isinstance(operand, str):
                if not operand or not operand.strip():
                    return False, "オペランドが空文字列です"

                operand = operand.strip()

                try:
                    float(operand)
                    return True, ""
                except ValueError:
                    pass

                if (
                    self._is_indicator_name(operand)
                    or operand in self.valid_data_sources
                ):
                    return True, ""

                return False, f"無効な文字列オペランド: '{operand}'"

            if isinstance(operand, dict):
                return self._validate_dict_operand_detailed(operand)

            return False, f"サポートされていないオペランド型: {type(operand)}"
        except Exception as e:
            return False, f"オペランド検証エラー: {e}"

    def _validate_dict_operand_detailed(self, operand: dict) -> Tuple[bool, str]:
        """辞書形式のオペランドを詳細に検証"""
        try:
            if operand.get("type") == "indicator":
                indicator_name = operand.get("name")
                if not indicator_name or not isinstance(indicator_name, str):
                    return False, "指標タイプの辞書にnameが設定されていません"
                if self._is_indicator_name(indicator_name.strip()):
                    return True, ""
                else:
                    return False, f"無効な指標名: '{indicator_name}'"

            elif operand.get("type") == "price":
                price_name = operand.get("name")
                if not price_name or not isinstance(price_name, str):
                    return False, "価格タイプの辞書にnameが設定されていません"
                if price_name.strip() in self.valid_data_sources:
                    return True, ""
                else:
                    return False, f"無効な価格データソース: '{price_name}'"

            elif operand.get("type") == "value":
                value = operand.get("value")
                if value is None:
                    return False, "数値タイプの辞書にvalueが設定されていません"
                if isinstance(value, (int, float)):
                    return True, ""
                elif isinstance(value, str):
                    try:
                        float(value.strip())
                        return True, ""
                    except ValueError:
                        return False, f"数値に変換できない文字列: '{value}'"
                else:
                    return False, f"無効な数値型: {type(value)}"

            else:
                return False, f"無効な辞書タイプ: '{operand.get('type')}'"
        except Exception as e:
            return False, f"辞書オペランド検証エラー: {e}"

    def _is_indicator_name(self, name: str) -> bool:
        """指標名かどうかを判定"""
        try:
            if not name or not name.strip():
                return False

            name = name.strip()

            if name.upper() == "UI":
                name = "UO"

            if name in self.valid_indicator_types:
                return True

            if "_" in name:
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    potential_indicator = parts[0].strip()
                    potential_param = parts[1].strip()

                    try:
                        float(potential_param)
                        if potential_indicator in self.valid_indicator_types:
                            return True
                    except ValueError:
                        pass

                indicator_type = name.split("_")[0].strip()
                if indicator_type in self.valid_indicator_types:
                    return True

            if name.endswith(("_0", "_1", "_2", "_3", "_4")):
                indicator_type = name.rsplit("_", 1)[0].strip()
                if indicator_type in self.valid_indicator_types:
                    return True

            return False
        except Exception as e:
            logger.error(f"指標名判定エラー: {e}")
            return False

    def clean_condition(self, condition) -> bool:
        """条件をクリーニングして修正可能な問題を自動修正"""
        try:
            if isinstance(condition.left_operand, str):
                condition.left_operand = condition.left_operand.strip()

            if isinstance(condition.right_operand, str):
                condition.right_operand = condition.right_operand.strip()

            if isinstance(condition.left_operand, dict):
                condition.left_operand = self._extract_operand_from_dict(
                    condition.left_operand
                )

            if isinstance(condition.right_operand, dict):
                condition.right_operand = self._extract_operand_from_dict(
                    condition.right_operand
                )

            if condition.operator == "above":
                condition.operator = ">"
            elif condition.operator == "below":
                condition.operator = "<"

            return True
        except Exception as e:
            logger.error(f"条件クリーニングエラー: {e}")
            return False

    def _extract_operand_from_dict(self, operand_dict: dict) -> str:
        """辞書形式のオペランドから文字列を抽出"""
        try:
            if operand_dict.get("type") == "indicator":
                return operand_dict.get("name", "")
            elif operand_dict.get("type") == "price":
                return operand_dict.get("name", "")
            elif operand_dict.get("type") == "value":
                value = operand_dict.get("value")
                return str(value) if value is not None else ""
            else:
                return str(operand_dict.get("name", ""))
        except Exception as e:
            logger.error(f"辞書オペランド抽出エラー: {e}")
            return ""

    def validate_strategy_gene(self, strategy_gene) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        errors = []

        try:
            # 指標数の制約チェック
            max_indicators = getattr(strategy_gene, "MAX_INDICATORS", 5)
            if len(strategy_gene.indicators) > max_indicators:
                errors.append(
                    f"指標数が上限({max_indicators})を超えています: {len(strategy_gene.indicators)}"
                )

            # 指標の妥当性チェック
            for i, indicator in enumerate(strategy_gene.indicators):
                if not self.validate_indicator_gene(indicator):
                    errors.append(f"指標{i}が無効です: {indicator.type}")

            # 条件の妥当性チェック
            def _validate_mixed_conditions(cond_list, label_prefix: str):
                for i, condition in enumerate(cond_list):
                    if isinstance(condition, ConditionGroup):
                        for j, c in enumerate(condition.conditions):
                            self.clean_condition(c)
                            is_valid, error_detail = self.validate_condition(c)
                            if not is_valid:
                                errors.append(
                                    f"{label_prefix}OR子条件{j}が無効です: {error_detail}"
                                )
                    else:
                        self.clean_condition(condition)
                        is_valid, error_detail = self.validate_condition(condition)
                        if not is_valid:
                            errors.append(
                                f"{label_prefix}{i}が無効です: {error_detail}"
                            )

            _validate_mixed_conditions(strategy_gene.entry_conditions, "エントリー条件")
            _validate_mixed_conditions(
                strategy_gene.long_entry_conditions, "ロングエントリー条件"
            )
            _validate_mixed_conditions(
                strategy_gene.short_entry_conditions, "ショートエントリー条件"
            )

            for i, condition in enumerate(strategy_gene.exit_conditions):
                self.clean_condition(condition)
                is_valid, error_detail = self.validate_condition(condition)
                if not is_valid:
                    errors.append(f"イグジット条件{i}が無効です: {error_detail}")

            # 最低限の条件チェック
            has_entry_conditions = (
                bool(strategy_gene.entry_conditions)
                or bool(strategy_gene.long_entry_conditions)
                or bool(strategy_gene.short_entry_conditions)
            )
            if not has_entry_conditions:
                errors.append("エントリー条件が設定されていません")

            # イグジット条件またはTP/SL遺伝子の存在チェック
            if not strategy_gene.exit_conditions:
                if not (strategy_gene.tpsl_gene and strategy_gene.tpsl_gene.enabled):
                    errors.append("イグジット条件が設定されていません")

            # 有効な指標の存在チェック
            enabled_indicators = [
                ind for ind in strategy_gene.indicators if ind.enabled
            ]
            if not enabled_indicators:
                errors.append("有効な指標が設定されていません")

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"戦略遺伝子バリデーションエラー: {e}")
            errors.append(f"バリデーション処理エラー: {e}")
            return False, errors


# =============================================================================
# Utility Functions
# =============================================================================


def create_random_position_sizing_gene(config=None) -> PositionSizingGene:
    """ランダムなポジションサイジング遺伝子を生成"""
    method_choices = list(PositionSizingMethod)
    method = random.choice(method_choices)

    return PositionSizingGene(
        method=method,
        lookback_period=random.randint(50, 200),
        optimal_f_multiplier=random.uniform(0.25, 0.75),
        atr_period=random.randint(10, 30),
        atr_multiplier=random.uniform(1.0, 4.0),
        risk_per_trade=random.uniform(0.01, 0.05),
        fixed_ratio=random.uniform(0.05, 0.3),
        fixed_quantity=random.uniform(0.1, 10.0),
        min_position_size=random.uniform(0.01, 0.05),
        max_position_size=random.uniform(5.0, 50.0),
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )


def create_random_tpsl_gene() -> TPSLGene:
    """ランダムなTP/SL遺伝子を生成"""
    method = random.choice(list(TPSLMethod))

    return TPSLGene(
        method=method,
        stop_loss_pct=random.uniform(0.01, 0.08),
        take_profit_pct=random.uniform(0.02, 0.15),
        risk_reward_ratio=random.uniform(1.2, 4.0),
        base_stop_loss=random.uniform(0.01, 0.06),
        atr_multiplier_sl=random.uniform(1.0, 3.0),
        atr_multiplier_tp=random.uniform(2.0, 5.0),
        atr_period=random.randint(10, 30),
        lookback_period=random.randint(50, 200),
        confidence_threshold=random.uniform(0.5, 0.9),
        method_weights={
            "fixed": random.uniform(0.1, 0.4),
            "risk_reward": random.uniform(0.2, 0.5),
            "volatility": random.uniform(0.1, 0.4),
            "statistical": random.uniform(0.1, 0.3),
        },
        enabled=True,
        priority=random.uniform(0.5, 1.5),
    )


def crossover_position_sizing_genes(
    parent1: PositionSizingGene, parent2: PositionSizingGene
) -> Tuple[PositionSizingGene, PositionSizingGene]:
    """ポジションサイジング遺伝子の交叉"""
    child1_method = random.choice([parent1.method, parent2.method])
    child2_method = random.choice([parent1.method, parent2.method])

    child1 = PositionSizingGene(
        method=child1_method,
        lookback_period=random.choice(
            [parent1.lookback_period, parent2.lookback_period]
        ),
        optimal_f_multiplier=(
            parent1.optimal_f_multiplier + parent2.optimal_f_multiplier
        )
        / 2,
        atr_period=random.choice([parent1.atr_period, parent2.atr_period]),
        atr_multiplier=(parent1.atr_multiplier + parent2.atr_multiplier) / 2,
        risk_per_trade=(parent1.risk_per_trade + parent2.risk_per_trade) / 2,
        fixed_ratio=(parent1.fixed_ratio + parent2.fixed_ratio) / 2,
        fixed_quantity=(parent1.fixed_quantity + parent2.fixed_quantity) / 2,
        min_position_size=(parent1.min_position_size + parent2.min_position_size) / 2,
        max_position_size=(parent1.max_position_size + parent2.max_position_size) / 2,
        enabled=random.choice([parent1.enabled, parent2.enabled]),
        priority=(parent1.priority + parent2.priority) / 2,
    )

    child2 = PositionSizingGene(
        method=child2_method,
        lookback_period=random.choice(
            [parent2.lookback_period, parent1.lookback_period]
        ),
        optimal_f_multiplier=(
            parent2.optimal_f_multiplier + parent1.optimal_f_multiplier
        )
        / 2,
        atr_period=random.choice([parent2.atr_period, parent1.atr_period]),
        atr_multiplier=(parent2.atr_multiplier + parent1.atr_multiplier) / 2,
        risk_per_trade=(parent2.risk_per_trade + parent1.risk_per_trade) / 2,
        fixed_ratio=(parent2.fixed_ratio + parent1.fixed_ratio) / 2,
        fixed_quantity=(parent2.fixed_quantity + parent1.fixed_quantity) / 2,
        min_position_size=(parent2.min_position_size + parent1.min_position_size) / 2,
        max_position_size=(parent2.max_position_size + parent1.max_position_size) / 2,
        enabled=random.choice([parent2.enabled, parent1.enabled]),
        priority=(parent2.priority + parent1.priority) / 2,
    )

    return child1, child2


def crossover_tpsl_genes(
    parent1: TPSLGene, parent2: TPSLGene
) -> Tuple[TPSLGene, TPSLGene]:
    """TP/SL遺伝子の交叉"""
    if random.random() < 0.5:
        child1_method = parent2.method
        child2_method = parent1.method
    else:
        child1_method = parent1.method
        child2_method = parent2.method

    child1 = TPSLGene(
        method=child1_method,
        stop_loss_pct=(parent1.stop_loss_pct + parent2.stop_loss_pct) / 2,
        take_profit_pct=(parent1.take_profit_pct + parent2.take_profit_pct) / 2,
        risk_reward_ratio=(parent1.risk_reward_ratio + parent2.risk_reward_ratio) / 2,
        base_stop_loss=(parent1.base_stop_loss + parent2.base_stop_loss) / 2,
        atr_multiplier_sl=(parent1.atr_multiplier_sl + parent2.atr_multiplier_sl) / 2,
        atr_multiplier_tp=(parent1.atr_multiplier_tp + parent2.atr_multiplier_tp) / 2,
        atr_period=random.choice([parent1.atr_period, parent2.atr_period]),
        lookback_period=random.choice(
            [parent1.lookback_period, parent2.lookback_period]
        ),
        confidence_threshold=(
            parent1.confidence_threshold + parent2.confidence_threshold
        )
        / 2,
    )

    child2 = TPSLGene(
        method=child2_method,
        stop_loss_pct=(parent2.stop_loss_pct + parent1.stop_loss_pct) / 2,
        take_profit_pct=(parent2.take_profit_pct + parent1.take_profit_pct) / 2,
        risk_reward_ratio=(parent2.risk_reward_ratio + parent1.risk_reward_ratio) / 2,
        base_stop_loss=(parent2.base_stop_loss + parent1.base_stop_loss) / 2,
        atr_multiplier_sl=(parent2.atr_multiplier_sl + parent1.atr_multiplier_sl) / 2,
        atr_multiplier_tp=(parent2.atr_multiplier_tp + parent1.atr_multiplier_tp) / 2,
        atr_period=random.choice([parent2.atr_period, parent1.atr_period]),
        lookback_period=random.choice(
            [parent2.lookback_period, parent1.lookback_period]
        ),
        confidence_threshold=(
            parent2.confidence_threshold + parent1.confidence_threshold
        )
        / 2,
    )

    return child1, child2


def mutate_position_sizing_gene(
    gene: PositionSizingGene, mutation_rate: float = 0.1
) -> PositionSizingGene:
    """ポジションサイジング遺伝子の突然変異"""
    mutated = PositionSizingGene(**gene.to_dict())

    if random.random() < mutation_rate:
        mutated.method = random.choice(list(PositionSizingMethod))

    if random.random() < mutation_rate:
        mutated.lookback_period = max(
            50, min(200, int(mutated.lookback_period * random.uniform(0.8, 1.2)))
        )

    if random.random() < mutation_rate:
        mutated.optimal_f_multiplier = max(
            0.25, min(0.75, mutated.optimal_f_multiplier * random.uniform(0.8, 1.2))
        )

    # 他のパラメータも同様に突然変異...

    return mutated


def mutate_tpsl_gene(gene: TPSLGene, mutation_rate: float = 0.1) -> TPSLGene:
    """TP/SL遺伝子の突然変異"""
    mutated = TPSLGene(**gene.to_dict())

    if random.random() < mutation_rate:
        mutated.method = random.choice(list(TPSLMethod))

    if random.random() < mutation_rate:
        mutated.stop_loss_pct *= random.uniform(0.8, 1.2)
        mutated.stop_loss_pct = max(0.005, min(mutated.stop_loss_pct, 0.15))

    # 他のパラメータも同様に突然変異...

    return mutated
