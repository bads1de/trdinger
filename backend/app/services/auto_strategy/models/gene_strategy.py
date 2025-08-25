"""
戦略遺伝子モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .condition_group import ConditionGroup

from .gene_position_sizing import PositionSizingGene
from .gene_tpsl import TPSLGene
from .gene_validation import GeneValidator

logger = logging.getLogger(__name__)


@dataclass
class IndicatorGene:
    """
    指標遺伝子

    単一のテクニカル指標の設定を表現します。
    """

    type: str  # 指標タイプ（例: "SMA", "RSI", "MACD"）
    parameters: Dict[str, Any] = field(default_factory=dict)  # 指標パラメータ
    enabled: bool = True  # 指標の有効/無効

    # JSON形式サポート用の追加フィールド
    json_config: Dict[str, Any] = field(default_factory=dict)  # JSON形式の設定

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証"""
        validator = GeneValidator()
        return validator.validate_indicator_gene(self)

    def get_json_config(self) -> Dict[str, Any]:
        """JSON形式の設定を取得"""
        try:
            # 新しいJSON形式のインジケーター設定を使用
            from app.services.indicators.config import indicator_registry

            config = indicator_registry.get_indicator_config(self.type)
            if config:
                # IndicatorConfigから完全なJSON設定を構築
                resolved_params = {}
                for param_name, param_config in config.parameters.items():
                    resolved_params[param_name] = self.parameters.get(
                        param_name, param_config.default_value
                    )
                return {"indicator": self.type, "parameters": resolved_params}

            # フォールバック: 基本的なJSON形式
            return {"indicator": self.type, "parameters": self.parameters}

        except ImportError:
            # 設定モジュールが利用できない場合のフォールバック
            return {"indicator": self.type, "parameters": self.parameters}

    

    


@dataclass
class Condition:
    """
    条件

    エントリー・イグジット条件を表現します。
    """

    left_operand: Union[Dict[str, Any], str, float]  # 左オペランド
    operator: str  # 演算子
    right_operand: Union[Dict[str, Any], str, float]  # 右オペランド

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

    def _is_indicator_name(self, name: str) -> bool:
        """指標名かどうかを判定"""
        validator = GeneValidator()
        return validator._is_indicator_name(name)


@dataclass
class StrategyGene:
    """
    戦略遺伝子

    完全な取引戦略を表現する遺伝子です。
    """

    # 制約定数
    MAX_INDICATORS = 5  # 最大指標数

    # 戦略構成要素
    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[Condition] = field(
        default_factory=list
    )  # 後方互換性のため保持
    exit_conditions: List[Condition] = field(default_factory=list)

    # ロング・ショート分離条件（新機能）
    long_entry_conditions: List[Union[Condition, "ConditionGroup"]] = field(default_factory=list)
    short_entry_conditions: List[Union[Condition, "ConditionGroup"]] = field(default_factory=list)

    risk_management: Dict[str, Any] = field(default_factory=dict)
    tpsl_gene: Optional[TPSLGene] = None  # TP/SL遺伝子（GA最適化対象）
    position_sizing_gene: Optional[PositionSizingGene] = (
        None  # ポジションサイジング遺伝子（GA最適化対象）
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """後方互換性のための初期化処理"""
        # 明示的にロング・ショート条件が設定されていない場合のみ、
        # 既存のentry_conditionsを使用（後方互換性）
        # ただし、long_entry_conditionsやshort_entry_conditionsは変更しない

    def get_effective_long_conditions(self) -> List[Union[Condition, "ConditionGroup"]]:
        """有効なロング条件を取得（後方互換性を考慮）"""
        # 明示的にlong_entry_conditionsが設定されていて、かつ空でない場合
        if self.long_entry_conditions:
            return self.long_entry_conditions
        # entry_conditionsがある場合は後方互換性で使用（long_entry_conditionsが空でも）
        elif self.entry_conditions:
            # 後方互換性：既存のentry_conditionsをロング条件として扱う
            # List[Condition] を List[Union[Condition, "ConditionGroup"]] に変換
            return list(self.entry_conditions)  # type: ignore
        else:
            return []

    def get_effective_short_conditions(self) -> List[Union[Condition, "ConditionGroup"]]:
        """有効なショート条件を取得（後方互換性を考慮）"""
        # 明示的にshort_entry_conditionsが設定されていて、かつ空でない場合
        if self.short_entry_conditions:
            return self.short_entry_conditions
        # entry_conditionsがある場合は後方互換性で使用（short_entry_conditionsが空でも）
        elif self.entry_conditions and not self.long_entry_conditions:
            # long_entry_conditionsが設定されていない場合のみ、entry_conditionsをショート条件としても使用
            # List[Condition] を List[Union[Condition, "ConditionGroup"]] に変換
            return list(self.entry_conditions)  # type: ignore
        else:
            return []

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック"""
        # 明示的にロング・ショート条件が設定されている場合のみTrue
        # 空のリストでも明示的に設定されていればTrue（後方互換性のため）
        return (
            self.long_entry_conditions is not None
            and len(self.long_entry_conditions) > 0
        ) or (
            self.short_entry_conditions is not None
            and len(self.short_entry_conditions) > 0
        )

    @property
    def method(self):
        """
        ポジションサイジングメソッドを取得（後方互換性のため）

        Returns:
            PositionSizingMethod: ポジションサイジングメソッド
        """
        if self.position_sizing_gene and hasattr(self.position_sizing_gene, 'method'):
            return self.position_sizing_gene.method
        else:
            # デフォルト値を返す
            from .gene_position_sizing import PositionSizingMethod
            return PositionSizingMethod.FIXED_RATIO

    def validate(self):
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す"""
        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors






