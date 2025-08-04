"""
戦略遺伝子モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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

    def normalize_parameters(self) -> Dict[str, Any]:
        """パラメータをJSON形式に正規化"""
        try:
            from app.services.indicators.config import indicator_registry

            config = indicator_registry.get_indicator_config(self.type)
            if config:
                # 設定に基づいてパラメータを正規化
                normalized = {}
                for param_name, param_config in config.parameters.items():
                    value = self.parameters.get(param_name, param_config.default_value)
                    normalized[param_name] = value
                return normalized

            # フォールバック: そのまま返す
            return self.parameters.copy()

        except ImportError:
            return self.parameters.copy()

    @classmethod
    def create_from_json_config(
        cls, json_config: Dict[str, Any], enabled: bool = True
    ) -> "IndicatorGene":
        """JSON形式の設定から指標遺伝子を作成"""
        indicator_type = json_config.get("indicator", "")
        parameters = json_config.get("parameters", {})

        return cls(
            type=indicator_type,
            parameters=parameters,
            enabled=enabled,
            json_config=json_config,
        )


@dataclass
class Condition:
    """
    条件

    エントリー・イグジット条件を表現します。
    """

    left_operand: Union[Dict[str, Any], str, float]  # 左オペランド
    operator: str  # 演算子
    right_operand: Union[Dict[str, Any], str, float]  # 右オペランド

    def validate(self) -> bool:
        """条件の妥当性を検証"""
        validator = GeneValidator()
        return validator.validate_condition(self)

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
    long_entry_conditions: List[Condition] = field(default_factory=list)
    short_entry_conditions: List[Condition] = field(default_factory=list)

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

    def get_effective_long_conditions(self) -> List[Condition]:
        """有効なロング条件を取得（後方互換性を考慮）"""
        # 明示的にlong_entry_conditionsが設定されていて、かつ空でない場合
        if self.long_entry_conditions:
            return self.long_entry_conditions
        # entry_conditionsがある場合は後方互換性で使用（long_entry_conditionsが空でも）
        elif self.entry_conditions:
            # 後方互換性：既存のentry_conditionsをロング条件として扱う
            return self.entry_conditions
        else:
            return []

    def get_effective_short_conditions(self) -> List[Condition]:
        """有効なショート条件を取得（後方互換性を考慮）"""
        # 明示的にshort_entry_conditionsが設定されていて、かつ空でない場合
        if self.short_entry_conditions:
            return self.short_entry_conditions
        # entry_conditionsがある場合は後方互換性で使用（short_entry_conditionsが空でも）
        elif self.entry_conditions and not self.long_entry_conditions:
            # long_entry_conditionsが設定されていない場合のみ、entry_conditionsをショート条件としても使用
            return self.entry_conditions
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

    def validate(self) -> tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証"""
        validator = GeneValidator()
        return validator.validate_strategy_gene(self)


def encode_gene_to_list(gene: StrategyGene) -> List[float]:
    """戦略遺伝子をGA用数値リストにエンコード"""
    from .gene_encoding import GeneEncoder

    encoder = GeneEncoder()
    return encoder.encode_strategy_gene_to_list(gene)


def decode_list_to_gene(encoded: List[float]) -> StrategyGene:
    """GA用数値リストから戦略遺伝子にデコード"""
    from .gene_encoding import GeneEncoder

    encoder = GeneEncoder()
    return encoder.decode_list_to_strategy_gene(encoded, StrategyGene)


def crossover_strategy_genes(
    parent1: StrategyGene, parent2: StrategyGene
) -> tuple[StrategyGene, StrategyGene]:
    """
    戦略遺伝子の交叉

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な交叉を実行します。

    Args:
        parent1: 親1の戦略遺伝子
        parent2: 親2の戦略遺伝子

    Returns:
        交叉後の子1、子2の戦略遺伝子のタプル
    """
    from ..operators.genetic_operators import crossover_strategy_genes as _crossover

    return _crossover(parent1, parent2)


def mutate_strategy_gene(
    gene: StrategyGene, mutation_rate: float = 0.1
) -> StrategyGene:
    """
    戦略遺伝子の突然変異

    指標遺伝子、条件、TP/SL遺伝子すべてを含む完全な突然変異を実行します。

    Args:
        gene: 突然変異対象の戦略遺伝子
        mutation_rate: 突然変異率

    Returns:
        突然変異後の戦略遺伝子
    """
    from ..operators.genetic_operators import mutate_strategy_gene as _mutate

    return _mutate(gene, mutation_rate)
