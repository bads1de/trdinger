"""
戦略遺伝子モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

# import logging

# 分離されたモジュール
from .gene_validation import GeneValidator
from .gene_serialization import GeneSerializer
from .gene_encoding import GeneEncoder
from .tpsl_gene import TPSLGene

# logger = logging.getLogger(__name__)


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
            from app.core.services.indicators.config import indicator_registry

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
            from app.core.services.indicators.config import indicator_registry

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """後方互換性のための初期化処理"""
        # 明示的にロング・ショート条件が設定されていない場合のみ、
        # 既存のentry_conditionsを使用（後方互換性）
        # ただし、long_entry_conditionsやshort_entry_conditionsは変更しない
        pass

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

    def to_dict(self) -> Dict[str, Any]:
        """戦略遺伝子を辞書形式に変換"""
        serializer = GeneSerializer()
        return serializer.strategy_gene_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGene":
        """辞書形式から戦略遺伝子を復元"""
        serializer = GeneSerializer()
        return serializer.dict_to_strategy_gene(data, cls)

    def to_json(self) -> str:
        """戦略遺伝子をJSON文字列に変換"""
        serializer = GeneSerializer()
        return serializer.strategy_gene_to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyGene":
        """JSON文字列から戦略遺伝子を復元"""
        serializer = GeneSerializer()
        return serializer.json_to_strategy_gene(json_str, cls)


def encode_gene_to_list(gene: StrategyGene) -> List[float]:
    """戦略遺伝子をGA用数値リストにエンコード"""
    encoder = GeneEncoder()
    return encoder.encode_strategy_gene_to_list(gene)


def decode_list_to_gene(encoded: List[float]) -> StrategyGene:
    """GA用数値リストから戦略遺伝子にデコード"""
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
    import random
    import uuid
    from .tpsl_gene import crossover_tpsl_genes

    try:
        # 指標遺伝子の交叉（単純な一点交叉）
        crossover_point = random.randint(
            1, min(len(parent1.indicators), len(parent2.indicators))
        )

        child1_indicators = (
            parent1.indicators[:crossover_point] + parent2.indicators[crossover_point:]
        )
        child2_indicators = (
            parent2.indicators[:crossover_point] + parent1.indicators[crossover_point:]
        )

        # 最大指標数制限
        max_indicators = getattr(parent1, "MAX_INDICATORS", 5)
        child1_indicators = child1_indicators[:max_indicators]
        child2_indicators = child2_indicators[:max_indicators]

        # 条件の交叉（ランダム選択）
        if random.random() < 0.5:
            child1_entry = parent1.entry_conditions.copy()
            child2_entry = parent2.entry_conditions.copy()
        else:
            child1_entry = parent2.entry_conditions.copy()
            child2_entry = parent1.entry_conditions.copy()

        if random.random() < 0.5:
            child1_exit = parent1.exit_conditions.copy()
            child2_exit = parent2.exit_conditions.copy()
        else:
            child1_exit = parent2.exit_conditions.copy()
            child2_exit = parent1.exit_conditions.copy()

        # リスク管理設定の交叉（平均値）
        child1_risk = {}
        child2_risk = {}

        all_keys = set(parent1.risk_management.keys()) | set(
            parent2.risk_management.keys()
        )
        for key in all_keys:
            val1 = parent1.risk_management.get(key, 0)
            val2 = parent2.risk_management.get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                child1_risk[key] = (val1 + val2) / 2
                child2_risk[key] = (val1 + val2) / 2
            else:
                child1_risk[key] = val1 if random.random() < 0.5 else val2
                child2_risk[key] = val2 if random.random() < 0.5 else val1

        # TP/SL遺伝子の交叉
        child1_tpsl = None
        child2_tpsl = None

        if parent1.tpsl_gene and parent2.tpsl_gene:
            child1_tpsl, child2_tpsl = crossover_tpsl_genes(
                parent1.tpsl_gene, parent2.tpsl_gene
            )
        elif parent1.tpsl_gene:
            child1_tpsl = parent1.tpsl_gene
            child2_tpsl = parent1.tpsl_gene  # コピー
        elif parent2.tpsl_gene:
            child1_tpsl = parent2.tpsl_gene
            child2_tpsl = parent2.tpsl_gene  # コピー

        # メタデータの継承
        child1_metadata = parent1.metadata.copy()
        child1_metadata["crossover_parent1"] = parent1.id
        child1_metadata["crossover_parent2"] = parent2.id

        child2_metadata = parent2.metadata.copy()
        child2_metadata["crossover_parent1"] = parent1.id
        child2_metadata["crossover_parent2"] = parent2.id

        # 子遺伝子の作成
        child1 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child1_indicators,
            entry_conditions=child1_entry,
            exit_conditions=child1_exit,
            risk_management=child1_risk,
            tpsl_gene=child1_tpsl,
            metadata=child1_metadata,
        )

        child2 = StrategyGene(
            id=str(uuid.uuid4()),
            indicators=child2_indicators,
            entry_conditions=child2_entry,
            exit_conditions=child2_exit,
            risk_management=child2_risk,
            tpsl_gene=child2_tpsl,
            metadata=child2_metadata,
        )

        return child1, child2

    except Exception as e:
        # logger.error(f"戦略遺伝子交叉エラー: {e}")
        # エラー時は親をそのまま返す
        return parent1, parent2


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
    import random
    import copy
    from .tpsl_gene import mutate_tpsl_gene, create_random_tpsl_gene

    try:
        # 深いコピーを作成
        mutated = copy.deepcopy(gene)

        # 指標遺伝子の突然変異
        for i, indicator in enumerate(mutated.indicators):
            if random.random() < mutation_rate:
                # パラメータの突然変異
                for param_name, param_value in indicator.parameters.items():
                    if (
                        isinstance(param_value, (int, float))
                        and random.random() < mutation_rate
                    ):
                        if param_name == "period":
                            # 期間パラメータの場合
                            mutated.indicators[i].parameters[param_name] = max(
                                1, min(200, int(param_value * random.uniform(0.8, 1.2)))
                            )
                        else:
                            # その他の数値パラメータ
                            mutated.indicators[i].parameters[param_name] = (
                                param_value * random.uniform(0.8, 1.2)
                            )

        # 指標の追加・削除（低確率）
        if random.random() < mutation_rate * 0.3:
            max_indicators = getattr(gene, "MAX_INDICATORS", 5)

            if len(mutated.indicators) < max_indicators and random.random() < 0.5:
                # 新しい指標を追加
                from ..generators.random_gene_generator import RandomGeneGenerator
                from ..models.ga_config import GAConfig

                generator = RandomGeneGenerator(GAConfig())
                new_indicators = generator._generate_random_indicators()
                if new_indicators:
                    mutated.indicators.append(random.choice(new_indicators))

            elif len(mutated.indicators) > 1 and random.random() < 0.5:
                # 指標を削除
                mutated.indicators.pop(random.randint(0, len(mutated.indicators) - 1))

        # 条件の突然変異（低確率）
        if random.random() < mutation_rate * 0.5:
            # エントリー条件の変更
            if mutated.entry_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.entry_conditions) - 1)
                condition = mutated.entry_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        if random.random() < mutation_rate * 0.5:
            # エグジット条件の変更
            if mutated.exit_conditions and random.random() < 0.5:
                condition_idx = random.randint(0, len(mutated.exit_conditions) - 1)
                condition = mutated.exit_conditions[condition_idx]

                # オペレーターの変更
                operators = [">", "<", ">=", "<=", "=="]
                condition.operator = random.choice(operators)

        # リスク管理設定の突然変異
        for key, value in mutated.risk_management.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                if key == "position_size":
                    # ポジションサイズの場合
                    mutated.risk_management[key] = max(
                        0.01, min(1.0, value * random.uniform(0.8, 1.2))
                    )
                else:
                    # その他の数値設定
                    mutated.risk_management[key] = value * random.uniform(0.8, 1.2)

        # TP/SL遺伝子の突然変異
        if mutated.tpsl_gene:
            if random.random() < mutation_rate:
                mutated.tpsl_gene = mutate_tpsl_gene(mutated.tpsl_gene, mutation_rate)
        else:
            # TP/SL遺伝子が存在しない場合、低確率で新規作成
            if random.random() < mutation_rate * 0.2:
                mutated.tpsl_gene = create_random_tpsl_gene()

        # メタデータの更新
        mutated.metadata["mutated"] = True
        mutated.metadata["mutation_rate"] = mutation_rate

        return mutated

    except Exception as e:
        # logger.error(f"戦略遺伝子突然変異エラー: {e}")
        # エラー時は元の遺伝子をそのまま返す
        return gene
