"""
戦略遺伝子モデル（リファクタリング版）

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple
import logging

# 分離されたモジュール
from .gene_validation import GeneValidator
from .gene_serialization import GeneSerializer
from .gene_encoding import GeneEncoder

logger = logging.getLogger(__name__)


@dataclass
class IndicatorGene:
    """
    指標遺伝子（リファクタリング版）
    
    単一のテクニカル指標の設定を表現します。
    """
    type: str  # 指標タイプ（例: "SMA", "RSI", "MACD"）
    parameters: Dict[str, Any] = field(default_factory=dict)  # 指標パラメータ
    enabled: bool = True  # 指標の有効/無効

    def validate(self) -> bool:
        """指標遺伝子の妥当性を検証（リファクタリング版）"""
        validator = GeneValidator()
        return validator.validate_indicator_gene(self)


@dataclass
class Condition:
    """
    条件（リファクタリング版）
    
    エントリー・イグジット条件を表現します。
    """
    left_operand: Union[str, float]  # 左オペランド
    operator: str  # 演算子
    right_operand: Union[str, float]  # 右オペランド

    def validate(self) -> bool:
        """条件の妥当性を検証（リファクタリング版）"""
        validator = GeneValidator()
        return validator.validate_condition(self)

    def _is_indicator_name(self, name: str) -> bool:
        """指標名かどうかを判定（リファクタリング版）"""
        validator = GeneValidator()
        return validator._is_indicator_name(name)


@dataclass
class StrategyGene:
    """
    戦略遺伝子（リファクタリング版）
    
    完全な取引戦略を表現する遺伝子です。
    """
    # 制約定数
    MAX_INDICATORS = 5  # 最大指標数

    # 戦略構成要素
    id: str = ""
    indicators: List[IndicatorGene] = field(default_factory=list)
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証（リファクタリング版）"""
        validator = GeneValidator()
        return validator.validate_strategy_gene(self)

    def to_dict(self) -> Dict[str, Any]:
        """戦略遺伝子を辞書形式に変換（リファクタリング版）"""
        serializer = GeneSerializer()
        return serializer.strategy_gene_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGene":
        """辞書形式から戦略遺伝子を復元（リファクタリング版）"""
        serializer = GeneSerializer()
        return serializer.dict_to_strategy_gene(data, cls)

    def to_json(self) -> str:
        """戦略遺伝子をJSON文字列に変換（リファクタリング版）"""
        serializer = GeneSerializer()
        return serializer.strategy_gene_to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyGene":
        """JSON文字列から戦略遺伝子を復元（リファクタリング版）"""
        serializer = GeneSerializer()
        return serializer.json_to_strategy_gene(json_str, cls)


# v1仕様用のエンコード/デコード関数（リファクタリング版）


def encode_gene_to_list(gene: StrategyGene) -> List[float]:
    """戦略遺伝子をGA用数値リストにエンコード（リファクタリング版）"""
    encoder = GeneEncoder()
    return encoder.encode_strategy_gene_to_list(gene)


def decode_list_to_gene(encoded: List[float]) -> StrategyGene:
    """GA用数値リストから戦略遺伝子にデコード（リファクタリング版）"""
    encoder = GeneEncoder()
    return encoder.decode_list_to_strategy_gene(encoded, StrategyGene)


# 全ての関数実装は分離されたモジュールに移動されました
# - gene_validation.py: バリデーション機能
# - gene_serialization.py: シリアライゼーション機能  
# - gene_encoding.py: GA用エンコード/デコード機能
