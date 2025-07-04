"""
戦略遺伝子モデル

遺伝的アルゴリズムで使用する戦略の遺伝子表現を定義します。
責任を分離し、各機能を専用モジュールに委譲します。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
import logging

# 分離されたモジュール
from .gene_validation import GeneValidator
from .gene_serialization import GeneSerializer
from .gene_encoding import GeneEncoder

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
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
