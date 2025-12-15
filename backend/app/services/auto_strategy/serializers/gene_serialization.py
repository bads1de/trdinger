"""
統一遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーション、エンコード・デコードを担当するモジュール。
分割されたコンポーネントを統合して提供します。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class GeneSerializer:
    """
    統一遺伝子シリアライザー

    分割されたコンポーネント（DictConverter, ListEncoder, ListDecoder, JsonConverter）を
    統合して統一的なインターフェースを提供します。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: ConditionGeneratorを使用するか
        """
        # コンポーネントの初期化
        from .dict_converter import DictConverter
        from .json_converter import JsonConverter
        from .list_decoder import ListDecoder
        from .list_encoder import ListEncoder

        self.dict_converter = DictConverter(enable_smart_generation)
        self.list_encoder = ListEncoder()
        self.list_decoder = ListDecoder(enable_smart_generation)
        self.json_converter = JsonConverter(self.dict_converter)

        self.enable_smart_generation = enable_smart_generation

    # DictConverterのメソッドを委譲
    def strategy_gene_to_dict(self, strategy_gene) -> Dict[str, Any]:
        """戦略遺伝子を辞書形式に変換"""
        return self.dict_converter.strategy_gene_to_dict(strategy_gene)

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """指標遺伝子を辞書形式に変換"""
        return self.dict_converter.indicator_gene_to_dict(indicator_gene)

    def dict_to_indicator_gene(self, data: Dict[str, Any]):
        """辞書形式から指標遺伝子を復元"""
        return self.dict_converter.dict_to_indicator_gene(data)

    def condition_to_dict(self, condition) -> Dict[str, Any]:
        """条件を辞書形式に変換"""
        return self.dict_converter.condition_to_dict(condition)

    def condition_or_group_to_dict(self, obj) -> Dict[str, Any]:
        """Condition または ConditionGroup を辞書に変換"""
        return self.dict_converter.condition_or_group_to_dict(obj)

    def tpsl_gene_to_dict(self, tpsl_gene):
        """TP/SL遺伝子を辞書形式に変換"""
        return self.dict_converter.tpsl_gene_to_dict(tpsl_gene)

    def dict_to_tpsl_gene(self, data: Dict[str, Any]):
        """辞書形式からTP/SL遺伝子を復元"""
        return self.dict_converter.dict_to_tpsl_gene(data)

    def position_sizing_gene_to_dict(self, position_sizing_gene):
        """ポジションサイジング遺伝子を辞書形式に変換"""
        return self.dict_converter.position_sizing_gene_to_dict(position_sizing_gene)

    def dict_to_position_sizing_gene(self, data: Dict[str, Any]):
        """辞書形式からポジションサイジング遺伝子を復元"""
        return self.dict_converter.dict_to_position_sizing_gene(data)

    def dict_to_strategy_gene(self, data: Dict[str, Any], strategy_gene_class):
        """辞書形式から戦略遺伝子を復元"""
        return self.dict_converter.dict_to_strategy_gene(data, strategy_gene_class)

    def dict_to_condition(self, data: Dict[str, Any]):
        """辞書形式から条件を復元"""
        return self.dict_converter.dict_to_condition(data)

    # ListEncoderのメソッドを委譲
    def to_list(self, strategy_gene) -> List[float]:
        """戦略遺伝子を固定長の数値リストにエンコード"""
        return self.list_encoder.to_list(strategy_gene)

    # ListDecoderのメソッドを委譲
    def from_list(self, encoded: List[float], strategy_gene_class):
        """数値リストから戦略遺伝子にデコード"""
        return self.list_decoder.from_list(encoded, strategy_gene_class)

    # JsonConverterのメソッドを委譲
    def strategy_gene_to_json(self, strategy_gene) -> str:
        """戦略遺伝子をJSON文字列に変換"""
        return self.json_converter.strategy_gene_to_json(strategy_gene)

    def json_to_strategy_gene(self, json_str: str, strategy_gene_class):
        """JSON文字列から戦略遺伝子を復元"""
        return self.json_converter.json_to_strategy_gene(json_str, strategy_gene_class)

    # 後方互換性のためのエイリアスメソッド
    def decode_list_to_strategy_gene(self, encoded: List[float], strategy_gene_class):
        """数値リストから戦略遺伝子にデコード（旧GeneDecoder.decode_list_to_strategy_gene）"""
        return self.from_list(encoded, strategy_gene_class)

    def encode_strategy_gene_to_list(self, strategy_gene):
        """戦略遺伝子を数値リストにエンコード（旧GeneEncoder.encode_strategy_gene_to_list）"""
        return self.to_list(strategy_gene)





