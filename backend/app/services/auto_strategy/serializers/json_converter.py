"""
JSON変換コンポーネント

戦略遺伝子のJSON形式変換機能を担当します。
"""

import json
import logging

logger = logging.getLogger(__name__)


class JsonConverter:
    """
    JSON変換クラス

    戦略遺伝子のJSON形式変換機能を担当します。
    """

    def __init__(self, dict_converter):
        """
        初期化

        Args:
            dict_converter: DictConverterインスタンス
        """
        self.dict_converter = dict_converter

    def strategy_gene_to_json(self, strategy_gene) -> str:
        """
        戦略遺伝子をJSON文字列に変換

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            JSON文字列
        """
        try:
            data = self.dict_converter.strategy_gene_to_dict(strategy_gene)
            return json.dumps(data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"戦略遺伝子JSON変換エラー: {e}")
            raise ValueError(f"戦略遺伝子のJSON変換に失敗: {e}")

    def json_to_strategy_gene(self, json_str: str, strategy_gene_class):
        """
        JSON文字列から戦略遺伝子を復元

        Args:
            json_str: JSON文字列
            strategy_gene_class: StrategyGeneクラス

        Returns:
            戦略遺伝子オブジェクト
        """
        try:
            data = json.loads(json_str)
            return self.dict_converter.dict_to_strategy_gene(data, strategy_gene_class)

        except Exception as e:
            logger.error(f"戦略遺伝子JSON復元エラー: {e}")
            raise ValueError(f"戦略遺伝子のJSON復元に失敗: {e}")





