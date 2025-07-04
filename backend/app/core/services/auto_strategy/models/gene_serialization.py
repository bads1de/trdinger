"""
遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーションを担当するモジュール。
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GeneSerializer:
    """
    遺伝子シリアライザー
    
    戦略遺伝子のシリアライゼーション・デシリアライゼーションを担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def strategy_gene_to_dict(self, strategy_gene) -> Dict[str, Any]:
        """
        戦略遺伝子を辞書形式に変換
        
        Args:
            strategy_gene: 戦略遺伝子オブジェクト
            
        Returns:
            辞書形式のデータ
        """
        try:
            return {
                "id": strategy_gene.id,
                "indicators": [
                    self.indicator_gene_to_dict(ind) for ind in strategy_gene.indicators
                ],
                "entry_conditions": [
                    self.condition_to_dict(cond) for cond in strategy_gene.entry_conditions
                ],
                "exit_conditions": [
                    self.condition_to_dict(cond) for cond in strategy_gene.exit_conditions
                ],
                "risk_management": strategy_gene.risk_management,
                "metadata": strategy_gene.metadata,
            }

        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def dict_to_strategy_gene(self, data: Dict[str, Any], strategy_gene_class):
        """
        辞書形式から戦略遺伝子を復元
        
        Args:
            data: 辞書形式のデータ
            strategy_gene_class: StrategyGeneクラス
            
        Returns:
            戦略遺伝子オブジェクト
        """
        try:
            indicators = [
                self.dict_to_indicator_gene(ind_data)
                for ind_data in data.get("indicators", [])
            ]

            entry_conditions = [
                self.dict_to_condition(cond_data)
                for cond_data in data.get("entry_conditions", [])
            ]

            exit_conditions = [
                self.dict_to_condition(cond_data)
                for cond_data in data.get("exit_conditions", [])
            ]

            return strategy_gene_class(
                id=data.get("id", ""),
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=data.get("risk_management", {}),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"戦略遺伝子復元エラー: {e}")
            raise ValueError(f"戦略遺伝子の復元に失敗: {e}")

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """
        指標遺伝子を辞書形式に変換
        
        Args:
            indicator_gene: 指標遺伝子オブジェクト
            
        Returns:
            辞書形式のデータ
        """
        try:
            return {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }

        except Exception as e:
            logger.error(f"指標遺伝子辞書変換エラー: {e}")
            raise ValueError(f"指標遺伝子の辞書変換に失敗: {e}")

    def dict_to_indicator_gene(self, data: Dict[str, Any]):
        """
        辞書形式から指標遺伝子を復元
        
        Args:
            data: 辞書形式のデータ
            
        Returns:
            指標遺伝子オブジェクト
        """
        try:
            # IndicatorGeneクラスを動的にインポート
            from .strategy_gene import IndicatorGene
            
            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
            )

        except Exception as e:
            logger.error(f"指標遺伝子復元エラー: {e}")
            raise ValueError(f"指標遺伝子の復元に失敗: {e}")

    def condition_to_dict(self, condition) -> Dict[str, Any]:
        """
        条件を辞書形式に変換
        
        Args:
            condition: 条件オブジェクト
            
        Returns:
            辞書形式のデータ
        """
        try:
            return {
                "left_operand": condition.left_operand,
                "operator": condition.operator,
                "right_operand": condition.right_operand,
            }

        except Exception as e:
            logger.error(f"条件辞書変換エラー: {e}")
            raise ValueError(f"条件の辞書変換に失敗: {e}")

    def dict_to_condition(self, data: Dict[str, Any]):
        """
        辞書形式から条件を復元
        
        Args:
            data: 辞書形式のデータ
            
        Returns:
            条件オブジェクト
        """
        try:
            # Conditionクラスを動的にインポート
            from .strategy_gene import Condition
            
            return Condition(
                left_operand=data["left_operand"],
                operator=data["operator"],
                right_operand=data["right_operand"],
            )

        except Exception as e:
            logger.error(f"条件復元エラー: {e}")
            raise ValueError(f"条件の復元に失敗: {e}")

    def strategy_gene_to_json(self, strategy_gene) -> str:
        """
        戦略遺伝子をJSON文字列に変換
        
        Args:
            strategy_gene: 戦略遺伝子オブジェクト
            
        Returns:
            JSON文字列
        """
        try:
            data = self.strategy_gene_to_dict(strategy_gene)
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
            return self.dict_to_strategy_gene(data, strategy_gene_class)

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析エラー: {e}")
            raise ValueError(f"無効なJSON: {e}")
        except Exception as e:
            logger.error(f"戦略遺伝子JSON復元エラー: {e}")
            raise ValueError(f"戦略遺伝子のJSON復元に失敗: {e}")

    def serialize_for_database(self, strategy_gene) -> Dict[str, Any]:
        """
        データベース保存用にシリアライズ
        
        Args:
            strategy_gene: 戦略遺伝子オブジェクト
            
        Returns:
            データベース保存用の辞書
        """
        try:
            data = self.strategy_gene_to_dict(strategy_gene)
            
            # データベース保存用の追加情報
            data["serialization_version"] = "1.0"
            data["serialization_timestamp"] = self._get_current_timestamp()
            
            return data

        except Exception as e:
            logger.error(f"データベースシリアライゼーションエラー: {e}")
            raise ValueError(f"データベース用シリアライゼーションに失敗: {e}")

    def deserialize_from_database(self, data: Dict[str, Any], strategy_gene_class):
        """
        データベースからデシリアライズ
        
        Args:
            data: データベースからの辞書データ
            strategy_gene_class: StrategyGeneクラス
            
        Returns:
            戦略遺伝子オブジェクト
        """
        try:
            # バージョン情報をチェック
            version = data.get("serialization_version", "1.0")
            if version != "1.0":
                logger.warning(f"異なるシリアライゼーションバージョン: {version}")
            
            # 不要なメタデータを除去
            clean_data = {k: v for k, v in data.items() 
                         if k not in ["serialization_version", "serialization_timestamp"]}
            
            return self.dict_to_strategy_gene(clean_data, strategy_gene_class)

        except Exception as e:
            logger.error(f"データベースデシリアライゼーションエラー: {e}")
            raise ValueError(f"データベースからのデシリアライゼーションに失敗: {e}")

    def serialize_for_api(self, strategy_gene) -> Dict[str, Any]:
        """
        API応答用にシリアライズ
        
        Args:
            strategy_gene: 戦略遺伝子オブジェクト
            
        Returns:
            API応答用の辞書
        """
        try:
            data = self.strategy_gene_to_dict(strategy_gene)
            
            # API応答用の追加情報
            data["api_version"] = "1.0"
            data["response_timestamp"] = self._get_current_timestamp()
            
            # 統計情報を追加
            data["statistics"] = {
                "indicator_count": len(strategy_gene.indicators),
                "enabled_indicator_count": len([ind for ind in strategy_gene.indicators if ind.enabled]),
                "entry_condition_count": len(strategy_gene.entry_conditions),
                "exit_condition_count": len(strategy_gene.exit_conditions),
            }
            
            return data

        except Exception as e:
            logger.error(f"APIシリアライゼーションエラー: {e}")
            raise ValueError(f"API用シリアライゼーションに失敗: {e}")

    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except Exception as e:
            logger.error(f"タイムスタンプ取得エラー: {e}")
            return ""

    def validate_serialized_data(self, data: Dict[str, Any]) -> bool:
        """
        シリアライズされたデータの妥当性を検証
        
        Args:
            data: シリアライズされたデータ
            
        Returns:
            妥当性（True/False）
        """
        try:
            required_keys = ["id", "indicators", "entry_conditions", "exit_conditions"]
            
            # 必須キーの存在確認
            for key in required_keys:
                if key not in data:
                    logger.error(f"必須キーが不足: {key}")
                    return False
            
            # データ型の確認
            if not isinstance(data["indicators"], list):
                logger.error("indicatorsがリストではありません")
                return False
            
            if not isinstance(data["entry_conditions"], list):
                logger.error("entry_conditionsがリストではありません")
                return False
            
            if not isinstance(data["exit_conditions"], list):
                logger.error("exit_conditionsがリストではありません")
                return False
            
            return True

        except Exception as e:
            logger.error(f"シリアライズデータ検証エラー: {e}")
            return False
