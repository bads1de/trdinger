"""
統一遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーションを担当するモジュール。
DictConverterとGeneSerializerを統合し、JSON/Dict形式の相互変換を提供します。
"""

import json
import logging
from typing import Any, Dict, Optional

from .strategy_gene_dict_codec import StrategyGeneDictCodec


logger = logging.getLogger(__name__)


class DictConverter:
    """
    戦略遺伝子と辞書形式の相互変換を行うコンバーター

    `StrategyGene` オブジェクトをシリアライズ可能な辞書形式に変換し、
    また辞書から元のクラス構造を再構築します。
    サブ遺伝子（TP/SL、サイジング等）や動的な指標パラメータの
    再帰的な変換を管理します。
    """

    def __init__(self) -> None:
        self._strategy_gene_codec = StrategyGeneDictCodec(self)

    def strategy_gene_to_dict(self, strategy_gene: Any) -> Dict[str, Any]:
        """戦略遺伝子オブジェクトをシリアライズ可能な辞書形式に変換"""
        return self._strategy_gene_codec.strategy_gene_to_dict(strategy_gene)

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """
        指標遺伝子を辞書形式に変換

        Args:
            indicator_gene: 指標遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            result = {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }
            # timeframe が設定されている場合のみ含める（後方互換性）
            if indicator_gene.timeframe is not None:
                result["timeframe"] = indicator_gene.timeframe
            return result

        except Exception as e:
            logger.error(f"指標遺伝子辞書変換エラー: {e}")
            raise ValueError(f"指標遺伝子の辞書変換に失敗: {e}")

    def dict_to_indicator_gene(self, data: Dict[str, Any]) -> "IndicatorGene":
        """
        辞書形式から指標遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            指標遺伝子オブジェクト
        """
        try:
            # IndicatorGeneクラスを動的にインポート
            from ..genes import IndicatorGene

            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
                timeframe=data.get("timeframe"),  # None の場合はデフォルトTFを使用
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
            辞書形式의データ
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

    def condition_or_group_to_dict(self, obj) -> Dict[str, Any]:
        """Condition または ConditionGroup を辞書に変換"""
        try:
            from ..genes import Condition, ConditionGroup

            if isinstance(obj, ConditionGroup):
                return {
                    "type": "GROUP",
                    "operator": obj.operator,
                    "conditions": [
                        self.condition_or_group_to_dict(c) for c in obj.conditions
                    ],
                }
            elif isinstance(obj, Condition) or hasattr(obj, "left_operand"):
                return self.condition_to_dict(obj)
            else:
                raise TypeError(f"未知の条件型: {type(obj)}")
        except Exception as e:
            logger.error(f"条件/グループ辞書変換エラー: {e}")
            raise ValueError(f"条件の辞書変換に失敗: {e}")

    def tpsl_gene_to_dict(self, tpsl_gene) -> Optional[Dict[str, Any]]:
        """
        TP/SL遺伝子を辞書形式に変換

        Args:
            tpsl_gene: TP/SL遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if tpsl_gene is None:
                return None

            return tpsl_gene.to_dict()

        except Exception as e:
            logger.error(f"TP/SL遺伝子辞書変換エラー: {e}")
            raise ValueError(f"TP/SL遺伝子の辞書変換に失敗: {e}")

    def dict_to_tpsl_gene(self, data: Dict[str, Any]) -> Optional["TPSLGene"]:
        """
        辞書形式からTP/SL遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            TP/SL遺伝子オブジェクト
        """
        try:
            if data is None:
                return None

            from ..genes import TPSLGene

            return TPSLGene.from_dict(data)

        except Exception as e:
            logger.error(f"TP/SL遺伝子復元エラー: {e}")
            raise ValueError(f"TP/SL遺伝子の復元に失敗: {e}")

    def position_sizing_gene_to_dict(
        self, position_sizing_gene
    ) -> Optional[Dict[str, Any]]:
        """
        ポジションサイジング遺伝子を辞書形式に変換

        Args:
            position_sizing_gene: ポジションサイジング遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if position_sizing_gene is None:
                return None

            return position_sizing_gene.to_dict()

        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子辞書変換エラー: {e}")
            raise ValueError(f"ポジションサイジング遺伝子の辞書変換に失敗: {e}")

    def dict_to_position_sizing_gene(
        self, data: Dict[str, Any]
    ) -> Optional["PositionSizingGene"]:
        """
        辞書形式からポジションサイジング遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            ポジションサイジング遺伝子オブジェクト
        """
        try:
            if data is None:
                return None

            # PositionSizingGeneクラスを動的にインポート
            from ..genes import PositionSizingGene

            return PositionSizingGene.from_dict(data)

        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子復元エラー: {e}")
            raise ValueError(f"ポジションサイジング遺伝子の復元に失敗: {e}")

    def entry_gene_to_dict(self, entry_gene) -> Optional[Dict[str, Any]]:
        """
        エントリー遺伝子を辞書形式に変換

        Args:
            entry_gene: エントリー遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if entry_gene is None:
                return None

            return entry_gene.to_dict()

        except Exception as e:
            logger.error(f"エントリー遺伝子辞書変換エラー: {e}")
            raise ValueError(f"エントリー遺伝子の辞書変換に失敗: {e}")

    def dict_to_entry_gene(self, data: Dict[str, Any]) -> Optional["EntryGene"]:
        """
        辞書形式からエントリー遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            エントリー遺伝子オブジェクト
        """
        try:
            if data is None:
                return None

            return EntryGene.from_dict(data)

        except Exception as e:
            logger.error(f"エントリー遺伝子復元エラー: {e}")
            raise ValueError(f"エントリー遺伝子の復元に失敗: {e}")

    def _clean_risk_management(self, risk_management: Dict[str, Any]) -> Dict[str, Any]:
        """risk_managementからTP/SL関連の設定を除外"""
        return self._strategy_gene_codec._clean_risk_management(risk_management)

    def dict_to_strategy_gene(self, data: Any, strategy_gene_class: Any):
        """辞書形式のデータから戦略遺伝子オブジェクトを復元"""
        return self._strategy_gene_codec.dict_to_strategy_gene(
            data, strategy_gene_class
        )

    def dict_to_condition(self, data: Dict[str, Any]) -> "Condition":
        """辞書形式から条件を復元"""
        return self._strategy_gene_codec.dict_to_condition(data)

    def stateful_condition_to_dict(
        self, stateful_condition
    ) -> Optional[Dict[str, Any]]:
        """
        StatefulCondition を辞書形式に変換

        Args:
            stateful_condition: StatefulCondition オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if stateful_condition is None:
                return None

            return {
                "trigger_condition": self.condition_to_dict(
                    stateful_condition.trigger_condition
                ),
                "follow_condition": self.condition_to_dict(
                    stateful_condition.follow_condition
                ),
                "lookback_bars": stateful_condition.lookback_bars,
                "cooldown_bars": stateful_condition.cooldown_bars,
                "direction": getattr(stateful_condition, "direction", "long"),
                "enabled": stateful_condition.enabled,
            }

        except Exception as e:
            logger.error(f"StatefulCondition辞書変換エラー: {e}")
            raise ValueError(f"StatefulConditionの辞書変換に失敗: {e}")

    def dict_to_stateful_condition(self, data: Dict[str, Any]):
        """
        辞書形式から StatefulCondition を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            StatefulCondition オブジェクト
        """
        try:
            if data is None:
                return None

            from ..genes.conditions import StatefulCondition

            trigger_condition = self.dict_to_condition(data["trigger_condition"])
            follow_condition = self.dict_to_condition(data["follow_condition"])

            return StatefulCondition(
                trigger_condition=trigger_condition,
                follow_condition=follow_condition,
                lookback_bars=data.get("lookback_bars", 5),
                cooldown_bars=data.get("cooldown_bars", 0),
                direction=data.get("direction", "long"),
                enabled=data.get("enabled", True),
            )

        except Exception as e:
            logger.error(f"StatefulCondition復元エラー: {e}")
            raise ValueError(f"StatefulConditionの復元に失敗: {e}")


class GeneSerializer(DictConverter):
    """
    戦略遺伝子のシリアライゼーション補助クラス

    辞書変換は `DictConverter` を継承して直接提供し、JSON 変換と
    DEAP 個体のリスト復元だけを追加します。
    """

    def __init__(self):
        """後方互換のために self を dict_converter として公開します。"""
        super().__init__()
        self.dict_converter = self

    def from_list(self, individual_list: list, strategy_gene_class: Any):
        """
        リスト形式（DEAP個体）から戦略遺伝子を復元します。

        Args:
            individual_list: DEAPの個体（StrategyGeneを継承したリスト、または[StrategyGene]）
            strategy_gene_class: 復元に使用するクラス

        Returns:
            復元されたStrategyGene
        """
        from ..genes import StrategyGene

        if not individual_list:
            return None

        # 1. 既にStrategyGeneのインスタンスである場合
        if isinstance(individual_list, StrategyGene):
            return individual_list

        # 2. リストの最初の要素がStrategyGeneである場合
        if len(individual_list) > 0 and isinstance(individual_list[0], StrategyGene):
            return individual_list[0]

        # 3. 属性アクセスを試行（DEAP個体はStrategyGeneを継承している場合がある）
        if hasattr(individual_list, "indicators"):
            return individual_list

        return None

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
        except Exception as e:
            logger.error(f"戦略遺伝子JSON復元エラー: {e}")
            raise ValueError(f"戦略遺伝子のJSON復元に失敗: {e}")
