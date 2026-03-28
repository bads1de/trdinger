"""
StrategyGene の dict encode/decode helper。
"""

import logging
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StrategyGeneDictCodec:
    """StrategyGene の辞書変換ロジックを担当する helper。"""

    def __init__(self, converter: Any) -> None:
        self.converter = converter

    def strategy_gene_to_dict(self, strategy_gene: Any) -> Dict[str, Any]:
        """戦略遺伝子オブジェクトをシリアライズ可能な辞書形式に変換。"""
        try:
            clean_risk_management = self._clean_risk_management(
                strategy_gene.risk_management
            )

            result = {
                "id": strategy_gene.id,
                "indicators": [
                    self.converter.indicator_gene_to_dict(ind)
                    for ind in strategy_gene.indicators
                ],
                "long_entry_conditions": [
                    self.converter.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.long_entry_conditions
                ],
                "short_entry_conditions": [
                    self.converter.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.short_entry_conditions
                ],
                "risk_management": clean_risk_management,
                "stateful_conditions": [
                    self.converter.stateful_condition_to_dict(sc)
                    for sc in getattr(strategy_gene, "stateful_conditions", [])
                ],
                "tool_genes": [
                    tg.to_dict() for tg in getattr(strategy_gene, "tool_genes", [])
                ],
                "metadata": strategy_gene.metadata,
            }

            sub_gene_fields = [
                "tpsl_gene",
                "long_tpsl_gene",
                "short_tpsl_gene",
                "position_sizing_gene",
                "entry_gene",
                "long_entry_gene",
                "short_entry_gene",
            ]
            for field in sub_gene_fields:
                gene_obj = getattr(strategy_gene, field, None)
                result[field] = gene_obj.to_dict() if gene_obj else None

            return result

        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def dict_to_strategy_gene(self, data: Any, strategy_gene_class: Any):
        """辞書形式のデータから戦略遺伝子オブジェクトを復元。"""
        try:
            if isinstance(data, strategy_gene_class):
                return data

            if hasattr(data, "indicators") and hasattr(data, "long_entry_conditions"):
                return data

            if not isinstance(data, dict):
                logger.error(
                    "dict_to_strategy_gene に渡されたデータの型が不正です: %s",
                    type(data).__name__,
                )
                return strategy_gene_class.create_default()

            if not data:
                logger.warning(
                    "戦略遺伝子データが空です。デフォルト戦略遺伝子を返します。"
                )
                return strategy_gene_class.create_default()

            indicators = [
                self.converter.dict_to_indicator_gene(ind_data)
                for ind_data in data.get("indicators", [])
            ]

            from ..genes import ConditionGroup

            def parse_condition_or_group(cond_data):
                if not isinstance(cond_data, dict):
                    return None
                if cond_data.get("type") in ("GROUP", "OR_GROUP"):
                    conditions = [
                        parse_condition_or_group(c)
                        for c in cond_data.get("conditions", [])
                    ]
                    operator = (
                        "OR"
                        if cond_data.get("type") == "OR_GROUP"
                        else cond_data.get("operator", "OR")
                    )
                    return ConditionGroup(operator=operator, conditions=conditions)
                return self.dict_to_condition(cond_data)

            long_entry_conditions = [
                parse_condition_or_group(c)
                for c in data.get("long_entry_conditions", [])
            ]
            short_entry_conditions = [
                parse_condition_or_group(c)
                for c in data.get("short_entry_conditions", [])
            ]

            from ..genes import EntryGene, PositionSizingGene, TPSLGene
            from ..genes.tool import ToolGene

            sub_genes = {}
            mapping = {
                "tpsl_gene": TPSLGene,
                "long_tpsl_gene": TPSLGene,
                "short_tpsl_gene": TPSLGene,
                "position_sizing_gene": PositionSizingGene,
                "entry_gene": EntryGene,
                "long_entry_gene": EntryGene,
                "short_entry_gene": EntryGene,
            }

            for field, cls in mapping.items():
                gene_data = data.get(field)
                sub_genes[field] = cls.from_dict(gene_data) if gene_data else None  # type: ignore[attr-defined]

            stateful_conditions = [
                self.converter.dict_to_stateful_condition(sc_data)
                for sc_data in data.get("stateful_conditions", [])
                if sc_data
            ]

            tool_genes = [
                ToolGene.from_dict(tg) for tg in data.get("tool_genes", []) if tg
            ]

            return strategy_gene_class(
                id=data.get("id", str(uuid.uuid4())),
                indicators=indicators,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                stateful_conditions=stateful_conditions,
                tool_genes=tool_genes,
                risk_management=data.get("risk_management", {"position_size": 0.1}),
                metadata=data.get("metadata", {}),
                **sub_genes,
            )

        except Exception as e:
            logger.error(f"戦略遺伝子辞書復元エラー: {e}")
            return strategy_gene_class.create_default()

    def dict_to_condition(self, data: Dict[str, Any]):
        """辞書形式から条件を復元。"""
        try:
            from ..genes import Condition

            return Condition(
                left_operand=data["left_operand"],
                operator=data["operator"],
                right_operand=data["right_operand"],
            )

        except Exception as e:
            logger.error(f"条件辞書復元エラー: {e}")
            raise ValueError(f"条件の復元に失敗: {e}")

    def _clean_risk_management(self, risk_management: Dict[str, Any]) -> Dict[str, Any]:
        """risk_management から TP/SL 関連の設定を除外。"""
        tpsl_keys = {
            "stop_loss",
            "take_profit",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_reward_ratio",
            "atr_multiplier_sl",
            "atr_multiplier_tp",
            "_tpsl_strategy",
            "_tpsl_method",
        }

        clean_risk_management = {}
        for key, value in risk_management.items():
            if key not in tpsl_keys:
                if isinstance(value, float):
                    if key == "position_size":
                        clean_risk_management[key] = round(value, 6)
                    else:
                        clean_risk_management[key] = round(value, 4)
                else:
                    clean_risk_management[key] = value

        return clean_risk_management
