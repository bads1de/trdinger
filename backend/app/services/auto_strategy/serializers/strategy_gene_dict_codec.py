"""
StrategyGene の dict encode/decode helper。
"""

import logging
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Tuple, cast

from ..genes import StrategyGene

logger = logging.getLogger(__name__)


class StrategyGeneDictCodec:
    """StrategyGene の辞書変換ロジックを担当する helper。"""

    def __init__(self, converter: Any) -> None:
        self.converter = converter

    @staticmethod
    def _get_sub_gene_field_names(strategy_gene_class: type) -> Tuple[str, ...]:
        """StrategyGene 系クラスのサブ遺伝子フィールド名を取得する。"""
        getter = getattr(strategy_gene_class, "sub_gene_field_names", None)
        if callable(getter):
            raw_field_names = getter()
            if isinstance(raw_field_names, (str, bytes)) or not isinstance(
                raw_field_names, Iterable
            ):
                raise TypeError(
                    "sub_gene_field_names は str/bytes ではない反復可能な文字列コレクションを返す必要があります。"
                )

            field_names = tuple(raw_field_names)
            if not all(isinstance(name, str) for name in field_names):
                raise TypeError(
                    "sub_gene_field_names は str のみで構成されたコレクションを返す必要があります。"
                )

            return cast(Tuple[str, ...], field_names)

        return StrategyGene.sub_gene_field_names()

    @staticmethod
    def _get_sub_gene_class_map(strategy_gene_class: type) -> Dict[str, Any]:
        """StrategyGene 系クラスのサブ遺伝子クラス対応表を取得する。"""
        getter = getattr(strategy_gene_class, "sub_gene_class_map", None)
        if callable(getter):
            raw_class_map = getter()
            if not isinstance(raw_class_map, Mapping):
                raise TypeError(
                    "sub_gene_class_map は Mapping[str, Any] を返す必要があります。"
                )

            if not all(isinstance(field_name, str) for field_name in raw_class_map):
                raise TypeError(
                    "sub_gene_class_map のキーはすべて str である必要があります。"
                )

            return dict(cast(Mapping[str, Any], raw_class_map))

        return StrategyGene.sub_gene_class_map()

    def strategy_gene_to_dict(self, strategy_gene: StrategyGene) -> Dict[str, Any]:
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
                "long_exit_conditions": [
                    self.converter.condition_or_group_to_dict(cond)
                    for cond in getattr(strategy_gene, "long_exit_conditions", [])
                ],
                "short_exit_conditions": [
                    self.converter.condition_or_group_to_dict(cond)
                    for cond in getattr(strategy_gene, "short_exit_conditions", [])
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

            sub_gene_fields = self._get_sub_gene_field_names(type(strategy_gene))
            for field in sub_gene_fields:
                gene_obj = getattr(strategy_gene, field, None)
                result[field] = gene_obj.to_dict() if gene_obj else None

            return result

        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def dict_to_strategy_gene(self, data: Dict[str, Any], strategy_gene_class: type):
        """辞書形式のデータから戦略遺伝子オブジェクトを復元。"""
        try:
            if isinstance(data, strategy_gene_class):
                return data

            if hasattr(data, "indicators") and hasattr(data, "long_entry_conditions"):
                return data

            if not isinstance(data, dict):
                raise TypeError(
                    "dict_to_strategy_gene に渡されたデータの型が不正です: "
                    f"{type(data).__name__}"
                )

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
                    raise TypeError(
                        "条件データはdictである必要があります: "
                        f"{type(cond_data).__name__}"
                    )
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

            long_exit_conditions = [
                parse_condition_or_group(c)
                for c in data.get("long_exit_conditions", [])
            ]
            short_exit_conditions = [
                parse_condition_or_group(c)
                for c in data.get("short_exit_conditions", [])
            ]

            from ..genes.tool import ToolGene

            sub_genes = {}
            mapping = self._get_sub_gene_class_map(strategy_gene_class)
            for field in self._get_sub_gene_field_names(strategy_gene_class):
                cls = mapping.get(field)
                if cls is None:
                    continue
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
                long_exit_conditions=long_exit_conditions,
                short_exit_conditions=short_exit_conditions,
                stateful_conditions=stateful_conditions,
                tool_genes=tool_genes,
                risk_management=data.get("risk_management", {"position_size": 0.1}),
                metadata=data.get("metadata", {}),
                **sub_genes,
            )

        except Exception as e:
            logger.error(f"戦略遺伝子辞書復元エラー: {e}")
            raise ValueError(f"戦略遺伝子の復元に失敗: {e}") from e

    def dict_to_condition(self, data: Dict[str, Any]):
        """辞書形式から条件を復元。"""
        try:
            from ..genes import Condition

            return Condition(
                left_operand=data["left_operand"],
                operator=data["operator"],
                right_operand=data["right_operand"],
                direction=data.get("direction"),
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
