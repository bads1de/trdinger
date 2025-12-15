"""
辞書形式変換コンポーネント

戦略遺伝子の辞書形式変換機能を担当します。
"""

import logging
import uuid
from typing import Any, Dict, Optional

from ..genes.entry_gene import EntryGene
from ..genes import (
    Condition,
    IndicatorGene,
    PositionSizingGene,
    TPSLGene,
)
from ..utils.gene_utils import GeneUtils
from ..utils.indicator_utils import get_all_indicator_ids

logger = logging.getLogger(__name__)


class DictConverter:
    """
    辞書形式変換クラス

    戦略遺伝子の辞書形式変換機能を担当します。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: ConditionGeneratorを使用するか
        """
        self.indicator_ids = get_all_indicator_ids()
        self.id_to_indicator = {v: k for k, v in self.indicator_ids.items()}

        # ConditionGeneratorの遅延インポート（循環インポート回避）
        self.enable_smart_generation = enable_smart_generation
        self._smart_condition_generator = None

    @property
    def smart_condition_generator(self):
        """ConditionGeneratorの遅延初期化"""
        if self._smart_condition_generator is None and self.enable_smart_generation:
            from ..generators.condition_generator import ConditionGenerator

            self._smart_condition_generator = ConditionGenerator(True)
        return self._smart_condition_generator

    def strategy_gene_to_dict(self, strategy_gene) -> Dict[str, Any]:
        """
        戦略遺伝子を辞書形式に変換

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            # risk_managementからTP/SL関連の設定を除外
            clean_risk_management = self._clean_risk_management(
                strategy_gene.risk_management
            )

            return {
                "id": strategy_gene.id,
                "indicators": [
                    self.indicator_gene_to_dict(ind) for ind in strategy_gene.indicators
                ],
                "entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.entry_conditions
                ],
                "long_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.long_entry_conditions
                ],
                "short_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.short_entry_conditions
                ],
                "exit_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.exit_conditions
                ],
                "risk_management": clean_risk_management,
                "tpsl_gene": (
                    self.tpsl_gene_to_dict(strategy_gene.tpsl_gene)
                    if strategy_gene.tpsl_gene
                    else None
                ),
                "long_tpsl_gene": (
                    self.tpsl_gene_to_dict(strategy_gene.long_tpsl_gene)
                    if getattr(strategy_gene, "long_tpsl_gene", None)
                    else None
                ),
                "short_tpsl_gene": (
                    self.tpsl_gene_to_dict(strategy_gene.short_tpsl_gene)
                    if getattr(strategy_gene, "short_tpsl_gene", None)
                    else None
                ),
                "position_sizing_gene": (
                    self.position_sizing_gene_to_dict(
                        strategy_gene.position_sizing_gene
                    )
                    if getattr(strategy_gene, "position_sizing_gene", None)
                    else None
                ),
                "entry_gene": (
                    self.entry_gene_to_dict(strategy_gene.entry_gene)
                    if getattr(strategy_gene, "entry_gene", None)
                    else None
                ),
                "long_entry_gene": (
                    self.entry_gene_to_dict(strategy_gene.long_entry_gene)
                    if getattr(strategy_gene, "long_entry_gene", None)
                    else None
                ),
                "short_entry_gene": (
                    self.entry_gene_to_dict(strategy_gene.short_entry_gene)
                    if getattr(strategy_gene, "short_entry_gene", None)
                    else None
                ),
                "stateful_conditions": [
                    self.stateful_condition_to_dict(sc)
                    for sc in getattr(strategy_gene, "stateful_conditions", [])
                ],
                "tool_genes": [
                    tg.to_dict() for tg in getattr(strategy_gene, "tool_genes", [])
                ],
                "metadata": strategy_gene.metadata,
            }

        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

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

            return TPSLGene.from_dict(data)  # type: ignore[cSpell] # TPSL is a valid trading acronym

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
        """
        risk_managementからTP/SL関連の設定を除外

        Args:
            risk_management: 元のリスク管理設定

        Returns:
            TP/SL関連設定を除外したリスク管理設定
        """
        # TP/SL関連のキーを除外
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
                # 数値の場合は適切な桁数に丸める
                if isinstance(value, float):
                    if key == "position_size":
                        # BTCトレードでは小数点以下の精度が重要なため、6桁の精度を保持
                        clean_risk_management[key] = round(value, 6)
                    else:
                        clean_risk_management[key] = round(value, 4)
                else:
                    clean_risk_management[key] = value

        return clean_risk_management

    def dict_to_strategy_gene(self, data: Dict[str, Any], strategy_gene_class):
        """
        辞書形式から戦略遺伝子を復元

        Args:
            data: 辞書形式の戦略遺伝子データ
            strategy_gene_class: StrategyGeneクラス

        Returns:
            戦略遺伝子オブジェクト
        """
        try:
            # 入力データがNoneまたは空でないことを確認
            if not data:
                logger.warning(
                    "戦略遺伝子データが空です。デフォルト戦略遺伝子を返します。"
                )
                return GeneUtils.create_default_strategy_gene(strategy_gene_class)

            # 指標遺伝子の復元
            indicators = [
                self.dict_to_indicator_gene(ind_data)
                for ind_data in data.get("indicators", [])
            ]

            # 条件の復元
            from ..genes import ConditionGroup

            def parse_condition_or_group(cond_data):
                if isinstance(cond_data, dict):
                    # 新しいグループ形式
                    if cond_data.get("type") == "GROUP":
                        conditions = [
                            parse_condition_or_group(c)
                            for c in cond_data.get("conditions", [])
                        ]
                        operator = cond_data.get("operator", "OR")
                        return ConditionGroup(operator=operator, conditions=conditions)
                    # 古いグループ形式（互換性用）
                    elif cond_data.get("type") == "OR_GROUP":
                        conditions = [
                            parse_condition_or_group(c)
                            for c in cond_data.get("conditions", [])
                        ]
                        return ConditionGroup(operator="OR", conditions=conditions)
                    else:
                        return self.dict_to_condition(cond_data)
                else:
                    logger.warning(f"不正な条件データ形式: {str(cond_data)[:50]}")
                    return None

            entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("entry_conditions", [])
            ]

            long_entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("long_entry_conditions", [])
            ]

            short_entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("short_entry_conditions", [])
            ]

            exit_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("exit_conditions", [])
            ]

            # リスク管理設定
            risk_management = data.get("risk_management", {})

            # TP/SL遺伝子の復元
            tpsl_gene = None
            if "tpsl_gene" in data and data["tpsl_gene"]:
                tpsl_gene = self.dict_to_tpsl_gene(data["tpsl_gene"])

            # Long/Short TP/SL遺伝子の復元
            long_tpsl_gene = None
            if "long_tpsl_gene" in data and data["long_tpsl_gene"]:
                long_tpsl_gene = self.dict_to_tpsl_gene(data["long_tpsl_gene"])

            short_tpsl_gene = None
            if "short_tpsl_gene" in data and data["short_tpsl_gene"]:
                short_tpsl_gene = self.dict_to_tpsl_gene(data["short_tpsl_gene"])

            # ポジションサイジング遺伝子の復元
            position_sizing_gene = None
            if "position_sizing_gene" in data and data["position_sizing_gene"]:
                position_sizing_gene = self.dict_to_position_sizing_gene(
                    data["position_sizing_gene"]
                )

            # エントリー遺伝子の復元
            entry_gene = None
            if "entry_gene" in data and data["entry_gene"]:
                entry_gene = self.dict_to_entry_gene(data["entry_gene"])

            long_entry_gene = None
            if "long_entry_gene" in data and data["long_entry_gene"]:
                long_entry_gene = self.dict_to_entry_gene(data["long_entry_gene"])

            short_entry_gene = None
            if "short_entry_gene" in data and data["short_entry_gene"]:
                short_entry_gene = self.dict_to_entry_gene(data["short_entry_gene"])

            # メタデータ
            metadata = data.get("metadata", {})

            # 有効な指標がない場合はデフォルト指標を追加
            enabled_indicators = [ind for ind in indicators if ind.enabled]
            if not enabled_indicators:
                from ..genes import IndicatorGene

                indicators.append(
                    IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)
                )
                logger.warning(
                    "有効な指標がなかったため、デフォルト指標SMAを追加しました"
                )

            # 後方互換性のための処理
            if not long_entry_conditions and entry_conditions:
                long_entry_conditions = entry_conditions
            if not short_entry_conditions and entry_conditions:
                short_entry_conditions = entry_conditions

            # ステートフル条件の復元
            stateful_conditions = [
                self.dict_to_stateful_condition(sc_data)
                for sc_data in data.get("stateful_conditions", [])
                if sc_data is not None
            ]

            # ツール遺伝子の復元
            from ..genes.tool_gene import ToolGene

            tool_genes = [
                ToolGene.from_dict(tg_data)
                for tg_data in data.get("tool_genes", [])
                if tg_data is not None
            ]

            return strategy_gene_class(
                id=data.get("id", str(uuid.uuid4())),
                indicators=indicators,
                entry_conditions=entry_conditions,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                exit_conditions=exit_conditions,
                stateful_conditions=stateful_conditions,
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,
                long_tpsl_gene=long_tpsl_gene,
                short_tpsl_gene=short_tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                entry_gene=entry_gene,
                long_entry_gene=long_entry_gene,
                short_entry_gene=short_entry_gene,
                tool_genes=tool_genes,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"戦略遺伝子辞書復元エラー: {e}")
            # エラー時はデフォルト戦略遺伝子を返す
            return GeneUtils.create_default_strategy_gene(strategy_gene_class)

    def dict_to_condition(self, data: Dict[str, Any]) -> "Condition":
        """
        辞書形式から条件を復元

        Args:
            data: 辞書形式の条件データ

        Returns:
            条件オブジェクト
        """
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





