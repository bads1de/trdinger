"""
統一遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーションを担当するモジュール。
DictConverterとGeneSerializerを統合し、JSON/Dict形式の相互変換を提供します。
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, cast

from app.types import SerializablePrimitive, SerializableValue

from ..genes import (
    Condition,
    ConditionGroup,
    IndicatorGene,
    StrategyGene,
)
from ..genes.conditions import StatefulCondition

# キャッシュキー用にハッシュ可能な構造
_FrozenKey = tuple | bytes | SerializablePrimitive

logger = logging.getLogger(__name__)


class DictConverter:
    """
    戦略遺伝子と辞書形式の相互変換を行うコンバーター

    `StrategyGene` オブジェクトをシリアライズ可能な辞書形式に変換し、
    また辞書から元のクラス構造を再構築します。
    サブ遺伝子（TP/SL、サイジング等）や動的な指標パラメータの
    再帰的な変換を管理します。
    """

    def __init__(self, cache_size: int = 1000) -> None:
        self._cache_size = cache_size
        self._serialize_cache: dict[
            tuple | bytes | SerializablePrimitive,
            dict[str, SerializableValue],
        ] = {}
        self._deserialize_cache: dict[
            tuple | bytes | SerializablePrimitive, object
        ] = {}

    def _freeze_for_cache_key(self, value: object) -> _FrozenKey:
        """キャッシュキー用にオブジェクトをハッシュ可能な構造へ正規化する。"""
        if value is None or isinstance(value, (str, int, bool, bytes)):
            return value

        if isinstance(value, float):
            if math.isnan(value):
                return ("float", "nan")
            if math.isinf(value):
                return ("float", "inf" if value > 0 else "-inf")
            return value

        if isinstance(value, Enum):
            return (
                "enum",
                type(value).__qualname__,
                self._freeze_for_cache_key(value.value),
            )

        if isinstance(value, Mapping):
            frozen_items = [
                (
                    self._freeze_for_cache_key(key),
                    self._freeze_for_cache_key(item_value),
                )
                for key, item_value in value.items()
            ]
            frozen_items.sort(key=lambda item: repr(item[0]))
            return ("mapping", tuple(frozen_items))

        if isinstance(value, (list, tuple)):
            return (
                "sequence",
                tuple(self._freeze_for_cache_key(item) for item in value),
            )

        if isinstance(value, set):
            set_items: list[Any] = [self._freeze_for_cache_key(item) for item in value]
            set_items.sort(key=repr)
            return ("set", tuple(set_items))

        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value).__qualname__,
                tuple(
                    (
                        field_info.name,
                        self._freeze_for_cache_key(getattr(value, field_info.name)),
                    )
                    for field_info in fields(value)
                ),
            )

        if hasattr(value, "__dict__"):
            attrs = [
                (attr_name, self._freeze_for_cache_key(attr_value))
                for attr_name, attr_value in vars(value).items()
            ]
            attrs.sort(key=lambda item: item[0])
            return ("object", type(value).__qualname__, tuple(attrs))

        return ("repr", repr(value))

    def _copy_cached_value(self, value: object) -> object:
        """
        キャッシュ内容を返却用に軽量コピーする。

        直列化済みの辞書値と、復元済みの StrategyGene オブジェクトの両方を
        扱うため、返り値は SerializableValue に固定しない。
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return value.hex()

        if isinstance(value, Enum):
            return value.value

        # データクラスはディープコピーを作成
        if is_dataclass(value) and not isinstance(value, type):
            from copy import deepcopy

            return deepcopy(value)

        if isinstance(value, Mapping):
            return {
                str(self._copy_cached_value(key)): self._copy_cached_value(item_value)
                for key, item_value in value.items()
            }

        if isinstance(value, list):
            return [self._copy_cached_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._copy_cached_value(item) for item in value)

        if isinstance(value, set):
            return [self._copy_cached_value(item) for item in value]

        try:
            from copy import deepcopy

            return deepcopy(value)
        except Exception as e:
            logger.debug("deepcopyに失敗しました、reprにフォールバックします: %s", e)
            return repr(value)

    def _generate_cache_key(self, strategy_gene: object) -> _FrozenKey:
        """戦略遺伝子の構造に基づいて安定したキャッシュキーを生成する。"""
        try:
            return self._freeze_for_cache_key(strategy_gene)
        except Exception as e:
            logger.debug(
                "キャッシュキー生成に失敗しました、フォールバックします: %s", e
            )
            return ("object_id", id(strategy_gene))

    def _generate_dict_cache_key(self, data: object) -> _FrozenKey:
        """辞書データ用のキャッシュキーを生成する。"""
        try:
            return self._freeze_for_cache_key(data)
        except Exception as e:
            logger.debug(
                "辞書キャッシュキー生成に失敗しました、フォールバックします: %s",
                e,
            )
            return ("object_id", id(data))

    def strategy_gene_to_dict(self, strategy_gene: StrategyGene) -> dict[str, Any]:
        """戦略遺伝子オブジェクトをシリアライズ可能な辞書形式に変換"""
        try:
            cache_key = self._generate_cache_key(strategy_gene)
            if cache_key in self._serialize_cache:
                result = self._serialize_cache[cache_key]
            else:
                result = self._encode_strategy_gene(strategy_gene)
                if len(self._serialize_cache) < self._cache_size:
                    self._serialize_cache[cache_key] = result
            return cast(dict[str, Any], self._copy_cached_value(result))
        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def indicator_gene_to_dict(self, indicator_gene: IndicatorGene) -> dict[str, Any]:
        """
        指標遺伝子を辞書形式に変換
        """
        try:
            result = {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }
            if indicator_gene.timeframe is not None:
                result["timeframe"] = indicator_gene.timeframe
            # 条件のオペランドは build_indicator_reference_name で生成される
            # 「TYPE_ID[:8]_OUTPUT_INDEX」形式の参照名を使うため、
            # シリアライズ時に id を保持しないと復元後の参照名が崩れ、
            # エントリー条件が評価不能になる（生成戦略が0トレードになる原因）。
            if indicator_gene.id is not None:
                result["id"] = indicator_gene.id
            return result
        except Exception as e:
            logger.error("指標遺伝子辞書変換エラー: %s", e)
            raise ValueError(f"指標遺伝子辞書変換に失敗: {e}") from e

    def dict_to_indicator_gene(self, data: dict[str, Any]) -> IndicatorGene:
        """
        辞書形式から指標遺伝子を復元
        """
        try:
            from ..genes import IndicatorGene

            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
                timeframe=data.get("timeframe"),
                # 参照名を復元するために id も保持する（古いデータには無い場合は None）
                id=data.get("id"),
            )
        except Exception as e:
            logger.error("指標遺伝子復元エラー: %s", e)
            raise ValueError(f"指標遺伝子の復元に失敗: {e}") from e

    def condition_to_dict(self, condition: Condition) -> dict[str, Any]:
        """条件を辞書形式に変換"""
        try:
            result: dict[str, Any] = {
                "left_operand": condition.left_operand,
                "operator": condition.operator,
                "right_operand": condition.right_operand,
            }
            if getattr(condition, "direction", None) is not None:
                result["direction"] = condition.direction
            return result
        except Exception as e:
            logger.error("条件辞書変換エラー: %s", e)
            raise ValueError(f"条件辞書変換に失敗: {e}") from e

    def condition_or_group_to_dict(
        self, obj: Condition | ConditionGroup
    ) -> dict[str, Any]:
        """Condition または ConditionGroup を辞書に変換"""
        try:
            if isinstance(obj, ConditionGroup):
                return {
                    "type": "GROUP",
                    "operator": obj.operator,
                    "conditions": [
                        self.condition_or_group_to_dict(c) for c in obj.conditions
                    ],
                }
            else:
                return self.condition_to_dict(obj)
        except Exception as e:
            logger.error("条件/グループ辞書変換エラー: %s", e)
            raise ValueError(f"条件/グループ辞書変換に失敗: {e}") from e

    def dict_to_strategy_gene(
        self, data: dict[str, Any], strategy_gene_class: type | None = None
    ) -> StrategyGene:
        """辞書形式のデータから戦略遺伝子オブジェクトを復元"""
        try:
            if strategy_gene_class is None:
                strategy_gene_class = StrategyGene

            cache_key = self._generate_dict_cache_key(data)
            if cache_key in self._deserialize_cache:
                result = self._deserialize_cache[cache_key]
            else:
                result = self._decode_strategy_gene(data, strategy_gene_class)
                if len(self._deserialize_cache) < self._cache_size:
                    self._deserialize_cache[cache_key] = result
            return cast(StrategyGene, self._copy_cached_value(result))
        except Exception as e:
            logger.error(f"戦略遺伝子復元エラー: {e}")
            raise ValueError(f"戦略遺伝子の復元に失敗: {e}")

    def dict_to_condition(self, data: dict[str, Any]) -> Condition:
        """辞書形式から条件を復元"""
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

    def stateful_condition_to_dict(
        self, stateful_condition: StatefulCondition | None
    ) -> dict[str, Any] | None:
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

    def dict_to_stateful_condition(
        self, data: dict[str, Any] | None
    ) -> StatefulCondition | None:
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

    def _encode_strategy_gene(self, strategy_gene: StrategyGene) -> dict[str, Any]:
        """StrategyGene をシリアライズ可能な辞書へ変換する内部実装。"""
        try:
            clean_risk_management = self._clean_risk_management(
                strategy_gene.risk_management
            )

            result: dict[str, Any] = {
                "id": strategy_gene.id,
                "indicators": [
                    self.indicator_gene_to_dict(ind) for ind in strategy_gene.indicators
                ],
                "long_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.long_entry_conditions
                ],
                "short_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.short_entry_conditions
                ],
                "long_exit_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in getattr(strategy_gene, "long_exit_conditions", [])
                ],
                "short_exit_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in getattr(strategy_gene, "short_exit_conditions", [])
                ],
                "risk_management": clean_risk_management,
                "stateful_conditions": [
                    self.stateful_condition_to_dict(sc)
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

    def _decode_strategy_gene(
        self, data: dict[str, Any], strategy_gene_class: type
    ) -> Any:
        """辞書から StrategyGene を復元する内部実装。"""
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
                return strategy_gene_class.create_default()  # type: ignore[attr-defined]

            indicators = [
                self.dict_to_indicator_gene(ind_data)
                for ind_data in data.get("indicators", [])
            ]

            from ..genes import ConditionGroup

            def parse_condition_or_group(cond_data: Any) -> Any:
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
                sub_genes[field] = cls.from_dict(gene_data) if gene_data else None

            stateful_conditions = [
                self.dict_to_stateful_condition(sc_data)
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

    @staticmethod
    def _get_sub_gene_field_names(strategy_gene_class: type) -> tuple[str, ...]:
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

            return cast(tuple[str, ...], field_names)

        return StrategyGene.sub_gene_field_names()

    @staticmethod
    def _get_sub_gene_class_map(strategy_gene_class: type) -> dict[str, Any]:
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

    def _clean_risk_management(self, risk_management: dict[str, Any]) -> dict[str, Any]:
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


class GeneSerializer(DictConverter):
    """
    戦略遺伝子のシリアライゼーション補助クラス

    辞書変換は `DictConverter` を継承して直接提供し、JSON 変換と
    DEAP 個体のリスト復元だけを追加します。
    """

    def __init__(self, cache_size: int = 1000):
        super().__init__(cache_size=cache_size)

    def from_list(
        self,
        individual_list: list[Any] | StrategyGene,
        strategy_gene_class: type,
    ) -> StrategyGene | None:
        """
        リスト形式（DEAP個体）から戦略遺伝子を復元します。

        Args:
            individual_list: DEAPの個体（StrategyGeneを継承したリスト、または[StrategyGene]）
            strategy_gene_class: 復元に使用するクラス

        Returns:
            復元されたStrategyGene
        """
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
            return cast(StrategyGene, individual_list)

        return None
