"""
統一遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーションを担当するモジュール。
DictConverterとGeneSerializerを統合し、JSON/Dict形式の相互変換を提供します。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import logging
import math
from typing import Any, Dict, Optional, cast

from .strategy_gene_dict_codec import StrategyGeneDictCodec


from ..genes import (
    Condition,
    EntryGene,
    ExitGene,
    IndicatorGene,
    PositionSizingGene,
    StrategyGene,
    TPSLGene,
)

from app.types import SerializableValue

# キャッシュキー用にハッシュ可能な構造
_FrozenKey = tuple | str | int | float | bool | None | bytes

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
        self._strategy_gene_codec = StrategyGeneDictCodec(self)
        self._cache_size = cache_size
        self._serialize_cache: dict[
            int | str | tuple | bytes | float | bool | None,
            dict[str, SerializableValue],
        ] = {}
        self._deserialize_cache: dict[
            int | str | tuple | bytes | float | bool | None, object
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
            frozen_items = [self._freeze_for_cache_key(item) for item in value]
            frozen_items.sort(key=repr)
            return ("set", tuple(frozen_items))

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
            logger.debug("キャッシュキー生成に失敗しました、フォールバックします: %s", e)
            return ("object_id", id(strategy_gene))

    def _generate_dict_cache_key(self, data: object) -> _FrozenKey:
        """辞書データ用のキャッシュキーを生成する。"""
        try:
            return self._freeze_for_cache_key(data)
        except Exception as e:
            logger.debug("辞書キャッシュキー生成に失敗しました、フォールバックします: %s", e)
            return ("object_id", id(data))

    @staticmethod
    def _safe_serialize(
        serialize_func, *args, error_prefix: str, **kwargs
    ):
        """シリアライゼーション処理を安全に実行するヘルパー。"""
        try:
            return serialize_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_prefix}エラー: {e}")
            raise ValueError(f"{error_prefix}に失敗: {e}")

    @staticmethod
    def _safe_deserialize(
        deserialize_func, *args, gene_class_name: str, **kwargs
    ):
        """デシリアライゼーション処理を安全に実行するヘルパー。"""
        try:
            return deserialize_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{gene_class_name}復元エラー: {e}")
            raise ValueError(f"{gene_class_name}の復元に失敗: {e}")

    def strategy_gene_to_dict(self, strategy_gene: Any) -> Dict[str, Any]:
        """戦略遺伝子オブジェクトをシリアライズ可能な辞書形式に変換"""
        try:
            cache_key = self._generate_cache_key(strategy_gene)
            if cache_key in self._serialize_cache:
                result = self._serialize_cache[cache_key]
            else:
                result = self._strategy_gene_codec.strategy_gene_to_dict(strategy_gene)
                if len(self._serialize_cache) < self._cache_size:
                    self._serialize_cache[cache_key] = result
            return cast(Dict[str, Any], self._copy_cached_value(result))
        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """
        指標遺伝子を辞書形式に変換
        """
        def _serialize():
            result = {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }
            if indicator_gene.timeframe is not None:
                result["timeframe"] = indicator_gene.timeframe
            return result
        return self._safe_serialize(_serialize, error_prefix="指標遺伝子辞書変換")

    def dict_to_indicator_gene(self, data: Dict[str, Any]) -> "IndicatorGene":
        """
        辞書形式から指標遺伝子を復元
        """
        def _deserialize():
            from ..genes import IndicatorGene
            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
                timeframe=data.get("timeframe"),
            )
        return self._safe_deserialize(_deserialize, gene_class_name="指標遺伝子")

    def condition_to_dict(self, condition) -> Dict[str, Any]:
        """条件を辞書形式に変換"""
        def _serialize():
            result = {
                "left_operand": condition.left_operand,
                "operator": condition.operator,
                "right_operand": condition.right_operand,
            }
            if getattr(condition, "direction", None) is not None:
                result["direction"] = condition.direction
            return result
        return self._safe_serialize(_serialize, error_prefix="条件辞書変換")

    def condition_or_group_to_dict(self, obj) -> Dict[str, Any]:
        """Condition または ConditionGroup を辞書に変換"""
        def _serialize():
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
        return self._safe_serialize(_serialize, error_prefix="条件/グループ辞書変換")

    def tpsl_gene_to_dict(self, tpsl_gene) -> Optional[Dict[str, Any]]:
        """TP/SL遺伝子を辞書形式に変換"""
        def _serialize():
            if tpsl_gene is None:
                return None
            return tpsl_gene.to_dict()
        return self._safe_serialize(_serialize, error_prefix="TP/SL遺伝子辞書変換")

    def dict_to_tpsl_gene(self, data: Dict[str, Any]) -> Optional["TPSLGene"]:
        """辞書形式からTP/SL遺伝子を復元"""
        def _deserialize():
            if data is None:
                return None
            from ..genes import TPSLGene
            return TPSLGene.from_dict(data)
        return self._safe_deserialize(_deserialize, gene_class_name="TP/SL遺伝子")

    def position_sizing_gene_to_dict(
        self, position_sizing_gene
    ) -> Optional[Dict[str, Any]]:
        """ポジションサイジング遺伝子を辞書形式に変換"""
        def _serialize():
            if position_sizing_gene is None:
                return None
            return position_sizing_gene.to_dict()
        return self._safe_serialize(_serialize, error_prefix="ポジションサイジング遺伝子辞書変換")

    def dict_to_position_sizing_gene(
        self, data: Dict[str, Any]
    ) -> Optional["PositionSizingGene"]:
        """辞書形式からポジションサイジング遺伝子を復元"""
        def _deserialize():
            if data is None:
                return None
            from ..genes import PositionSizingGene
            return PositionSizingGene.from_dict(data)
        return self._safe_deserialize(_deserialize, gene_class_name="ポジションサイジング遺伝子")

    def entry_gene_to_dict(self, entry_gene) -> Optional[Dict[str, Any]]:
        """エントリー遺伝子を辞書形式に変換"""
        def _serialize():
            if entry_gene is None:
                return None
            return entry_gene.to_dict()
        return self._safe_serialize(_serialize, error_prefix="エントリー遺伝子辞書変換")

    def dict_to_entry_gene(self, data: Dict[str, Any]) -> Optional["EntryGene"]:
        """辞書形式からエントリー遺伝子を復元"""
        def _deserialize():
            if data is None:
                return None
            from ..genes import EntryGene
            return EntryGene.from_dict(data)
        return self._safe_deserialize(_deserialize, gene_class_name="エントリー遺伝子")

    def exit_gene_to_dict(self, exit_gene) -> Optional[Dict[str, Any]]:
        """イグジット遺伝子を辞書形式に変換"""
        def _serialize():
            if exit_gene is None:
                return None
            return exit_gene.to_dict()
        return self._safe_serialize(_serialize, error_prefix="イグジット遺伝子辞書変換")

    def dict_to_exit_gene(self, data: Dict[str, Any]) -> Optional[ExitGene]:
        """辞書形式からイグジット遺伝子を復元"""
        def _deserialize():
            if data is None:
                return None
            return ExitGene.from_dict(data)
        return self._safe_deserialize(_deserialize, gene_class_name="イグジット遺伝子")

    def _clean_risk_management(self, risk_management: Dict[str, Any]) -> Dict[str, Any]:
        """risk_managementからTP/SL関連の設定を除外"""
        return self._strategy_gene_codec._clean_risk_management(risk_management)

    def dict_to_strategy_gene(self, data: Any, strategy_gene_class: Any):
        """辞書形式のデータから戦略遺伝子オブジェクトを復元"""
        try:
            if strategy_gene_class is None:
                from ..genes import StrategyGene

                strategy_gene_class = StrategyGene

            cache_key = self._generate_dict_cache_key(data)
            if cache_key in self._deserialize_cache:
                result = self._deserialize_cache[cache_key]
            else:
                result = self._strategy_gene_codec.dict_to_strategy_gene(
                    data, strategy_gene_class
                )
                if len(self._deserialize_cache) < self._cache_size:
                    self._deserialize_cache[cache_key] = result
            return self._copy_cached_value(result)
        except Exception as e:
            logger.error(f"戦略遺伝子復元エラー: {e}")
            raise ValueError(f"戦略遺伝子の復元に失敗: {e}")

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

    def clear_caches(self) -> None:
        """最適化されたキャッシュをクリアする。"""
        self._serialize_cache.clear()
        self._deserialize_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """最適化されたキャッシュ統計を返す。"""
        return {
            "serialize_cache_size": len(self._serialize_cache),
            "deserialize_cache_size": len(self._deserialize_cache),
            "cache_limit": self._cache_size,
        }


class GeneSerializer(DictConverter):
    """
    戦略遺伝子のシリアライゼーション補助クラス

    辞書変換は `DictConverter` を継承して直接提供し、JSON 変換と
    DEAP 個体のリスト復元だけを追加します。
    """

    def __init__(self, cache_size: int = 1000):
        super().__init__(cache_size=cache_size)

    def from_list(
        self, individual_list: list, strategy_gene_class: Any
    ) -> Optional[StrategyGene]:
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
            return cast(StrategyGene, individual_list)

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
