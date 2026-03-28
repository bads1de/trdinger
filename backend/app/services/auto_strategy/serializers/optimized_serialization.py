"""
最適化されたシリアライゼーションモジュール

シリアライズ・デシリアライズのキャッシング、バイナリシリアライズなどの最適化を提供します。
"""

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Dict

from .strategy_gene_dict_codec import StrategyGeneDictCodec

if TYPE_CHECKING:
    from ..genes import Condition, IndicatorGene

logger = logging.getLogger(__name__)


class OptimizedDictConverter:
    """
    最適化された戦略遺伝子と辞書形式の相互変換を行うコンバーター

    主な最適化ポイント:
    1. シリアライズ結果のキャッシング
    2. デシリアライズ結果のキャッシング
    3. ハッシュベースのキャッシュキー生成
    """

    def __init__(self, cache_size: int = 1000) -> None:
        self._strategy_gene_codec = StrategyGeneDictCodec(self)
        self._cache_size = cache_size

        # キャッシュ
        self._serialize_cache: Dict[str, Dict[str, Any]] = {}
        self._deserialize_cache: Dict[str, Any] = {}

    def strategy_gene_to_dict(self, strategy_gene: Any) -> Dict[str, Any]:
        """戦略遺伝子オブジェクトをシリアライズ可能な辞書形式に変換（最適化版）"""
        # キャッシュキーを生成
        cache_key = self._generate_cache_key(strategy_gene)

        # キャッシュチェック
        if cache_key in self._serialize_cache:
            logger.debug("シリアライズキャッシュヒット: %s", cache_key)
            return self._serialize_cache[cache_key]

        # シリアライズ実行
        result = self._strategy_gene_codec.strategy_gene_to_dict(strategy_gene)

        # キャッシュに保存
        if len(self._serialize_cache) < self._cache_size:
            self._serialize_cache[cache_key] = result

        return result

    def dict_to_strategy_gene(self, data: Dict[str, Any]) -> Any:
        """辞書形式から戦略遺伝子を復元（最適化版）"""
        # キャッシュキーを生成
        cache_key = self._generate_dict_cache_key(data)

        # キャッシュチェック
        if cache_key in self._deserialize_cache:
            logger.debug("デシリアライズキャッシュヒット: %s", cache_key)
            return self._deserialize_cache[cache_key]

        # デシリアライズ実行
        from ..genes import StrategyGene

        result = self._strategy_gene_codec.dict_to_strategy_gene(data, StrategyGene)

        # キャッシュに保存
        if len(self._deserialize_cache) < self._cache_size:
            self._deserialize_cache[cache_key] = result

        return result

    def _generate_cache_key(self, strategy_gene: Any) -> str:
        """キャッシュキーを生成"""
        try:
            gene_id = getattr(strategy_gene, "id", "") or ""
            if gene_id:
                return f"gene_{gene_id}"

            # IDがない場合はハッシュを生成
            indicators = getattr(strategy_gene, "indicators", [])
            long_conditions = getattr(strategy_gene, "long_entry_conditions", [])
            short_conditions = getattr(strategy_gene, "short_entry_conditions", [])

            key_parts = [
                str(len(indicators)),
                str(len(long_conditions)),
                str(len(short_conditions)),
            ]

            for ind in indicators:
                ind_type = getattr(ind, "type", "")
                ind_params = getattr(ind, "parameters", {})
                key_parts.append(f"{ind_type}_{str(ind_params)}")

            key_str = "|".join(key_parts)
            return hashlib.md5(key_str.encode()).hexdigest()[:16]

        except Exception:
            return str(id(strategy_gene))

    def _generate_dict_cache_key(self, data: Dict[str, Any]) -> str:
        """辞書データのキャッシュキーを生成"""
        try:
            key_str = json.dumps(data, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()[:16]
        except Exception:
            return str(id(data))

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """指標遺伝子を辞書形式に変換"""
        try:
            result = {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }
            if indicator_gene.timeframe is not None:
                result["timeframe"] = indicator_gene.timeframe
            return result
        except Exception as e:
            logger.error(f"指標遺伝子辞書変換エラー: {e}")
            raise ValueError(f"指標遺伝子の辞書変換に失敗: {e}")

    def dict_to_indicator_gene(self, data: Dict[str, Any]) -> "IndicatorGene":
        """辞書形式から指標遺伝子を復元"""
        try:
            from ..genes import IndicatorGene

            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
                timeframe=data.get("timeframe"),
            )
        except Exception as e:
            logger.error(f"指標遺伝子復元エラー: {e}")
            raise ValueError(f"指標遺伝子の復元に失敗: {e}")

    def condition_to_dict(self, condition) -> Dict[str, Any]:
        """条件を辞書形式に変換"""
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

    def dict_to_condition(self, data: Dict[str, Any]) -> "Condition":
        """辞書形式から条件を復元"""
        try:
            from ..genes import Condition

            return Condition(
                left_operand=data["left_operand"],
                operator=data["operator"],
                right_operand=data["right_operand"],
            )
        except Exception as e:
            logger.error(f"条件復元エラー: {e}")
            raise ValueError(f"条件の復元に失敗: {e}")

    def clear_caches(self):
        """キャッシュをクリア"""
        self._serialize_cache.clear()
        self._deserialize_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        return {
            "serialize_cache_size": len(self._serialize_cache),
            "deserialize_cache_size": len(self._deserialize_cache),
            "cache_limit": self._cache_size,
        }


# 後方互換性のためのエイリアス
DictConverter = OptimizedDictConverter
