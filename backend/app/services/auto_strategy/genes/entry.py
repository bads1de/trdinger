"""
エントリー注文遺伝子

エントリー注文のタイプとパラメータを定義します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from app.types import StrategyGeneDict
if TYPE_CHECKING:
    from ..config.ga.ga_config import GAConfig

from app.utils.serialization import dataclass_to_dict

from ..config.constants import EntryType
from .gene_constants import ENTRY_TYPE_WEIGHTS, PRIORITY_GENERATION_RANGE
from .gene_ranges import ENTRY_GENERATION_RANGES


@dataclass(slots=True)
class EntryGene:
    """
    エントリー注文遺伝子

    GAによって最適化されるエントリー注文のパラメータを定義します。
    """

    # 遺伝的操作のための設定
    NUMERIC_FIELDS = [
        "limit_offset_pct",
        "stop_offset_pct",
        "order_validity_bars",
        "priority",
    ]
    ENUM_FIELDS = ["entry_type"]
    CHOICE_FIELDS = ["enabled"]
    NUMERIC_RANGES = {
        "limit_offset_pct": (0.0, 0.1),
        "stop_offset_pct": (0.0, 0.1),
        "order_validity_bars": (0, 100),
        "priority": (0.5, 1.5),
    }

    # エントリータイプ
    entry_type: EntryType = EntryType.MARKET

    # 指値のオフセット（現在価格からの乖離率）
    # Long: 現在価格 * (1 - limit_offset_pct) で買い指値
    # Short: 現在価格 * (1 + limit_offset_pct) で売り指値
    limit_offset_pct: float = 0.005  # 0.5%

    # 逆指値のオフセット（現在価格からの乖離率）
    # Long: 現在価格 * (1 + stop_offset_pct) でブレイクアウト買い
    # Short: 現在価格 * (1 - stop_offset_pct) でブレイクアウト売り
    stop_offset_pct: float = 0.005  # 0.5%

    # 注文の有効期限（バー数） - 0は無制限
    order_validity_bars: int = 5

    # 有効/無効フラグ
    enabled: bool = True

    # 優先度（将来の拡張用）
    priority: float = 1.0

    def validate(self) -> Tuple[bool, List[str]]:
        """
        エントリー遺伝子の妥当性を検証

        Returns:
            (is_valid, errors) のタプル
        """
        errors: List[str] = []

        # limit_offset_pct の範囲チェック (0% ~ 10%)
        if self.limit_offset_pct < 0 or self.limit_offset_pct > 0.1:
            errors.append(
                f"limit_offset_pct は 0~0.1 の範囲である必要があります: {self.limit_offset_pct}"
            )

        # stop_offset_pct の範囲チェック (0% ~ 10%)
        if self.stop_offset_pct < 0 or self.stop_offset_pct > 0.1:
            errors.append(
                f"stop_offset_pct は 0~0.1 の範囲である必要があります: {self.stop_offset_pct}"
            )

        # order_validity_bars の範囲チェック (0以上)
        if self.order_validity_bars < 0:
            errors.append(
                f"order_validity_bars は 0以上である必要があります: {self.order_validity_bars}"
            )

        # entry_type が有効な値かチェック
        if not isinstance(self.entry_type, EntryType):
            try:
                EntryType(self.entry_type)
            except ValueError:
                errors.append(f"無効な entry_type です: {self.entry_type}")

        return len(errors) == 0, errors

    def to_dict(self) -> StrategyGeneDict:
        """
        辞書形式に変換

        Returns:
            辞書形式のデータ
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: StrategyGeneDict) -> "EntryGene":
        """
        辞書形式から復元

        Args:
            data: 辞書形式のデータ

        Returns:
            EntryGene オブジェクト
        """
        entry_type_value = data.get("entry_type", "market")
        if isinstance(entry_type_value, str):
            entry_type = EntryType(entry_type_value)
        elif isinstance(entry_type_value, EntryType):
            entry_type = entry_type_value
        else:
            entry_type = EntryType.MARKET

        return cls(
            entry_type=entry_type,
            limit_offset_pct=data.get("limit_offset_pct", 0.005),
            stop_offset_pct=data.get("stop_offset_pct", 0.005),
            order_validity_bars=data.get("order_validity_bars", 5),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def clone(self) -> EntryGene:
        """軽量コピーを作成"""
        return EntryGene(
            entry_type=self.entry_type,
            limit_offset_pct=self.limit_offset_pct,
            stop_offset_pct=self.stop_offset_pct,
            order_validity_bars=self.order_validity_bars,
            enabled=self.enabled,
            priority=self.priority,
        )

    def mutate(self, mutation_rate: float = 0.1) -> "EntryGene":
        """突然変異"""
        import random

        gene = self.clone()

        if random.random() < mutation_rate:
            # タイプの変更
            if random.random() < 0.3:
                gene.entry_type = random.choice(list(EntryType))

            # パラメータの摂動
            if random.random() < 0.5:
                gene.limit_offset_pct = max(
                    0.0,
                    min(0.1, gene.limit_offset_pct * random.uniform(0.8, 1.2)),
                )
                gene.stop_offset_pct = max(
                    0.0,
                    min(0.1, gene.stop_offset_pct * random.uniform(0.8, 1.2)),
                )

            # 有効期限の変更
            if random.random() < 0.3:
                gene.order_validity_bars = max(
                    1,
                    min(
                        100,
                        int(
                            gene.order_validity_bars * random.uniform(0.8, 1.2)
                        ),
                    ),
                )

        return gene

    @classmethod
    def crossover(
        cls, parent1: "EntryGene", parent2: "EntryGene"
    ) -> Tuple["EntryGene", "EntryGene"]:
        """エントリー遺伝子の交叉"""
        from .genetic_utils import GeneticUtils

        return GeneticUtils.crossover_generic_genes(
            parent1_gene=parent1,
            parent2_gene=parent2,
            gene_class=cls,
            numeric_fields=cls.NUMERIC_FIELDS,
            enum_fields=cls.ENUM_FIELDS,
            choice_fields=cls.CHOICE_FIELDS,
        )


def create_random_entry_gene(config: Optional[Any] = None) -> EntryGene:
    """
    ランダムなエントリー遺伝子を生成

    Args:
        config: GA設定オブジェクト（オプション）

    Returns:
        生成されたエントリー遺伝子
    """
    import random

    default_weights = ENTRY_TYPE_WEIGHTS
    ranges = ENTRY_GENERATION_RANGES

    try:
        # 重み付きでエントリータイプを選択
        weights = default_weights
        if config is not None:
            try:
                custom_weights = config.entry_type_weights
                if custom_weights:
                    weights = {
                        EntryType(k): v for k, v in custom_weights.items()
                    }
            except AttributeError:
                pass

        types = list(weights.keys())
        type_weights = list(weights.values())

        entry_type = random.choices(types, weights=type_weights, k=1)[0]

        # オフセット値をランダム生成
        limit_offset_pct = random.uniform(*ranges["limit_offset_pct"])
        stop_offset_pct = random.uniform(*ranges["stop_offset_pct"])

        # 有効期限をランダム生成
        order_validity_bars = random.randint(
            int(ranges["order_validity_bars"][0]),
            int(ranges["order_validity_bars"][1]),
        )

        return EntryGene(
            entry_type=entry_type,
            limit_offset_pct=limit_offset_pct,
            stop_offset_pct=stop_offset_pct,
            order_validity_bars=order_validity_bars,
            enabled=True,
            priority=random.uniform(*PRIORITY_GENERATION_RANGE),
        )

    except (ValueError, TypeError, KeyError, AttributeError):
        # フォールバック: 成行注文のデフォルト遺伝子
        return EntryGene(
            entry_type=EntryType.MARKET,
            limit_offset_pct=ranges["limit_offset_pct"][0],
            stop_offset_pct=ranges["stop_offset_pct"][0],
            order_validity_bars=int(ranges["order_validity_bars"][0]),
            enabled=True,
        )
