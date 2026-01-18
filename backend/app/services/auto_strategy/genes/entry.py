"""
エントリー注文遺伝子

エントリー注文のタイプとパラメータを定義します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from ..config.constants import EntryType


@dataclass(slots=True)
class EntryGene:
    """
    エントリー注文遺伝子

    GAによって最適化されるエントリー注文のパラメータを定義します。
    """

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

    def to_dict(self) -> dict:
        """
        辞書形式に変換

        Returns:
            辞書形式のデータ
        """
        return {
            "entry_type": (
                self.entry_type.value
                if isinstance(self.entry_type, EntryType)
                else self.entry_type
            ),
            "limit_offset_pct": self.limit_offset_pct,
            "stop_offset_pct": self.stop_offset_pct,
            "order_validity_bars": self.order_validity_bars,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntryGene":
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
                    0.0, min(0.1, gene.limit_offset_pct * random.uniform(0.8, 1.2))
                )
                gene.stop_offset_pct = max(
                    0.0, min(0.1, gene.stop_offset_pct * random.uniform(0.8, 1.2))
                )

            # 有効期限の変更
            if random.random() < 0.3:
                gene.order_validity_bars = max(
                    1,
                    min(
                        100,
                        int(gene.order_validity_bars * random.uniform(0.8, 1.2)),
                    ),
                )

        return gene

    @classmethod
    def crossover(
        cls, parent1: "EntryGene", parent2: "EntryGene"
    ) -> Tuple["EntryGene", "EntryGene"]:
        """エントリー遺伝子の交叉"""
        import random

        # 単純なフィールドごとのユニフォーム交叉
        c1_params = {}
        c2_params = {}

        fields = [
            "entry_type",
            "limit_offset_pct",
            "stop_offset_pct",
            "order_validity_bars",
            "enabled",
            "priority",
        ]

        for field in fields:
            val1 = getattr(parent1, field)
            val2 = getattr(parent2, field)

            if random.random() < 0.5:
                c1_params[field] = val1
                c2_params[field] = val2
            else:
                c1_params[field] = val2
                c2_params[field] = val1

        return cls(**c1_params), cls(**c2_params)


def create_random_entry_gene(config: Any = None) -> EntryGene:
    """
    ランダムなエントリー遺伝子を生成

    Args:
        config: GA設定オブジェクト（オプション）

    Returns:
        生成されたエントリー遺伝子
    """
    import random

    # エントリータイプの出現確率（初期段階では成行注文を多めに設定）
    default_weights = {
        EntryType.MARKET: 0.6,
        EntryType.LIMIT: 0.2,
        EntryType.STOP: 0.15,
        EntryType.STOP_LIMIT: 0.05,
    }

    try:
        # 重み付きでエントリータイプを選択
        weights = default_weights
        if config and hasattr(config, "entry_type_weights"):
            custom_weights = config.entry_type_weights
            if custom_weights:
                weights = {EntryType(k): v for k, v in custom_weights.items()}

        types = list(weights.keys())
        type_weights = list(weights.values())

        entry_type = random.choices(types, weights=type_weights, k=1)[0]

        # オフセット値をランダム生成（0.1% ~ 2.0%）
        limit_offset_pct = random.uniform(0.001, 0.02)
        stop_offset_pct = random.uniform(0.001, 0.02)

        # 有効期限をランダム生成（1 ~ 20バー）
        order_validity_bars = random.randint(1, 20)

        return EntryGene(
            entry_type=entry_type,
            limit_offset_pct=limit_offset_pct,
            stop_offset_pct=stop_offset_pct,
            order_validity_bars=order_validity_bars,
            enabled=True,
            priority=random.uniform(0.5, 1.5),
        )

    except (ValueError, TypeError, KeyError, AttributeError):
        # フォールバック: 成行注文のデフォルト遺伝子
        return EntryGene(
            entry_type=EntryType.MARKET,
            limit_offset_pct=0.005,
            stop_offset_pct=0.005,
            order_validity_bars=5,
            enabled=True,
        )
