"""
イグジット注文遺伝子

イグジット注文のタイプとパラメータを定義します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from app.utils.serialization import dataclass_to_dict

from ..config.constants import ExitType
from .gene_constants import (
    EXIT_PARTIAL_ENABLED_PROBABILITY,
    EXIT_TRAILING_ACTIVATION_PROBABILITY,
    EXIT_TYPE_WEIGHTS,
    PRIORITY_GENERATION_RANGE,
)
from .gene_ranges import EXIT_GENERATION_RANGES


@dataclass(slots=True)
class ExitGene:
    """
    イグジット注文遺伝子

    GAによって最適化されるイグジット注文のパラメータを定義します。
    """

    # 遺伝的操作のための設定
    NUMERIC_FIELDS = ["partial_exit_pct", "priority"]
    ENUM_FIELDS = ["exit_type"]
    CHOICE_FIELDS = ["partial_exit_enabled", "trailing_stop_activation", "enabled"]
    NUMERIC_RANGES = {
        "partial_exit_pct": (0.1, 0.9),
        "priority": (0.5, 1.5),
    }

    # イグジットタイプ
    exit_type: ExitType = ExitType.FULL

    # 部分決済の割合（0.1 〜 0.9）
    # exit_type が PARTIAL の場合、この割合でポジションを決済
    partial_exit_pct: float = 0.5  # 50%

    # 部分決済が有効か
    partial_exit_enabled: bool = False

    # 条件成立時にトレーリングSLを起動する（決済しない）
    trailing_stop_activation: bool = False

    # 有効/無効フラグ
    enabled: bool = True

    # 優先度（将来の拡張用）
    priority: float = 1.0

    def validate(self) -> Tuple[bool, List[str]]:
        """
        イグジット遺伝子の妥当性を検証

        Returns:
            (is_valid, errors) のタプル
        """
        errors: List[str] = []

        # partial_exit_pct の範囲チェック (10% ~ 90%)
        if self.partial_exit_pct < 0.1 or self.partial_exit_pct > 0.9:
            errors.append(
                f"partial_exit_pct は 0.1~0.9 の範囲である必要があります: {self.partial_exit_pct}"
            )

        # exit_type が有効な値かチェック
        if not isinstance(self.exit_type, ExitType):
            try:
                ExitType(self.exit_type)
            except ValueError:
                errors.append(f"無効な exit_type です: {self.exit_type}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        """
        辞書形式に変換

        Returns:
            辞書形式のデータ
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExitGene":
        """
        辞書形式から復元

        Args:
            data: 辞書形式のデータ

        Returns:
            ExitGene オブジェクト
        """
        exit_type_value = data.get("exit_type", "full")
        if isinstance(exit_type_value, str):
            exit_type = ExitType(exit_type_value)
        elif isinstance(exit_type_value, ExitType):
            exit_type = exit_type_value
        else:
            exit_type = ExitType.FULL

        return cls(
            exit_type=exit_type,
            partial_exit_pct=data.get("partial_exit_pct", 0.5),
            partial_exit_enabled=data.get("partial_exit_enabled", False),
            trailing_stop_activation=data.get("trailing_stop_activation", False),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1.0),
        )

    def clone(self) -> ExitGene:
        """軽量コピーを作成"""
        return ExitGene(
            exit_type=self.exit_type,
            partial_exit_pct=self.partial_exit_pct,
            partial_exit_enabled=self.partial_exit_enabled,
            trailing_stop_activation=self.trailing_stop_activation,
            enabled=self.enabled,
            priority=self.priority,
        )

    def mutate(self, mutation_rate: float = 0.1) -> "ExitGene":
        """突然変異"""
        import random

        gene = self.clone()

        if random.random() < mutation_rate:
            # タイプの変更
            if random.random() < 0.3:
                gene.exit_type = random.choice(list(ExitType))

            # 部分決済割合の摂動
            if random.random() < 0.5:
                gene.partial_exit_pct = max(
                    0.1, min(0.9, gene.partial_exit_pct * random.uniform(0.8, 1.2))
                )

            # フラグのトグル
            if random.random() < 0.2:
                gene.partial_exit_enabled = not gene.partial_exit_enabled
            if random.random() < 0.2:
                gene.trailing_stop_activation = not gene.trailing_stop_activation

        return gene

    @classmethod
    def crossover(
        cls, parent1: "ExitGene", parent2: "ExitGene"
    ) -> Tuple["ExitGene", "ExitGene"]:
        """イグジット遺伝子の交叉"""
        from .genetic_utils import GeneticUtils

        return GeneticUtils.crossover_generic_genes(
            parent1_gene=parent1,
            parent2_gene=parent2,
            gene_class=cls,
            numeric_fields=cls.NUMERIC_FIELDS,
            enum_fields=cls.ENUM_FIELDS,
            choice_fields=cls.CHOICE_FIELDS,
        )


def create_random_exit_gene(config: Any = None) -> ExitGene:
    """
    ランダムなイグジット遺伝子を生成

    Args:
        config: GA設定オブジェクト（オプション）

    Returns:
        生成されたイグジット遺伝子
    """
    import random

    default_weights = EXIT_TYPE_WEIGHTS
    ranges = EXIT_GENERATION_RANGES

    try:
        weights = default_weights
        if config and hasattr(config, "exit_type_weights"):
            custom_weights = config.exit_type_weights
            if custom_weights:
                weights = {ExitType(k): v for k, v in custom_weights.items()}

        types = list(weights.keys())
        type_weights = list(weights.values())

        exit_type = random.choices(types, weights=type_weights, k=1)[0]

        # 部分決済割合をランダム生成
        partial_exit_pct = random.uniform(*ranges["partial_exit_pct"])

        # フラグをランダムに設定
        partial_exit_enabled = random.random() < EXIT_PARTIAL_ENABLED_PROBABILITY
        trailing_stop_activation = random.random() < EXIT_TRAILING_ACTIVATION_PROBABILITY

        return ExitGene(
            exit_type=exit_type,
            partial_exit_pct=partial_exit_pct,
            partial_exit_enabled=partial_exit_enabled,
            trailing_stop_activation=trailing_stop_activation,
            enabled=True,
            priority=random.uniform(*PRIORITY_GENERATION_RANGE),
        )

    except (ValueError, TypeError, KeyError, AttributeError):
        # フォールバック: 全決済のデフォルト遺伝子
        fallback_ranges = EXIT_GENERATION_RANGES
        return ExitGene(
            exit_type=ExitType.FULL,
            partial_exit_pct=fallback_ranges["partial_exit_pct"][0],
            partial_exit_enabled=False,
            trailing_stop_activation=False,
            enabled=True,
        )
