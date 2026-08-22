"""
戦略遺伝子モデル

GA（遺伝的アルゴリズム）によって進化させる取引戦略の設計図を表します。
インジケーター、エントリー条件、エグジット条件、リスク管理などの
遺伝子を統合し、交叉・突然変異・クローン操作をサポートします。
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, cast

from app.types import SerializableValue

from ..utils.indicator_references import build_indicator_reference_name
from .conditions import Condition, ConditionGroup, StatefulCondition
from .entry import EntryGene
from .exit import ExitGene
from .indicator import IndicatorGene
from .position_sizing import PositionSizingGene
from .tool import ToolGene
from .tpsl import TPSLGene

if TYPE_CHECKING:
    from ..config.ga_config import GAConfig

logger = logging.getLogger(__name__)

_PRICE_OPERANDS = frozenset({"close", "open", "high", "low", "volume"})
_OUTPUT_INDEX_SUFFIX = re.compile(r"_\d+$")


def _operand_reference_name(operand: object) -> str | None:
    """条件オペランドから参照名を抽出する（数値リテラルは None）。"""
    if isinstance(operand, bool):
        return None
    if isinstance(operand, (int, float)):
        return None
    if isinstance(operand, str):
        return operand
    if isinstance(operand, dict):
        name = operand.get("name") or operand.get("indicator")
        return str(name) if name else None
    return None


def _is_resolvable_operand(operand_name: str, valid_names: set[str]) -> bool:
    """オペランド名が解決可能か（価格列・自身の指標列・数値リテラル）。"""
    try:
        float(operand_name)
        return True
    except ValueError:
        pass
    # 生成器には "Close" の大小両方が存在したため価格列はケース不感応に判定する
    if operand_name.lower() in _PRICE_OPERANDS or operand_name in valid_names:
        return True
    # 複数出力指標（MACD_xxxxxxxx_0 など）の出力インデックス接尾辞を外して再判定
    base = _OUTPUT_INDEX_SUFFIX.sub("", operand_name)
    return base != operand_name and base in valid_names


def _filter_condition_item(
    item: Condition | ConditionGroup, valid_names: set[str]
) -> tuple[Condition | ConditionGroup | None, int]:
    """単一条件または条件グループを参照整合性でフィルタする。

    Returns:
        (フィルタ後のアイテム。全滅した場合は None, 除去した条件・グループ数)
    """
    if isinstance(item, ConditionGroup):
        kept: list[Condition | ConditionGroup] = []
        dropped = 0
        for sub in item.conditions:
            filtered, sub_dropped = _filter_condition_item(sub, valid_names)
            dropped += sub_dropped
            if filtered is not None:
                kept.append(filtered)
        if not kept:
            return None, dropped + 1
        item.conditions = kept
        return item, dropped
    for operand in (item.left_operand, item.right_operand):
        name = _operand_reference_name(operand)
        if name is not None and not _is_resolvable_operand(name, valid_names):
            return None, 1
    return item, 0


@dataclass(slots=True)
class StrategyGene:
    """
    遺伝的アルゴリズム（GA）の進化対象となる、一つの完結した取引戦略を表す「遺伝子」です。

    このクラスは、テクニカル指標の定義からエントリー・決済条件、リスク管理、資金管理までの
    全パラメータを保持し、戦略の「設計図」として機能します。

    Attributes:
        id (str): 戦略の一意識別子（UUID）。
        indicators (List[IndicatorGene]): 戦略で使用する全テクニカル指標の定義。
        long_entry_conditions (List): ロングエントリーを許可するための論理条件。
        short_entry_conditions (List): ショートエントリーを許可するための論理条件。
        stateful_conditions (List): 状態（フラグ）を保持する特殊な条件。
        risk_management (Dict): 証拠金やレバレッジ等のリスク設定。
        tpsl_gene (Optional[TPSLGene]): 共通の利確・損切設定。
        long_tpsl_gene (Optional[TPSLGene]): ロング専用の利確・損切設定。
        short_tpsl_gene (Optional[TPSLGene]): ショート専用の利確・損切設定。
        position_sizing_gene (Optional[PositionSizingGene]): 資金管理（ロットサイズ決定）ロジックの設定。
        tool_genes (List[ToolGene]): 週末フィルタや時間帯制限等の補助ツールの設定。
        metadata (Dict): 生成日時や親のID等の付随情報。
    """

    id: str = ""
    indicators: list[IndicatorGene] = field(default_factory=list)
    long_entry_conditions: list[Condition | ConditionGroup] = field(
        default_factory=list
    )
    short_entry_conditions: list[Condition | ConditionGroup] = field(
        default_factory=list
    )
    stateful_conditions: list[StatefulCondition] = field(default_factory=list)
    risk_management: dict[str, SerializableValue] = field(default_factory=dict)
    tpsl_gene: TPSLGene | None = None
    long_tpsl_gene: TPSLGene | None = None
    short_tpsl_gene: TPSLGene | None = None
    position_sizing_gene: PositionSizingGene | None = None
    entry_gene: EntryGene | None = None
    long_entry_gene: EntryGene | None = None
    short_entry_gene: EntryGene | None = None
    exit_gene: ExitGene | None = None
    long_exit_conditions: list[Condition | ConditionGroup] = field(default_factory=list)
    short_exit_conditions: list[Condition | ConditionGroup] = field(
        default_factory=list
    )
    tool_genes: list[ToolGene] = field(default_factory=list)
    metadata: dict[str, SerializableValue] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> StrategyGene:
        """デフォルトの戦略遺伝子を作成する。

        安全なデフォルト値を持つStrategyGeneインスタンスを生成します。
        GAの初期集団生成やテスト用途に使用されます。

        Returns:
            StrategyGene: デフォルト値で初期化された戦略遺伝子。
        """
        from .strategy_factory import create_default_strategy_gene

        return create_default_strategy_gene(cls)

    @classmethod
    def assemble(
        cls,
        indicators: list[IndicatorGene],
        long_entry_conditions: list[Condition | ConditionGroup],
        short_entry_conditions: list[Condition | ConditionGroup],
        tpsl_gene: TPSLGene | None = None,
        position_sizing_gene: PositionSizingGene | None = None,
        long_tpsl_gene: TPSLGene | None = None,
        short_tpsl_gene: TPSLGene | None = None,
        entry_gene: EntryGene | None = None,
        long_entry_gene: EntryGene | None = None,
        short_entry_gene: EntryGene | None = None,
        exit_gene: ExitGene | None = None,
        long_exit_conditions: list[Condition | ConditionGroup] | None = None,
        short_exit_conditions: list[Condition | ConditionGroup] | None = None,
        tool_genes: list[ToolGene] | None = None,
        risk_management: dict[str, SerializableValue] | None = None,
        metadata: dict[str, SerializableValue] | None = None,
    ) -> StrategyGene:
        """
        個別の遺伝子パーツから、一つの完結した `StrategyGene` オブジェクトを組み立てます。

        このメソッドは主に、交叉（Crossover）や突然変異（Mutation）によって
        新しく生成された属性セットから個体を再構成する際に使用されます。

        Args:
            indicators: テクニカル指標のリスト。
            long_entry_conditions: ロングのエントリー条件。
            short_entry_conditions: ショートのエントリー条件。
            tpsl_gene: 共通のTP/SL設定。
            position_sizing_gene: 資金管理設定。
            long_tpsl_gene: ロング専用のTP/SL設定。
            short_tpsl_gene: ショート専用のTP/SL設定。
            entry_gene: 共通のエントリー管理設定。
            long_entry_gene: ロング専用のエントリー管理。
            short_entry_gene: ショート専用のエントリー管理。
            tool_genes: 補助ツールの設定リスト。
            risk_management: リスク管理の詳細設定辞書。
            metadata: 実験ID等のメタデータ。

        Returns:
            StrategyGene: 組み立てられた新しい戦略個体。
        """
        from .strategy_factory import assemble_strategy_gene

        return assemble_strategy_gene(
            cls,
            indicators=indicators,
            long_entry_conditions=long_entry_conditions,
            short_entry_conditions=short_entry_conditions,
            tpsl_gene=tpsl_gene,
            position_sizing_gene=position_sizing_gene,
            long_tpsl_gene=long_tpsl_gene,
            short_tpsl_gene=short_tpsl_gene,
            entry_gene=entry_gene,
            long_entry_gene=long_entry_gene,
            short_entry_gene=short_entry_gene,
            exit_gene=exit_gene,
            long_exit_conditions=long_exit_conditions,
            short_exit_conditions=short_exit_conditions,
            tool_genes=tool_genes,
            risk_management=risk_management,
            metadata=metadata,
        )

    def validate(self) -> tuple[bool, list[str]]:
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す。"""
        from .validator import GeneValidator

        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors

    @classmethod
    def clone_field_names(cls) -> tuple[str, ...]:
        """clone 対象となるフィールド名を返す。"""
        return tuple(
            field_info.name for field_info in fields(cls) if field_info.name != "id"
        )

    @classmethod
    def crossover_field_names(cls) -> tuple[str, ...]:
        """uniform crossover 対象となるフィールド名を返す。"""
        return tuple(name for name in cls.clone_field_names() if name != "metadata")

    @classmethod
    def sub_gene_field_names(cls) -> tuple[str, ...]:
        """StrategyGene が保持するサブ遺伝子のフィールド名を返す。"""
        return (
            "tpsl_gene",
            "long_tpsl_gene",
            "short_tpsl_gene",
            "position_sizing_gene",
            "entry_gene",
            "long_entry_gene",
            "short_entry_gene",
            "exit_gene",
        )

    @classmethod
    def sub_gene_class_map(cls) -> dict[str, type]:
        """サブ遺伝子フィールドと対応クラスの対応表を返す。"""
        return {
            "tpsl_gene": TPSLGene,
            "long_tpsl_gene": TPSLGene,
            "short_tpsl_gene": TPSLGene,
            "position_sizing_gene": PositionSizingGene,
            "entry_gene": EntryGene,
            "long_entry_gene": EntryGene,
            "short_entry_gene": EntryGene,
            "exit_gene": ExitGene,
        }

    def clone(self, keep_id: bool = False) -> StrategyGene:
        """軽量コピーを作成。"""
        from .genetic_utils import GeneticUtils

        cloned_fields = {
            field_name: GeneticUtils.smart_copy(getattr(self, field_name))
            for field_name in self.clone_field_names()
        }
        cloned_fields["id"] = self.id if keep_id else str(uuid.uuid4())
        return type(self)(**cast(dict[str, SerializableValue], cloned_fields))  # type: ignore[arg-type]

    def repair_non_price_indicators(self, config: GAConfig) -> StrategyGene:
        """
        非価格指標（OI/FR/LSR由来）の最低数を確保する。

        突然変異・交叉で非価格指標が失われた場合、設定された
        ``min_non_price_indicators`` を満たすよう不足分を補充する。
        """
        min_non_price = getattr(config, "min_non_price_indicators", 0) or 0
        if min_non_price <= 0:
            return self

        from ..config.indicator_universe import get_non_price_indicator_names
        from .indicator import _ensure_min_non_price_indicators

        non_price_names = set(get_non_price_indicator_names(config))
        if not non_price_names:
            return self

        if (
            sum(1 for ind in self.indicators if ind.type in non_price_names)
            >= min_non_price
        ):
            return self

        self.indicators = _ensure_min_non_price_indicators(self.indicators, config)
        return self

    def valid_operand_names(self) -> set[str]:
        """条件オペランドとして解決可能な自身の指標名の集合を返す。"""
        names: set[str] = set()
        for indicator in self.indicators:
            names.add(indicator.type)
            timeframe = getattr(indicator, "timeframe", None)
            if timeframe:
                names.add(f"{indicator.type}_{timeframe}")
            names.add(build_indicator_reference_name(indicator))
        return names

    def repair_condition_references(self) -> int:
        """
        自身の指標リストに存在しない参照を含む条件を除去する。

        交叉は指標と条件を別々の親から混ぜ、突然変異は指標を除去するため、
        条件が存在しない指標列（dangling reference）を参照しうる。評価器は
        未知の列を実質 0 扱いにするため、放置すると ``close > 未知列`` が
        常に真となり、毎バーエントリーするだけの個体が選抜され続ける。
        演算子適用直後に本メソッドを呼び、参照整合性を保証する。

        Returns:
            除去した条件・条件グループ・ステートフル条件の合計数。
        """
        valid_names = self.valid_operand_names()
        removed = 0
        for attr in (
            "long_entry_conditions",
            "short_entry_conditions",
            "long_exit_conditions",
            "short_exit_conditions",
        ):
            conditions = getattr(self, attr)
            kept: list[Condition | ConditionGroup] = []
            for item in conditions:
                filtered, dropped = _filter_condition_item(item, valid_names)
                removed += dropped
                if filtered is not None:
                    kept.append(filtered)
            setattr(self, attr, kept)

        kept_stateful: list[StatefulCondition] = []
        for stateful in self.stateful_conditions:
            trigger_ok, _ = _filter_condition_item(
                stateful.trigger_condition, valid_names
            )
            follow_ok, _ = _filter_condition_item(
                stateful.follow_condition, valid_names
            )
            if trigger_ok is not None and follow_ok is not None:
                kept_stateful.append(stateful)
            else:
                removed += 1
        self.stateful_conditions = kept_stateful
        return removed

    def ensure_min_entry_conditions(self, min_conditions: int = 1) -> int:
        """
        エントリー条件が空の場合に最低数を保証する。

        repair_condition_references 後に呼ぶことで、参照切れ除去で 0 本に
        なった個体の無効評価を防ぐ。
        """
        if min_conditions <= 0 or not self.indicators:
            return 0
        from ..utils.scale_compat import default_threshold_for, is_close_comparable_type

        added = 0
        for attr, side in (
            ("long_entry_conditions", "long"),
            ("short_entry_conditions", "short"),
        ):
            conditions = getattr(self, attr)
            if len(conditions) >= min_conditions:
                continue
            enabled_inds = [
                ind for ind in self.indicators if ind.enabled
            ] or self.indicators
            if not enabled_inds:
                continue
            import random

            while len(getattr(self, attr)) < min_conditions:
                ind = random.choice(enabled_inds)
                ref_name = build_indicator_reference_name(ind)
                if is_close_comparable_type(ind.type):
                    op = ">" if side == "long" else "<"
                    cond = Condition(
                        left_operand="close", operator=op, right_operand=ref_name
                    )
                else:
                    thresh = default_threshold_for(ind.type, bullish=(side == "long"))
                    op = "<" if side == "long" else ">"
                    cond = Condition(
                        left_operand=ref_name, operator=op, right_operand=thresh
                    )
                getattr(self, attr).append(cond)
                added += 1
        return added

    def repair_condition_scales(self) -> int:
        """
        価格オペランドと価格と同尺度でない指標の比較を閾値比較へ書き直す。

        スケールガード導入前に保存されたシード遺伝子には
        ``close > FUNDING_RATE_LEVEL`` のような直接比較（値域が価格と
        桁違いのため恒真/恒偽になる）が残っており、交叉で新一代に持ち
        込まれる。指標側のオペランドを左辺に残し、右辺をスケール既定の
        閾値（レジストリ normal プロファイル優先）へ置き換える。

        Returns:
            書き直した条件の合計数。
        """
        from ..utils.scale_compat import (
            default_threshold_for,
            is_bullish_operator,
            is_close_comparable_type,
            is_price_operand,
        )

        name_to_gene: dict[str, IndicatorGene] = {}
        for indicator in self.indicators:
            name_to_gene.setdefault(indicator.type, indicator)
            timeframe = getattr(indicator, "timeframe", None)
            if timeframe:
                name_to_gene.setdefault(f"{indicator.type}_{timeframe}", indicator)
            name_to_gene[build_indicator_reference_name(indicator)] = indicator

        def _match_gene(operand: object) -> IndicatorGene | None:
            name = _operand_reference_name(operand)
            if name is None:
                return None
            gene = name_to_gene.get(name)
            if gene is None:
                base = _OUTPUT_INDEX_SUFFIX.sub("", name)
                gene = name_to_gene.get(base)
            return gene

        def _fix_leaf(condition: Condition) -> bool:
            for price_attr, other_attr in (
                ("left_operand", "right_operand"),
                ("right_operand", "left_operand"),
            ):
                if not is_price_operand(getattr(condition, price_attr)):
                    continue
                other = getattr(condition, other_attr)
                other_name = _operand_reference_name(other)
                if other_name is None:
                    continue
                gene = _match_gene(other)
                if gene is None or is_close_comparable_type(gene.type):
                    continue
                condition.left_operand = other_name
                condition.right_operand = default_threshold_for(
                    gene.type, bullish=is_bullish_operator(condition.operator)
                )
                return True
            return False

        def _fix_item(item: Condition | ConditionGroup) -> int:
            if isinstance(item, ConditionGroup):
                return sum(_fix_item(sub) for sub in item.conditions)
            return 1 if _fix_leaf(item) else 0

        repaired = 0
        for attr in (
            "long_entry_conditions",
            "short_entry_conditions",
            "long_exit_conditions",
            "short_exit_conditions",
        ):
            repaired += sum(_fix_item(item) for item in getattr(self, attr))

        for stateful in self.stateful_conditions:
            repaired += _fix_item(stateful.trigger_condition)
            repaired += _fix_item(stateful.follow_condition)
        return repaired

    def mutate(self, config: GAConfig, mutation_rate: float = 0.1) -> StrategyGene:
        """戦略遺伝子の突然変異を実行する。"""
        from .operators import mutate_strategy_gene

        mutated = mutate_strategy_gene(self, config, mutation_rate=mutation_rate)
        return mutated.repair_non_price_indicators(config)

    def adaptive_mutate(
        self,
        population: list[StrategyGene],
        config: GAConfig,
        base_mutation_rate: float = 0.1,
    ) -> StrategyGene:
        """集団の多様性に基づいて変異率を調整する。"""
        from .operators import adaptive_mutate_strategy_gene

        return adaptive_mutate_strategy_gene(
            self,
            population,
            config,
            base_mutation_rate=base_mutation_rate,
        )

    @classmethod
    def crossover(
        cls,
        parent1: StrategyGene,
        parent2: StrategyGene,
        config: GAConfig,
        crossover_type: str = "uniform",
    ) -> tuple[StrategyGene, StrategyGene]:
        """2つの親個体から交叉により新しい子個体を生成する。"""
        from .operators import crossover_strategy_genes

        child1, child2 = crossover_strategy_genes(
            cls,
            parent1,
            parent2,
            config,
            crossover_type=crossover_type,
        )
        child1.repair_non_price_indicators(config)
        child2.repair_non_price_indicators(config)
        # repair_non_price_indicators は max_indicators 到達時に価格指標を
        # 置換するため、交叉内で修復済みの条件参照が切れうる。最後に再修復する。
        child1.repair_condition_references()
        child2.repair_condition_references()
        return child1, child2
