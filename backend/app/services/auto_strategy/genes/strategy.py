"""
戦略遺伝子モデル

GA（遺伝的アルゴリズム）によって進化させる取引戦略の設計図を表します。
インジケーター、エントリー条件、エグジット条件、リスク管理などの
遺伝子を統合し、交叉・突然変異・クローン操作をサポートします。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

from app.types import SerializableValue

from .conditions import Condition, ConditionGroup, StatefulCondition
from .entry import EntryGene
from .exit import ExitGene
from .indicator import IndicatorGene
from .position_sizing import PositionSizingGene
from .tool import ToolGene
from .tpsl import TPSLGene

if TYPE_CHECKING:
    from ..config.ga.ga_config import GAConfig

logger = logging.getLogger(__name__)


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
    indicators: List[IndicatorGene] = field(default_factory=list)
    long_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    short_entry_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    stateful_conditions: List[StatefulCondition] = field(default_factory=list)
    risk_management: Dict[str, SerializableValue] = field(default_factory=dict)
    tpsl_gene: Optional[TPSLGene] = None
    long_tpsl_gene: Optional[TPSLGene] = None
    short_tpsl_gene: Optional[TPSLGene] = None
    position_sizing_gene: Optional[PositionSizingGene] = None
    entry_gene: Optional[EntryGene] = None
    long_entry_gene: Optional[EntryGene] = None
    short_entry_gene: Optional[EntryGene] = None
    exit_gene: Optional[ExitGene] = None
    long_exit_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    short_exit_conditions: List[Union[Condition, ConditionGroup]] = field(
        default_factory=list
    )
    tool_genes: List[ToolGene] = field(default_factory=list)
    metadata: Dict[str, SerializableValue] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> "StrategyGene":
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
        indicators: List[IndicatorGene],
        long_entry_conditions: List[Union[Condition, ConditionGroup]],
        short_entry_conditions: List[Union[Condition, ConditionGroup]],
        tpsl_gene: Optional[TPSLGene] = None,
        position_sizing_gene: Optional[PositionSizingGene] = None,
        long_tpsl_gene: Optional[TPSLGene] = None,
        short_tpsl_gene: Optional[TPSLGene] = None,
        entry_gene: Optional[EntryGene] = None,
        long_entry_gene: Optional[EntryGene] = None,
        short_entry_gene: Optional[EntryGene] = None,
        exit_gene: Optional[ExitGene] = None,
        long_exit_conditions: Optional[
            List[Union[Condition, ConditionGroup]]
        ] = None,
        short_exit_conditions: Optional[
            List[Union[Condition, ConditionGroup]]
        ] = None,
        tool_genes: Optional[List[ToolGene]] = None,
        risk_management: Optional[Dict[str, SerializableValue]] = None,
        metadata: Optional[Dict[str, SerializableValue]] = None,
    ) -> "StrategyGene":
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

    def has_long_short_separation(self) -> bool:
        """ロング・ショート条件が分離されているかチェック（常にTrue）。"""
        return True

    def validate(self) -> Tuple[bool, List[str]]:
        """戦略遺伝子の妥当性を検証し、(is_valid, errors) を返す。"""
        from .validator import GeneValidator

        validator = GeneValidator()
        is_valid, errors = validator.validate_strategy_gene(self)
        return is_valid, errors

    @classmethod
    def clone_field_names(cls) -> Tuple[str, ...]:
        """clone 対象となるフィールド名を返す。"""
        return tuple(
            field_info.name for field_info in fields(cls) if field_info.name != "id"  # type: ignore[arg-type]
        )

    @classmethod
    def crossover_field_names(cls) -> Tuple[str, ...]:
        """uniform crossover 対象となるフィールド名を返す。"""
        return tuple(
            name for name in cls.clone_field_names() if name != "metadata"
        )

    @classmethod
    def sub_gene_field_names(cls) -> Tuple[str, ...]:
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
    def sub_gene_class_map(cls) -> Dict[str, type]:
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

    def clone(self, keep_id: bool = False) -> "StrategyGene":
        """軽量コピーを作成。"""
        from .genetic_utils import GeneticUtils

        cloned_fields = {
            field_name: GeneticUtils.smart_copy(getattr(self, field_name))
            for field_name in self.clone_field_names()
        }
        cloned_fields["id"] = self.id if keep_id else str(uuid.uuid4())
        return type(self)(**cast(Dict[str, SerializableValue], cloned_fields))  # type: ignore[arg-type]

    def mutate(
        self, config: GAConfig, mutation_rate: float = 0.1
    ) -> "StrategyGene":
        """戦略遺伝子の突然変異を実行する。"""
        from .operators import mutate_strategy_gene

        return mutate_strategy_gene(self, config, mutation_rate=mutation_rate)

    def adaptive_mutate(
        self,
        population: List["StrategyGene"],
        config: GAConfig,
        base_mutation_rate: float = 0.1,
    ) -> "StrategyGene":
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
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: GAConfig,
        crossover_type: str = "uniform",
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """2つの親個体から交叉により新しい子個体を生成する。"""
        from .operators import crossover_strategy_genes

        return crossover_strategy_genes(
            cls,
            parent1,
            parent2,
            config,
            crossover_type=crossover_type,
        )

    @staticmethod
    def _mutate_indicators(
        mutated: "StrategyGene", mutation_rate: float, config: GAConfig
    ) -> None:
        """指標遺伝子の突然変異処理。"""
        from .operators import mutate_indicators

        mutate_indicators(mutated, mutation_rate, config)

    @staticmethod
    def _mutate_conditions(
        mutated: "StrategyGene", mutation_rate: float, config: GAConfig
    ) -> None:
        """取引条件の突然変異処理。"""
        from .operators import mutate_conditions

        mutate_conditions(mutated, mutation_rate, config)

    @staticmethod
    def _crossover_tpsl_genes(
        parent1_tpsl: Optional[TPSLGene],
        parent2_tpsl: Optional[TPSLGene],
    ) -> Tuple[Optional[TPSLGene], Optional[TPSLGene]]:
        """TPSL 遺伝子の交叉処理。"""
        from .operators import crossover_tpsl_genes

        return crossover_tpsl_genes(parent1_tpsl, parent2_tpsl)

    @staticmethod
    def _crossover_position_sizing_genes(
        parent1_ps: Optional[PositionSizingGene],
        parent2_ps: Optional[PositionSizingGene],
    ) -> Tuple[Optional[PositionSizingGene], Optional[PositionSizingGene]]:
        """ポジションサイズ遺伝子の交叉処理。"""
        from .operators import crossover_position_sizing_genes

        return crossover_position_sizing_genes(parent1_ps, parent2_ps)

    @staticmethod
    def _crossover_entry_genes(
        parent1_entry: Optional[EntryGene],
        parent2_entry: Optional[EntryGene],
    ) -> Tuple[Optional[EntryGene], Optional[EntryGene]]:
        """エントリー遺伝子の交叉処理。"""
        from .operators import crossover_entry_genes

        return crossover_entry_genes(parent1_entry, parent2_entry)

    @classmethod
    def _uniform_crossover(
        cls,
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: GAConfig,
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """ユニフォーム交叉。"""
        from .operators import uniform_crossover

        return uniform_crossover(cls, parent1, parent2, config)

    @classmethod
    def _single_point_crossover(
        cls,
        parent1: "StrategyGene",
        parent2: "StrategyGene",
        config: GAConfig,
    ) -> Tuple["StrategyGene", "StrategyGene"]:
        """一点交叉。"""
        from .operators import single_point_crossover

        return single_point_crossover(cls, parent1, parent2, config)
