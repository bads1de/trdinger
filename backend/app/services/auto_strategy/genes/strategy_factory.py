"""
StrategyGene 構築用のファクトリ関数。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from app.types import SerializableValue

from .conditions import Condition, ConditionGroup
from .entry import EntryGene
from .exit import ExitGene
from .indicator import IndicatorGene
from .position_sizing import PositionSizingGene, PositionSizingMethod
from .tool import ToolGene
from .tpsl import TPSLGene


def create_default_strategy_gene(strategy_gene_class):
    """デフォルトの StrategyGene を構築する。"""
    indicators = [IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True)]

    return strategy_gene_class(
        id=str(uuid.uuid4()),
        indicators=indicators,
        long_entry_conditions=[
            Condition(left_operand="close", operator=">", right_operand="open")
        ],
        short_entry_conditions=[
            Condition(left_operand="close", operator="<", right_operand="open")
        ],
        risk_management={"position_size": 0.1},
        tpsl_gene=TPSLGene(take_profit_pct=0.01, stop_loss_pct=0.005, enabled=True),
        position_sizing_gene=PositionSizingGene(
            method=PositionSizingMethod.FIXED_QUANTITY,
            fixed_quantity=1000,
            enabled=True,
        ),
        metadata={"generated_by": "create_default"},
    )


def assemble_strategy_gene(
    strategy_gene_class,
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
):
    """部品から StrategyGene を構築する。"""
    final_metadata = dict(metadata or {})
    final_metadata.setdefault("assembled_at", datetime.now().isoformat())

    return strategy_gene_class(
        id=str(uuid.uuid4()),
        indicators=indicators,
        long_entry_conditions=long_entry_conditions,
        short_entry_conditions=short_entry_conditions,
        tpsl_gene=tpsl_gene,
        long_tpsl_gene=long_tpsl_gene,
        short_tpsl_gene=short_tpsl_gene,
        position_sizing_gene=position_sizing_gene,
        entry_gene=entry_gene,
        long_entry_gene=long_entry_gene,
        short_entry_gene=short_entry_gene,
        exit_gene=exit_gene,
        long_exit_conditions=long_exit_conditions or [],
        short_exit_conditions=short_exit_conditions or [],
        tool_genes=tool_genes or [],
        risk_management=risk_management or {"position_size": 0.1},
        metadata=final_metadata,
    )
