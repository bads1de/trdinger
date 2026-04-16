"""
自動戦略実行パッケージ

戦略遺伝子を実際の取引ロジックに変換し、
エントリー・エグジット・ポジション管理を実行します。

主なコンポーネント:
- universal_strategy.py: 統一戦略クラス（StrategyGeneを実行可能な形式に変換）
- entry_decision_engine.py: エントリー判断エンジン
- position_manager.py: ポジション管理
- order_manager.py: 注文管理
- ml_filter.py: MLモデルによるフィルター
- stateful_conditions.py: 状態依存条件評価
- runtime_state.py: ランタイム状態管理
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .early_termination import (
        StrategyEarlyTermination,
        StrategyEarlyTerminationController,
    )
    from .entry_decision_engine import EntryDecisionEngine
    from .execution_cycle import StrategyExecutionCycle
    from .order_manager import OrderManager
    from .position_manager import PositionManager
    from .strategy_initializer import StrategyInitializer
    from .universal_strategy import UniversalStrategy

_ATTRIBUTE_EXPORTS = {
    "UniversalStrategy": ".universal_strategy",
    "StrategyEarlyTermination": ".early_termination",
    "StrategyEarlyTerminationController": ".early_termination",
    "StrategyInitializer": ".strategy_initializer",
    "EntryDecisionEngine": ".entry_decision_engine",
    "StrategyExecutionCycle": ".execution_cycle",
    "PositionManager": ".position_manager",
    "OrderManager": ".order_manager",
}

__all__ = [
    "UniversalStrategy",
    "StrategyEarlyTermination",
    "StrategyEarlyTerminationController",
    "StrategyInitializer",
    "EntryDecisionEngine",
    "StrategyExecutionCycle",
    "PositionManager",
    "OrderManager",
]

from .._lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS, __all__)
