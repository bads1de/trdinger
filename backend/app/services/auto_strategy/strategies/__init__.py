"""
自動戦略実行パッケージ

戦略遺伝子を実際の取引ロジックに変換し、
エントリー・エグジット・ポジション管理を実行します。

主なコンポーネント:
- universal_strategy.py: 統一戦略クラス（StrategyGeneを実行可能な形式に変換）
- entry_decision_engine.py: エントリー判断エンジン
- position_exit_engine.py: ポジション決済エンジン
- position_manager.py: ポジション管理
- order_manager.py: 注文管理
- ml_filter.py: MLモデルによるフィルター
- stateful_conditions.py: 状態依存条件評価
- runtime_state.py: ランタイム状態管理
"""

from .universal_strategy import UniversalStrategy
from .early_termination import StrategyEarlyTermination, StrategyEarlyTerminationController
from .entry_decision_engine import EntryDecisionEngine
from .execution_cycle import StrategyExecutionCycle
from .position_exit_engine import PositionExitEngine
from .position_manager import PositionManager
from .order_manager import OrderManager
from .strategy_initializer import StrategyInitializer

__all__ = [
    "UniversalStrategy",
    "StrategyEarlyTermination",
    "StrategyEarlyTerminationController",
    "StrategyInitializer",
    "EntryDecisionEngine",
    "StrategyExecutionCycle",
    "PositionExitEngine",
    "PositionManager",
    "OrderManager",
]
