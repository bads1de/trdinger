"""
Auto Strategy Generators モジュール

戦略生成に関連するジェネレータークラスを提供します。
factories/ の機能を統合しています。
"""

from .gene_factory import GeneGeneratorFactory
from .random_gene_generator import RandomGeneGenerator
from .smart_condition_generator import SmartConditionGenerator
# ConditionGenerator is now integrated into SmartConditionGenerator for backward compatibility
ConditionGenerator = SmartConditionGenerator

# TPSL ジェネレーター (RiskReward, Statistical, Volatility ジェネレーターを統合)
from ..tpsl.generator import UnifiedTPSLGenerator

# factories からの統合
from .strategy_factory import StrategyFactory

__all__ = [
    "GeneGeneratorFactory",
    "RandomGeneGenerator",
    "SmartConditionGenerator",
    "ConditionGenerator",  # Phase 1.3: 条件生成統合
    "UnifiedTPSLGenerator",  # 統合 TPSL ジェネレーター
    # Factories
    "StrategyFactory",
]
