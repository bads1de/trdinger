"""
バックテスト設定パッケージ

バックテスト関連の設定スキーマと定数を提供します。
"""

from .backtest_config import (
    BacktestConfig,
    BacktestConfigValidationError,
    GeneratedGAParameters,
    StrategyConfig,
)
from .constants import SUPPORTED_STRATEGIES

__all__ = [
    "BacktestConfig",
    "BacktestConfigValidationError",
    "GeneratedGAParameters",
    "StrategyConfig",
    "SUPPORTED_STRATEGIES",
]
