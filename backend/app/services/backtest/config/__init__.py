"""
バックテスト設定パッケージ

バックテスト関連の設定スキーマと定数を提供します。
"""

from .backtest_settings import BacktestConfig
from .backtest_config import (
    BacktestRunConfig,
    BacktestRunConfigValidationError,
    GeneratedGAParameters,
    StrategyConfig,
)
from .constants import SUPPORTED_STRATEGIES

__all__ = [
    "BacktestConfig",
    "BacktestRunConfig",
    "BacktestRunConfigValidationError",
    "GeneratedGAParameters",
    "StrategyConfig",
    "SUPPORTED_STRATEGIES",
]
