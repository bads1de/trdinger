"""
Auto Strategy Utils パッケージ

戦略生成・評価・統合に関連する共通ユーティリティを提供します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .normalization import NormalizationUtils, create_default_strategy_gene

_ATTRIBUTE_EXPORTS = {
    "NormalizationUtils": ".normalization",
    "create_default_strategy_gene": ".normalization",
}

__all__ = [
    # Core Utilities
    "NormalizationUtils",
    # Utility functions
    "create_default_strategy_gene",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
