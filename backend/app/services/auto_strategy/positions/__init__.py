"""
ポジションサイジングパッケージ

ポジションサイズ計算に関連する機能を統合します。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entry_executor import EntryExecutor
    from .position_sizing_service import (
        PositionSizingResult,
        PositionSizingService,
    )

_ATTRIBUTE_EXPORTS = {
    "PositionSizingService": ".position_sizing_service",
    "PositionSizingResult": ".position_sizing_service",
    "EntryExecutor": ".entry_executor",
}

__all__ = [
    "PositionSizingService",
    "PositionSizingResult",
    "EntryExecutor",
]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
