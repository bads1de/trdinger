"""
機械学習サービスモジュール

本モジュールは、学習・評価・保存などML機能のエントリポイントを提供します。
実装の詳細や最適化手法には踏み込まず、利用側からの統一的なアクセスを目的とします。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestration.ml_training_orchestration_service import MLTrainingService

_ATTRIBUTE_EXPORTS = {
    "MLTrainingService": ".orchestration.ml_training_orchestration_service",
}


def __getattr__(name: str) -> type:
    """遅延インポートで循環参照を回避"""
    module_path = _ATTRIBUTE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_ATTRIBUTE_EXPORTS})


__all__ = ["MLTrainingService"]
