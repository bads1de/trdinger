"""
機械学習サービスモジュール

本モジュールは、学習・評価・保存などML機能のエントリポイントを提供します。
実装の詳細や最適化手法には踏み込まず、利用側からの統一的なアクセスを目的とします。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestration.ml_training_orchestration_service import (
        MLTrainingService,
    )

_ATTRIBUTE_EXPORTS = {
    "MLTrainingService": ".orchestration.ml_training_orchestration_service",
}

__all__ = ["MLTrainingService"]

from app.utils.lazy_import import setup_lazy_import  # noqa: E402

setup_lazy_import(globals(), _ATTRIBUTE_EXPORTS)
