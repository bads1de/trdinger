"""
機械学習サービスモジュール

本モジュールは、学習・評価・保存などML機能のエントリポイントを提供します。
実装の詳細や最適化手法には踏み込まず、利用側からの統一的なアクセスを目的とします。
"""

# 循環参照を防ぐため、ここではインポートを行わない
# from .orchestration.ml_training_service import MLTrainingService

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """遅延インポートで循環参照を回避"""
    if name == "MLTrainingService":
        from .orchestration.ml_training_orchestration_service import MLTrainingService

        return MLTrainingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MLTrainingService"]
