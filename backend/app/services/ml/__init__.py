"""
機械学習サービスモジュール

本モジュールは、学習・評価・保存などML機能のエントリポイントを提供します。
実装の詳細や最適化手法には踏み込まず、利用側からの統一的なアクセスを目的とします。
"""

# 循環参照を防ぐため、ここではインポートを行わない
# from .ml_training_service import MLTrainingService

__all__ = ["MLTrainingService"]
