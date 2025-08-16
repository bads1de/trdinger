"""
AutoMLメモリ最適化ユーティリティ

メモリ効率的なAutoML処理のためのヘルパー関数を提供します。
"""

import functools
import gc
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


def get_system_memory_info() -> Dict[str, float]:
    """
    システムメモリ情報を取得

    Returns:
        メモリ情報辞書
    """
    try:
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent,
            "process_mb": process.memory_info().rss / (1024**2),
        }
    except Exception as e:
        logger.error(f"メモリ情報取得エラー: {e}")
        return {}








@contextmanager
def memory_efficient_processing(operation_name: str = "処理"):
    """
    メモリ効率的な処理のためのコンテキストマネージャー

    Args:
        operation_name: 操作名
    """
    # 開始時のメモリ使用量
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    logger.debug(f"{operation_name}開始時メモリ: {start_memory:.2f}MB")

    try:
        yield
    finally:
        # 終了時のクリーンアップ
        gc.collect()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_diff = end_memory - start_memory
        logger.debug(
            f"{operation_name}終了時メモリ: {end_memory:.2f}MB (差分: {memory_diff:+.2f}MB)"
        )








def estimate_autofeat_memory_usage(
    data_size_mb: float, num_features: int, feateng_steps: int
) -> float:
    """
    AutoFeatのメモリ使用量を推定

    Args:
        data_size_mb: 入力データサイズ（MB）
        num_features: 特徴量数
        feateng_steps: 特徴量エンジニアリングステップ数

    Returns:
        推定メモリ使用量（MB）
    """
    # 基本的な推定式（経験的な値）
    base_memory = data_size_mb * 2  # 基本的にデータサイズの2倍
    feature_factor = num_features * 0.1  # 特徴量数による係数
    step_factor = feateng_steps * 1.5  # ステップ数による係数

    estimated_memory = base_memory * (1 + feature_factor) * step_factor

    # 最小値と最大値の制限
    return max(100, min(estimated_memory, 8000))  # 100MB〜8GB









