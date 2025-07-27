"""
AutoMLメモリ最適化ユーティリティ

メモリ効率的なAutoML処理のためのヘルパー関数を提供します。
"""

import gc
import logging
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
from contextlib import contextmanager
import functools

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


def calculate_optimal_batch_size(
    data_size_mb: float, available_memory_gb: float
) -> int:
    """
    データサイズと利用可能メモリに基づいて最適なバッチサイズを計算

    Args:
        data_size_mb: データサイズ（MB）
        available_memory_gb: 利用可能メモリ（GB）

    Returns:
        最適なバッチサイズ
    """
    # 利用可能メモリの50%を使用する前提
    usable_memory_mb = available_memory_gb * 1024 * 0.5

    # データサイズに基づく基本バッチサイズ
    if data_size_mb < 50:
        base_batch_size = 5000
    elif data_size_mb < 200:
        base_batch_size = 3000
    elif data_size_mb < 500:
        base_batch_size = 2000
    else:
        base_batch_size = 1000

    # メモリ制約に基づく調整
    memory_factor = min(1.0, usable_memory_mb / data_size_mb)
    optimal_batch_size = int(base_batch_size * memory_factor)

    # 最小値と最大値の制限
    return max(500, min(optimal_batch_size, 10000))


def optimize_dataframe_dtypes(
    df: pd.DataFrame, aggressive: bool = False
) -> pd.DataFrame:
    """
    DataFrameのデータ型を最適化してメモリ使用量を削減

    Args:
        df: 最適化するDataFrame
        aggressive: より積極的な最適化を行うか

    Returns:
        最適化されたDataFrame
    """
    try:
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            # 数値型の最適化
            if pd.api.types.is_integer_dtype(col_type):
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)

            elif pd.api.types.is_float_dtype(col_type):
                if aggressive or col_type == "float64":
                    # float32で十分な精度かチェック
                    optimized_df[col] = pd.to_numeric(
                        optimized_df[col], downcast="float"
                    )

            # オブジェクト型の最適化
            elif col_type == "object":
                # 文字列の場合、カテゴリカルに変換可能かチェック
                if optimized_df[col].dtype == "object":
                    num_unique_values = len(optimized_df[col].unique())
                    num_total_values = len(optimized_df[col])

                    # ユニーク値が50%未満の場合はカテゴリカルに変換
                    if num_unique_values / num_total_values < 0.5:
                        optimized_df[col] = optimized_df[col].astype("category")

        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100

        if reduction > 1:  # 1%以上の削減があった場合のみログ出力
            logger.info(
                f"データ型最適化: {reduction:.1f}%削減 "
                f"({original_memory/1024/1024:.1f}MB → {optimized_memory/1024/1024:.1f}MB)"
            )

        return optimized_df

    except Exception as e:
        logger.error(f"データ型最適化エラー: {e}")
        return df


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
        collected = gc.collect()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_diff = end_memory - start_memory

        if abs(memory_diff) > 10:  # 10MB以上の変化があった場合のみログ出力
            logger.info(
                f"{operation_name}完了: メモリ変化 {memory_diff:+.2f}MB, "
                f"GC回収 {collected}オブジェクト"
            )


def memory_monitor_decorator(func: Callable) -> Callable:
    """
    メモリ使用量を監視するデコレータ

    Args:
        func: 監視対象の関数

    Returns:
        デコレートされた関数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with memory_efficient_processing(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def check_memory_availability(required_mb: float) -> bool:
    """
    必要なメモリが利用可能かチェック

    Args:
        required_mb: 必要なメモリ量（MB）

    Returns:
        利用可能かどうか
    """
    try:
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024

        # 安全マージンとして20%を確保
        safe_available_mb = available_mb * 0.8

        return safe_available_mb >= required_mb
    except Exception as e:
        logger.error(f"メモリ可用性チェックエラー: {e}")
        return False


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


def get_memory_efficient_autofeat_config(
    data_size_mb: float, target_memory_gb: float = 2.0
) -> Dict[str, Any]:
    """
    メモリ効率的なAutoFeat設定を取得

    Args:
        data_size_mb: データサイズ（MB）
        target_memory_gb: 目標メモリ使用量（GB）

    Returns:
        最適化されたAutoFeat設定
    """
    target_memory_mb = target_memory_gb * 1024

    # データサイズに基づく基本設定
    if data_size_mb > 500:  # 大量データ
        config = {
            "max_features": 30,
            "feateng_steps": 1,
            "max_gb": min(1.0, target_memory_gb),
            "featsel_runs": 1,
            "verbose": 0,
            "n_jobs": 1,
        }
    elif data_size_mb > 100:  # 中量データ
        config = {
            "max_features": 50,
            "feateng_steps": 2,
            "max_gb": min(2.0, target_memory_gb),
            "featsel_runs": 1,
            "verbose": 0,
            "n_jobs": 1,
        }
    else:  # 小量データ
        config = {
            "max_features": 100,
            "feateng_steps": 2,
            "max_gb": min(3.0, target_memory_gb),
            "featsel_runs": 1,
            "verbose": 1,
            "n_jobs": 1,
        }

    # メモリ制約に基づく調整
    estimated_memory = estimate_autofeat_memory_usage(
        data_size_mb, config["max_features"], config["feateng_steps"]
    )

    if estimated_memory > target_memory_mb:
        # メモリ使用量が目標を超える場合は設定を調整
        reduction_factor = target_memory_mb / estimated_memory
        config["max_features"] = int(config["max_features"] * reduction_factor)
        config["feateng_steps"] = max(
            1, int(config["feateng_steps"] * reduction_factor)
        )
        config["max_gb"] = min(config["max_gb"], target_memory_gb * 0.8)

    return config


def cleanup_autofeat_memory(autofeat_model=None):
    """
    AutoFeat特有のメモリクリーンアップ

    Args:
        autofeat_model: AutoFeatモデルインスタンス
    """
    try:
        if autofeat_model is not None:
            # AutoFeatモデルの内部属性をクリア
            if hasattr(autofeat_model, "feateng_cols_"):
                autofeat_model.feateng_cols_ = None
            if hasattr(autofeat_model, "featsel_"):
                autofeat_model.featsel_ = None
            if hasattr(autofeat_model, "model_"):
                autofeat_model.model_ = None
            if hasattr(autofeat_model, "scaler_"):
                autofeat_model.scaler_ = None

        # 強制ガベージコレクション
        collected = gc.collect()
        logger.debug(f"AutoFeatメモリクリーンアップ: {collected}オブジェクト回収")

    except Exception as e:
        logger.error(f"AutoFeatメモリクリーンアップエラー: {e}")


def log_memory_usage(operation: str):
    """
    メモリ使用量をログ出力

    Args:
        operation: 操作名
    """
    try:
        memory_info = get_system_memory_info()
        logger.info(
            f"{operation} - メモリ使用状況: "
            f"プロセス {memory_info.get('process_mb', 0):.1f}MB, "
            f"システム使用率 {memory_info.get('used_percent', 0):.1f}%"
        )
    except Exception as e:
        logger.debug(f"メモリ使用量ログ出力エラー: {e}")
