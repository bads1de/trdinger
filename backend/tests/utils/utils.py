"""
MLテスト共通ユーティリティ

MLテストで使用する共通データ生成、設定、メトリクス計算機能を提供します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import psutil
import os


@dataclass
class MLTestConfig:
    """MLテスト設定"""
    sample_size: int = 1000
    prediction_horizon: int = 24
    threshold_up: float = 0.02
    threshold_down: float = -0.02
    test_train_split: float = 0.8
    random_seed: int = 42
    performance_timeout: float = 30.0
    memory_limit_mb: int = 1000


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    prediction_count: int
    throughput_per_second: float


def create_sample_ohlcv_data(size: int = 1000, start_price: float = 50000.0) -> pd.DataFrame:
    """
    サンプルOHLCVデータを生成
    
    Args:
        size: データサイズ
        start_price: 開始価格
        
    Returns:
        OHLCVデータフレーム
    """
    np.random.seed(42)
    
    # 時間インデックス生成
    start_time = datetime.now() - timedelta(hours=size)
    timestamps = [start_time + timedelta(hours=i) for i in range(size)]
    
    # 価格データ生成（ランダムウォーク）
    returns = np.random.normal(0, 0.02, size)
    prices = [start_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # 最低価格1.0
    
    # OHLCV生成
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # 高値・安値・始値を生成
        volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close
        
        # ボリューム生成
        volume = np.random.lognormal(10, 1)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def create_sample_funding_rate_data(size: int = 1000) -> pd.DataFrame:
    """
    サンプルファンディングレートデータを生成
    
    Args:
        size: データサイズ
        
    Returns:
        ファンディングレートデータフレーム
    """
    np.random.seed(42)
    
    start_time = datetime.now() - timedelta(hours=size * 8)  # 8時間間隔
    timestamps = [start_time + timedelta(hours=i * 8) for i in range(size)]
    
    # ファンディングレート生成（-0.1% ~ 0.1%）
    funding_rates = np.random.normal(0, 0.0005, size)
    funding_rates = np.clip(funding_rates, -0.001, 0.001)
    
    data = []
    for timestamp, rate in zip(timestamps, funding_rates):
        data.append({
            'timestamp': timestamp,
            'funding_rate': rate
        })
    
    return pd.DataFrame(data)


def create_sample_open_interest_data(size: int = 1000) -> pd.DataFrame:
    """
    サンプル建玉残高データを生成
    
    Args:
        size: データサイズ
        
    Returns:
        建玉残高データフレーム
    """
    np.random.seed(42)
    
    start_time = datetime.now() - timedelta(hours=size)
    timestamps = [start_time + timedelta(hours=i) for i in range(size)]
    
    # 建玉残高生成（トレンドあり）
    base_oi = 1000000
    trend = np.random.normal(0, 0.01, size).cumsum()
    open_interests = base_oi * (1 + trend * 0.1)
    open_interests = np.maximum(open_interests, base_oi * 0.5)  # 最低50%
    
    data = []
    for timestamp, oi in zip(timestamps, open_interests):
        data.append({
            'timestamp': timestamp,
            'open_interest': oi
        })
    
    return pd.DataFrame(data)


def measure_performance(func, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
    """
    関数のパフォーマンスを測定
    
    Args:
        func: 測定対象関数
        *args: 関数引数
        **kwargs: 関数キーワード引数
        
    Returns:
        (関数結果, パフォーマンスメトリクス)
    """
    # 初期メモリ使用量
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 実行時間測定
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    
    # 関数実行
    result = func(*args, **kwargs)
    
    # 終了時間・メモリ測定
    end_time = time.time()
    end_cpu = psutil.cpu_percent()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_usage = final_memory - initial_memory
    cpu_usage = (start_cpu + end_cpu) / 2
    
    # 予測数推定（結果がリストや配列の場合）
    prediction_count = 1
    if hasattr(result, '__len__'):
        prediction_count = len(result)
    elif isinstance(result, dict) and 'predictions' in result:
        prediction_count = len(result['predictions'])
    
    throughput = prediction_count / execution_time if execution_time > 0 else 0
    
    metrics = PerformanceMetrics(
        execution_time=execution_time,
        memory_usage_mb=memory_usage,
        cpu_usage_percent=cpu_usage,
        prediction_count=prediction_count,
        throughput_per_second=throughput
    )
    
    return result, metrics


def validate_ml_predictions(predictions: Dict[str, float]) -> bool:
    """
    ML予測結果の妥当性を検証
    
    Args:
        predictions: 予測結果辞書
        
    Returns:
        妥当性判定結果
    """
    required_keys = ['up', 'down', 'range']
    
    # 必要なキーの存在確認
    if not all(key in predictions for key in required_keys):
        return False
    
    # 確率値の範囲確認（0-1）
    for key in required_keys:
        if not (0 <= predictions[key] <= 1):
            return False
    
    # 確率の合計確認（1に近い）
    total_prob = sum(predictions[key] for key in required_keys)
    if abs(total_prob - 1.0) > 0.1:
        return False
    
    return True


def create_comprehensive_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    包括テスト用のデータセットを生成
    
    Returns:
        (OHLCV, ファンディングレート, 建玉残高)データフレーム
    """
    config = MLTestConfig()
    
    ohlcv_data = create_sample_ohlcv_data(config.sample_size)
    funding_rate_data = create_sample_funding_rate_data(config.sample_size // 8)
    open_interest_data = create_sample_open_interest_data(config.sample_size)
    
    return ohlcv_data, funding_rate_data, open_interest_data
