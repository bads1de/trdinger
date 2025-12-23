import time
import numpy as np
import pandas as pd
import pytest
from app.services.ml.label_generation.triple_barrier import TripleBarrier
from app.services.ml.label_generation.trend_scanning import TrendScanning


@pytest.fixture
def large_market_data():
    """大規模な市場データの生成"""
    np.random.seed(42)
    n = 10000  # 1万行
    index = pd.date_range("2020-01-01", periods=n, freq="1h")
    close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.01) * 100, index=index)

    # イベントは全期間
    t_events = index

    # ボラティリティ（ターゲット）
    target = pd.Series(0.01, index=index)

    return close, t_events, target


def test_triple_barrier_performance(large_market_data):
    """Triple Barrierのパフォーマンス計測"""
    close, t_events, target = large_market_data

    tb = TripleBarrier(pt=1.0, sl=1.0, min_ret=0.001)

    # JIT Warmup
    _ = tb.get_events(
        close=close.iloc[:100],
        t_events=t_events[:100],
        pt_sl=[1.0, 1.0],
        target=target[:100],
        min_ret=0.001,
    )

    start_time = time.time()
    events = tb.get_events(
        close=close, t_events=t_events, pt_sl=[1.0, 1.0], target=target, min_ret=0.001
    )
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nTriple Barrier (n={len(close)}): {duration:.4f} seconds")

    # 最適化後は < 0.1s を期待
    assert not events.empty


def test_trend_scanning_performance(large_market_data):
    """Trend Scanningのパフォーマンス計測"""
    close, t_events, _ = large_market_data

    ts = TrendScanning(min_window=5, max_window=20, step=1)

    # JIT Warmup
    _ = ts.get_labels(close.iloc[:100], t_events[:100])

    start_time = time.time()
    labels = ts.get_labels(close, t_events)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nTrend Scanning (n={len(close)}): {duration:.4f} seconds")

    # 最適化後は < 0.1s を期待 (1万行でも)
    assert not labels.empty
