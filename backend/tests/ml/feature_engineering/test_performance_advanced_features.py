import time
import numpy as np
import pandas as pd


def generate_test_data(n=5000):
    """
    テスト用のランダムデータを生成
    """
    np.random.seed(42)
    close = pd.Series(
        100 + np.cumsum(np.random.randn(n)),
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    return close


def test_benchmark_advanced_features():
    """
    高度な特徴量のベンチマークテスト
    """
    n = 5000

    print(f"\n--- Performance Benchmark (N={n}) ---")

    # 1. Fractional Differentiation (FFD)
    start = time.time()

    duration = time.time() - start
    print(f"Fractional Differentiation (FFD): {duration:.4f}s")

    # 2. Hurst Exponent
    start = time.time()

    duration = time.time() - start
    print(f"Hurst Exponent: {duration:.4f}s")

    # 3. Sample Entropy (これはNumbaなしだとN=5000は数分かかる可能性がある)
    start = time.time()

    duration = time.time() - start
    print(f"Sample Entropy (N=1000, win=50): {duration:.4f}s")

    # 4. Katz Fractal Dimension
    start = time.time()

    duration = time.time() - start
    print(f"Katz Fractal Dimension: {duration:.4f}s")


if __name__ == "__main__":
    test_benchmark_advanced_features()
