import time
import numpy as np
import pandas as pd
from app.services.indicators.technical_indicators.advanced_features import (
    AdvancedFeatures,
)


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
    close = generate_test_data(n)

    print(f"\n--- Performance Benchmark (N={n}) ---")

    # 1. Fractional Differentiation (FFD)
    start = time.time()
    ffd_result = AdvancedFeatures.frac_diff_ffd(close, d=0.4, window=500)
    duration = time.time() - start
    print(f"Fractional Differentiation (FFD): {duration:.4f}s")

    # 2. Hurst Exponent
    start = time.time()
    hurst_result = AdvancedFeatures.hurst_exponent(close, window=100)
    duration = time.time() - start
    print(f"Hurst Exponent: {duration:.4f}s")

    # 3. Sample Entropy (これはNumbaなしだとN=5000は数分かかる可能性がある)
    start = time.time()
    entropy_result = AdvancedFeatures.sample_entropy(
        close[:1000], window=50
    )  # ここは1000件でテスト
    duration = time.time() - start
    print(f"Sample Entropy (N=1000, win=50): {duration:.4f}s")

    # 4. Katz Fractal Dimension
    start = time.time()
    katz_result = AdvancedFeatures.fractal_dimension(close, window=30)
    duration = time.time() - start
    print(f"Katz Fractal Dimension: {duration:.4f}s")


if __name__ == "__main__":
    test_benchmark_advanced_features()
