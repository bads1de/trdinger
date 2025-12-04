import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from app.services.ml.label_generation.trend_scanning import TrendScanning
from scripts.feature_evaluation.common_feature_evaluator import CommonFeatureEvaluator


def check_label_distribution():
    # データ取得
    evaluator = CommonFeatureEvaluator()
    data = evaluator.fetch_data(symbol="BTC/USDT:USDT", timeframe="1h", limit=5000)
    close = data.ohlcv["close"]

    # Trend Scanning パラメータ (最適化結果に近い値)
    min_window = 8
    max_window = 40
    threshold = 10.0

    ts = TrendScanning(
        min_window=min_window, max_window=max_window, min_t_value=threshold
    )

    labels_df = ts.get_labels(close)

    # 方向性ラベル (-1, 0, 1)
    print("=== Directional Labels (-1, 0, 1) ===")
    print(labels_df["bin"].value_counts(normalize=True))
    print(labels_df["bin"].value_counts())

    # バイナリラベル (0, 1) - 今回のパイプラインで使用
    binary_labels = labels_df["bin"].abs()
    print("\n=== Binary Labels (0, 1) ===")
    print(binary_labels.value_counts(normalize=True))
    print(binary_labels.value_counts())

    # t値の統計
    print("\n=== t-value Statistics ===")
    print(labels_df["t_value"].describe())


if __name__ == "__main__":
    check_label_distribution()
