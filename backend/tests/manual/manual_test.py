#!/usr/bin/env python3
"""
手動テストスクリプト
自動テストが通らない場合のデバッグ用
"""

import os
import sys

import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ml.feature_engineering.price_features import PriceFeatureCalculator
from app.utils.data_processing.validators.data_validator import validate_data_integrity


def test_timestamp_validation():
    """タイムスタンプ検証の手動テスト"""
    print("=== タイムスタンプ検証の手動テスト ===")

    # タイムスタンプなしデータを作成
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")

    data_without_timestamp = pd.DataFrame(
        {
            "open": 10000 + np.random.randn(len(dates)) * 100,
            "high": 10100 + np.random.randn(len(dates)) * 150,
            "low": 9900 + np.random.randn(len(dates)) * 150,
            "close": 10000 + np.random.randn(len(dates)) * 100,
            "volume": 1000 + np.random.randint(100, 1000, len(dates)),
        }
    )

    print("タイムスタンプなしデータ:")
    print(data_without_timestamp.head())

    try:
        result = validate_data_integrity(data_without_timestamp)
        print(f"OK 検証成功: {result}")
    except Exception as e:
        print(f"NG 検証失敗: {e}")


def test_price_feature_columns():
    """価格特徴量カラム名の手動テスト"""
    print("\n=== 価格特徴量カラム名の手動テスト ===")

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")

    # 小文字カラムのデータ
    data_lower = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 10000 + np.random.randn(len(dates)) * 100,
            "high": 10100 + np.random.randn(len(dates)) * 150,
            "low": 9900 + np.random.randn(len(dates)) * 150,
            "close": 10000 + np.random.randn(len(dates)) * 100,
            "volume": 1000 + np.random.randint(100, 1000, len(dates)),
        }
    )

    # 大文字カラムのデータ
    data_upper = pd.DataFrame(
        {
            "timestamp": dates,
            "Open": 10000 + np.random.randn(len(dates)) * 100,
            "High": 10100 + np.random.randn(len(dates)) * 150,
            "Low": 9900 + np.random.randn(len(dates)) * 150,
            "Close": 10000 + np.random.randn(len(dates)) * 100,
            "Volume": 1000 + np.random.randint(100, 1000, len(dates)),
        }
    )

    print("小文字カラムデータ:")
    print(data_lower.head())
    print("\n大文字カラムデータ:")
    print(data_upper.head())

    calculator = PriceFeatureCalculator()

    # 小文字カラムでテスト
    try:
        result_lower = calculator.validate_input_data(
            data_lower, ["close", "open", "high", "low"]
        )
        print(f"OK 小文字カラム検証: {result_lower}")
    except Exception as e:
        print(f"NG 小文字カラム検証失敗: {e}")

    # 大文字カラムでテスト
    try:
        result_upper = calculator.validate_input_data(
            data_upper, ["close", "open", "high", "low"]
        )
        print(f"OK 大文字カラム検証: {result_upper}")
    except Exception as e:
        print(f"NG 大文字カラム検証失敗: {e}")


def test_base_feature_calculator():
    """BaseFeatureCalculatorの手動テスト"""
    print("\n=== BaseFeatureCalculatorの手動テスト ===")

    # validate_input_dataメソッドのみをテスト
    # BaseFeatureCalculatorは抽象クラスなので、サブクラスを使用
    from app.services.ml.feature_engineering.price_features import (
        PriceFeatureCalculator,
    )

    calculator = PriceFeatureCalculator()

    # 空データテスト
    empty_df = pd.DataFrame()
    result_empty = calculator.validate_input_data(empty_df)
    print(f"OK 空データ検証: {result_empty}")

    # Noneデータテスト
    result_none = calculator.validate_input_data(None)
    print(f"OK Noneデータ検証: {result_none}")


if __name__ == "__main__":
    print("手動テストを開始します...")
    test_timestamp_validation()
    test_price_feature_columns()
    test_base_feature_calculator()
    print("手動テスト完了！")


