#!/usr/bin/env python3
"""
NaN値対応の手動テスト
"""

import os
import sys

import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.data_processing.pipelines.preprocessing_pipeline import (
    CategoricalPipelineTransformer,
    create_preprocessing_pipeline,
)


def test_nan_handling():
    """NaN値処理の手動テスト"""
    print("\n=== NaN値処理の手動テスト ===")

    # NaNを含むデータを作成
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")

    # 数値列にNaNを含むデータ
    data_with_nan = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "feature2": [
                np.nan,
                20.0,
                30.0,
                np.nan,
                50.0,
                60.0,
                np.nan,
                80.0,
                90.0,
                np.nan,
            ],
            "feature3": [
                100.0,
                np.nan,
                np.nan,
                400.0,
                500.0,
                np.nan,
                700.0,
                800.0,
                np.nan,
                1000.0,
            ],
            "category": ["A", "B", None, "A", "B", None, "A", "B", None, "A"],
        }
    )

    print("NaNを含む入力データ:")
    print(data_with_nan)
    print("\nNaNの分布:")
    print(data_with_nan.isnull().sum())

    # CategoricalPipelineTransformerのテスト
    print("\n=== CategoricalPipelineTransformerのNaN処理 ===")

    # 全てNaNの列を含むデータ
    all_nan_data = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "some_nan": [1.0, np.nan, 3.0, np.nan],
            "no_nan": [10, 20, 30, 40],
        }
    )

    print("全NaN列を含むデータ:")
    print(all_nan_data)

    try:
        cat_transformer = CategoricalPipelineTransformer(strategy="mean")
        cat_transformer.fit(all_nan_data)

        result = cat_transformer.transform(all_nan_data)
        print("✅ 変換成功:")
        print(result)
        print("列の型:")
        print(result.dtypes)

    except Exception as e:
        print(f"NG 変換失敗: {e}")


def test_preprocessing_pipeline():
    """前処理パイプラインのテスト"""
    print("\n=== 前処理パイプラインのテスト ===")

    # 全てNaNの列を含むデータ
    all_nan_data = pd.DataFrame(
        {
            "all_nan_num": [np.nan, np.nan, np.nan, np.nan],
            "some_nan_num": [1.0, np.nan, 3.0, np.nan],
            "no_nan_num": [10, 20, 30, 40],
            "all_nan_cat": [None, None, None, None],
            "some_nan_cat": ["A", None, "B", None],
            "no_nan_cat": ["X", "Y", "Z", "W"],
        }
    )

    print("入力データ:")
    print(all_nan_data)

    try:
        # 前処理パイプラインを作成
        pipeline = create_preprocessing_pipeline(
            outlier_method=None,  # 外れ値除去を無効化
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
        )

        print("OK パイプライン作成成功")

        # パイプラインを適用
        result = pipeline.fit_transform(all_nan_data)
        print("OK パイプライン適用成功")
        print("結果:")
        print(result)

    except Exception as e:
        print(f"NG パイプライン処理失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("NaN値対応の手動テストを開始します...")
    test_nan_handling()
    test_preprocessing_pipeline()
    print("\nNaN値テスト完了！")
