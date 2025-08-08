import numpy as np
import pandas as pd
import pytest

from app.utils.data_processing import DataProcessor


@pytest.fixture
def df_with_outlier():
    # 数値1列のみを持つDataFrame（外れ値を含む）
    return pd.DataFrame({
        "x": [1.0, 2.0, 2.0, 3.0, 1000.0]
    })


def extract_numeric_output(transformed: np.ndarray):
    """ColumnTransformerの出力から数値部分を抽出する（今回は1列のみの前提で簡略化）"""
    arr = np.asarray(transformed)
    # 形状 (n_samples, n_features) を想定
    assert arr.ndim == 2
    return arr[:, 0]


def test_robust_transform_median_zero_iqr_one(df_with_outlier):
    dp = DataProcessor()
    # outlier_transform='robust' を指定してパイプラインを生成
    pipeline = dp.create_preprocessing_pipeline(
        numeric_strategy="median",
        categorical_strategy="most_frequent",
        scaling_method="standard",  # 無視される想定（robust指定時）
        remove_outliers=False,
        outlier_method="iqr",
        outlier_threshold=3.0,
        # 新パラメータ
        outlier_transform="robust",
    )

    transformed = pipeline.fit_transform(df_with_outlier)
    x_t = extract_numeric_output(transformed)

    # RobustScaler は中央値で中心化、IQRでスケーリング => 中央値は約0、IQRは約1
    median = float(np.median(x_t))
    q1, q3 = np.percentile(x_t, [25, 75])
    iqr = float(q3 - q1)

    assert abs(median) < 1e-6
    assert abs(iqr - 1.0) < 1e-6


def test_quantile_transform_bounds_extremes(df_with_outlier):
    dp = DataProcessor()
    pipeline = dp.create_preprocessing_pipeline(
        numeric_strategy="median",
        categorical_strategy="most_frequent",
        scaling_method="standard",
        remove_outliers=False,
        outlier_transform="quantile",
    )

    transformed = pipeline.fit_transform(df_with_outlier)
    x_t = extract_numeric_output(transformed)

    # QuantileTransformer(output_distribution='normal') なら極端値でも有限で、おおむね±4以内に収まるはず
    assert np.all(np.isfinite(x_t))
    assert np.max(np.abs(x_t)) < 6.0  # 小サンプルのランク変換を考慮し少しマージン


def test_power_transform_is_finite(df_with_outlier):
    dp = DataProcessor()
    pipeline = dp.create_preprocessing_pipeline(
        numeric_strategy="median",
        categorical_strategy="most_frequent",
        scaling_method="standard",
        remove_outliers=False,
        outlier_transform="power",
    )

    transformed = pipeline.fit_transform(df_with_outlier)
    x_t = extract_numeric_output(transformed)

    # 変換後にNaNやinfが無いこと
    assert np.all(np.isfinite(x_t))

