import numpy as np
import pandas as pd
import pytest

from app.utils.data_processing import DataProcessor


def test_iqr_outlier_removal_vectorized_correctness():
    dp = DataProcessor()
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 1000],  # 1000 should be an outlier for IQR with 1.5
            "B": [10, 11, 12, 11, 10],  # no outliers
        }
    )

    result = dp._remove_outliers(df, columns=["A", "B"], threshold=1.5, method="iqr")

    # A's last value should be NaN
    assert np.isnan(result.loc[4, "A"]), "IQR外れ値がNaNに置き換わっていません"
    # B should remain intact
    assert pd.isna(result["B"]).sum() == 0, "外れ値のないカラムにNaNが発生しています"


def test_zscore_outlier_removal_vectorized_correctness():
    dp = DataProcessor()
    # one strong outlier at 10
    df = pd.DataFrame({"A": [0.0, 0.1, -0.2, 0.05, 100.0]})

    result = dp._remove_outliers(df, columns=["A"], threshold=1.5, method="zscore")

    assert np.isnan(result.loc[4, "A"]), "Z-score外れ値がNaNに置き換わっていません"


def test_zscore_zero_std_safe():
    dp = DataProcessor()
    # zero std column should not divide by zero
    df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]})

    result = dp._remove_outliers(df, columns=["A"], threshold=3.0, method="zscore")

    # no NaNs introduced
    assert pd.isna(result["A"]).sum() == 0, "分散0でもNaNが発生しています"


def test_non_numeric_columns_ignored():
    dp = DataProcessor()
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 1000],
            "C": ["x", "y", "z", "w", "v"],  # non-numeric
        }
    )

    result = dp._remove_outliers(df, columns=["A", "C"], threshold=1.5, method="iqr")

    # non-numeric column should stay exactly the same
    assert result["C"].equals(df["C"]), "非数値カラムが変更されています"


def test_pipeline_integration_outlier_then_impute():
    dp = DataProcessor()
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 1000.0]})

    # Use internal remover to ensure NaN is produced
    removed = dp._remove_outliers(df, columns=["A"], threshold=1.5, method="iqr")
    assert np.isnan(removed.loc[4, "A"])

    # Now run transform_missing_values to ensure NaN will be imputed (median)
    imputed = dp.transform_missing_values(removed, strategy="median", columns=["A"])
    assert not pd.isna(imputed["A"]).any(), "欠損値補完が行われていません"
