import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


def test_pandas_fillna_method_bfill_deprecated():
    """pandas fillna method='bfill' がdeprecatedであることを確認するテスト"""
    # 非推奨方法を使用
    series = pd.Series([1, np.nan, 3, np.nan, 5])

    # 非推奨のmethodパラメータを使用した場合、エラーが発生せずに動作するはず
    with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
        result = series.fillna(method="bfill")


def test_pandas_fillna_method_ffill_deprecated():
    """pandas fillna method='ffill' がdeprecatedであることを確認するテスト"""
    # 非推奨方法を使用
    series = pd.Series([1, np.nan, 3, np.nan, 5])

    # 非推奨のmethodパラメータを使用した場合、エラーが発生せずに動作するはず
    with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
        result = series.fillna(method="ffill")


def test_pandas_fillna_new_api_bfill():
    """新しいAPI bfill() が動作することを確認するテスト"""
    series = pd.Series([1, np.nan, 3, np.nan, 5])
    expected = pd.Series([1.0, 3.0, 3.0, 5.0, 5.0], index=series.index)

    result = series.bfill()

    pd.testing.assert_series_equal(result, expected)


def test_pandas_fillna_new_api_ffill():
    """新しいAPI ffill() が動作することを確認するテスト"""
    series = pd.Series([1, np.nan, 3, np.nan, 5])
    expected = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0], index=series.index)

    result = series.ffill()

    pd.testing.assert_series_equal(result, expected)


def test_dataframe_fillna_method_bfill_deprecated():
    """DataFrame のfillna method='bfill' がdeprecatedであることを確認するテスト"""
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})

    # 非推奨のmethodパラメータを使用した場合、エラーが発生せずに動作するはず
    with pytest.warns(FutureWarning, match="fillna with 'method' is deprecated"):
        result = df.fillna(method="bfill")


def test_dataframe_fillna_new_api_bfill():
    """DataFrame の新しいAPI bfill() が動作することを確認するテスト"""
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})
    expected = pd.DataFrame({"A": [1.0, 3.0, 3.0], "B": [4.0, 6.0, 6.0]})

    result = df.bfill()

    pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])