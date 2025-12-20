import pytest
import numpy as np
import pandas as pd
from app.services.indicators.data_validation import (
    validate_input,
    handle_pandas_ta_errors,
    validate_series_params,
    validate_multi_series_params,
    PandasTAError,
)


class TestIndicatorUtils:
    def test_validate_input_none(self):
        with pytest.raises(PandasTAError, match="入力データがNoneです"):
            validate_input(None, 10)

    def test_validate_input_not_series(self):
        with pytest.raises(
            PandasTAError, match="入力データはpandas.Seriesである必要があります"
        ):
            validate_input([1, 2, 3], 10)

    def test_validate_input_empty(self):
        with pytest.raises(PandasTAError, match="入力データが空です"):
            validate_input(pd.Series([], dtype=float), 10)

    def test_validate_input_invalid_period(self):
        with pytest.raises(PandasTAError, match="期間は正の整数である必要があります"):
            validate_input(pd.Series([1, 2, 3]), 0)

    def test_validate_input_length_short(self):
        with pytest.raises(
            PandasTAError, match="データ長\(3\)が期間\(10\)より短いです"
        ):
            validate_input(pd.Series([1, 2, 3]), 10)

    def test_validate_input_inf(self):
        with pytest.raises(
            PandasTAError, match="入力データに無限大の値が含まれています"
        ):
            validate_input(pd.Series([1, np.inf, 3]), 2)

    def test_validate_input_nan_warning(self, caplog):
        validate_input(pd.Series([1, np.nan, 3]), 2)
        assert "入力データにNaN値が含まれています" in caplog.text

    def test_handle_pandas_ta_errors_none_result(self):
        @handle_pandas_ta_errors
        def mock_func():
            return None

        with pytest.raises(PandasTAError, match="mock_func: 計算結果がNoneです"):
            mock_func()

    def test_handle_pandas_ta_errors_empty_array(self):
        @handle_pandas_ta_errors
        def mock_func():
            return np.array([])

        with pytest.raises(PandasTAError, match="mock_func: 計算結果が空です"):
            mock_func()

    def test_handle_pandas_ta_errors_all_nan(self):
        @handle_pandas_ta_errors
        def mock_func():
            return np.array([np.nan, np.nan])

        with pytest.raises(PandasTAError, match="mock_func: 計算結果が全てNaNです"):
            mock_func()

    def test_handle_pandas_ta_errors_invalid_tuple(self):
        @handle_pandas_ta_errors
        def mock_func():
            return (np.array([1, 2]), None)

        with pytest.raises(PandasTAError, match="mock_func: 結果\[1\]が無効です"):
            mock_func()

    def test_handle_pandas_ta_errors_general_exception(self):
        @handle_pandas_ta_errors
        def mock_func():
            raise Exception("Some error")

        with pytest.raises(PandasTAError, match="mock_func 計算エラー: Some error"):
            mock_func()

    def test_validate_series_params_not_series(self):
        with pytest.raises(TypeError, match="data must be pandas Series"):
            validate_series_params([1, 2, 3])

    def test_validate_series_params_invalid_length(self):
        with pytest.raises(ValueError, match="length must be positive"):
            validate_series_params(pd.Series([1, 2, 3]), length=0)

    def test_validate_series_params_empty(self):
        res = validate_series_params(pd.Series([], dtype=float))
        assert isinstance(res, pd.Series)
        assert len(res) == 0

    def test_validate_series_params_short(self):
        res = validate_series_params(pd.Series([1, 2, 3]), min_data_length=10)
        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert np.isnan(res).all()

    def test_validate_series_params_ok(self):
        assert validate_series_params(pd.Series([1, 2, 3]), min_data_length=2) is None

    def test_validate_multi_series_params_empty_dict(self):
        with pytest.raises(ValueError, match="series_dict cannot be empty"):
            validate_multi_series_params({})

    def test_validate_multi_series_params_not_series(self):
        with pytest.raises(TypeError, match="high must be pandas Series"):
            validate_multi_series_params({"high": [1, 2, 3]})

    def test_validate_multi_series_params_mismatch_length(self):
        with pytest.raises(ValueError, match="All series must have the same length"):
            validate_multi_series_params(
                {"high": pd.Series([1, 2, 3]), "low": pd.Series([1, 2])}
            )

    def test_validate_multi_series_params_invalid_length(self):
        with pytest.raises(ValueError, match="length must be positive"):
            validate_multi_series_params({"high": pd.Series([1, 2, 3])}, length=0)

    def test_validate_multi_series_params_empty(self):
        res = validate_multi_series_params({"high": pd.Series([], dtype=float)})
        assert len(res) == 0

    def test_validate_multi_series_params_short(self):
        res = validate_multi_series_params(
            {"high": pd.Series([1, 2, 3])}, min_data_length=10
        )
        assert len(res) == 3
        assert np.isnan(res).all()

    def test_validate_multi_series_params_ok(self):
        assert (
            validate_multi_series_params(
                {"high": pd.Series([1, 2, 3])}, min_data_length=2
            )
            is None
        )
