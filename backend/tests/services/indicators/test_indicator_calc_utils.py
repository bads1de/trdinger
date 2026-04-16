from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.data_validation import (
    PandasTAError,
    create_nan_result,
    create_nan_series_bundle,
    create_nan_series_like,
    create_nan_series_map,
    handle_pandas_ta_errors,
    run_multi_series_indicator,
    run_series_indicator,
    validate_data_length_with_fallback,
    validate_input,
    validate_multi_series_params,
    validate_series_params,
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
            PandasTAError, match=r"データ長\(3\)が期間\(10\)より短いです"
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

        with pytest.raises(PandasTAError, match=r"mock_func: 結果\[1\]が無効です"):
            mock_func()

    def test_handle_pandas_ta_errors_empty_tuple_allowed(self):
        @handle_pandas_ta_errors
        def mock_func():
            return (pd.Series([], dtype=float), pd.Series([], dtype=float))

        result = mock_func()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == 0

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

    def test_create_nan_series_like(self):
        series = pd.Series([1, 2, 3], name="close")

        result = create_nan_series_like(series, fill_value=0.0, name="fallback")

        assert isinstance(result, pd.Series)
        assert result.name == "fallback"
        assert result.index.equals(series.index)
        assert (result == 0.0).all()

    def test_create_nan_series_bundle(self):
        series = pd.Series([1, 2, 3], name="close")

        result = create_nan_series_bundle(series, 3)

        assert isinstance(result, tuple)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, pd.Series)
            assert item.index.equals(series.index)
            assert np.isnan(item).all()

    def test_create_nan_series_map(self):
        series = pd.Series([1, 2, 3], name="close")

        result = create_nan_series_map(series, ["a", "b"])

        assert set(result.keys()) == {"a", "b"}
        assert all(isinstance(item, pd.Series) for item in result.values())
        assert result["a"].name == "a"
        assert result["b"].name == "b"
        assert all(np.isnan(item).all() for item in result.values())

    @patch("app.services.indicators.data_validation._get_indicator_config")
    def test_create_nan_result_single(self, mock_get_config):
        mock_get_config.return_value = None

        result = create_nan_result(pd.DataFrame({"close": [1, 2, 3]}), "UNKNOWN")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.isnan(result).all()

    @patch("app.services.indicators.data_validation._get_indicator_config")
    def test_create_nan_result_multiple(self, mock_get_config):
        mock_config = type(
            "MockConfig",
            (),
            {"returns": "multiple", "return_cols": ["a", "b"]},
        )()
        mock_get_config.return_value = mock_config

        result = create_nan_result(pd.DataFrame({"close": [1, 2, 3]}), "TEST")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        assert np.isnan(result).all()

    def test_run_series_indicator_validation_fallback(self):
        series = pd.Series([1, 2, 3], name="close")

        result = run_series_indicator(
            series,
            None,
            lambda: pd.Series([9, 9, 9], index=series.index),
            min_data_length=5,
        )

        assert isinstance(result, pd.Series)
        assert result.index.equals(series.index)
        assert np.isnan(result).all()

    def test_run_series_indicator_result_fallback(self):
        series = pd.Series([1, 2, 3], name="close")

        result = run_series_indicator(series, None, lambda: None)

        assert isinstance(result, pd.Series)
        assert result.index.equals(series.index)
        assert np.isnan(result).all()

    def test_run_multi_series_indicator_bundle_fallback(self):
        series = pd.Series([1, 2, 3], name="close")

        result = run_multi_series_indicator(
            {"high": series, "low": series},
            None,
            lambda: None,
            fallback_factory=lambda: create_nan_series_bundle(series, 2),
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(item, pd.Series) for item in result)
        assert all(np.isnan(item).all() for item in result)

    def test_validate_data_length_with_fallback(self, sample_df):
        result = validate_data_length_with_fallback(sample_df, "SMA", {"length": 10})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    def test_validate_data_length_with_fallback_insufficient(self):
        data = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})

        is_valid, min_length = validate_data_length_with_fallback(
            data, "SMA", {"length": 14}
        )

        assert isinstance(is_valid, bool)
        assert isinstance(min_length, int)
