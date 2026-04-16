"""
data_validation モジュールのユニットテスト

検証ユーティリティ、NaN 生成ヘルパー、エラーハンドリングデコレーターを
直接テストする。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.data_validation import (
    PandasTAError,
    _create_nan_array,
    _is_missing_indicator_result,
    _return_nan_series_if_needed,
    _run_indicator_with_validation,
    _validate_positive_length,
    _validate_series_collection,
    create_nan_result,
    create_nan_series_bundle,
    create_nan_series_like,
    create_nan_series_map,
    get_param_value,
    handle_pandas_ta_errors,
    normalize_non_finite,
    run_multi_series_indicator,
    run_series_indicator,
    validate_input,
    validate_multi_series_params,
    validate_series_params,
)


@pytest.fixture
def sample_series() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.date_range("2024-01-01", periods=5, freq="h"),
        name="close",
    )


@pytest.fixture
def nan_series() -> pd.Series:
    return pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )


# ---------------------------------------------------------------------------
# PandasTAError
# ---------------------------------------------------------------------------


class TestPandasTAError:
    def test_is_exception(self):
        assert issubclass(PandasTAError, Exception)

    def test_raise_and_message(self):
        with pytest.raises(PandasTAError, match="test error"):
            raise PandasTAError("test error")


# ---------------------------------------------------------------------------
# _validate_positive_length
# ---------------------------------------------------------------------------


class TestValidatePositiveLength:
    def test_none_passes(self):
        _validate_positive_length(None)  # should not raise

    def test_positive_passes(self):
        _validate_positive_length(10)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_length(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_positive_length(-1)


# ---------------------------------------------------------------------------
# _return_nan_series_if_needed
# ---------------------------------------------------------------------------


class TestReturnNanSeriesIfNeeded:
    def test_empty_series_returns_nan(self, sample_series):
        empty = sample_series.iloc[:0]
        result = _return_nan_series_if_needed(empty, min_data_length=0)
        assert result is not None
        assert len(result) == 0

    def test_too_short_returns_nan(self, sample_series):
        result = _return_nan_series_if_needed(sample_series, min_data_length=100)
        assert result is not None
        assert result.isna().all()

    def test_sufficient_length_returns_none(self, sample_series):
        result = _return_nan_series_if_needed(sample_series, min_data_length=3)
        assert result is None


# ---------------------------------------------------------------------------
# _validate_series_collection
# ---------------------------------------------------------------------------


class TestValidateSeriesCollection:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_series_collection({})

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="must be pandas Series"):
            _validate_series_collection({"x": [1, 2, 3]})

    def test_mismatched_lengths_raises(self, sample_series):
        other = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            _validate_series_collection({"a": sample_series, "b": other})

    def test_matching_series_passes(self, sample_series):
        result = _validate_series_collection({"a": sample_series, "b": sample_series})
        assert result is None

    def test_short_data_returns_nan(self, sample_series):
        result = _validate_series_collection({"a": sample_series}, min_data_length=100)
        assert result is not None
        assert result.isna().all()

    def test_negative_length_raises(self, sample_series):
        with pytest.raises(ValueError, match="positive"):
            _validate_series_collection({"a": sample_series}, length=-1)


# ---------------------------------------------------------------------------
# _create_nan_array
# ---------------------------------------------------------------------------


class TestCreateNanArray:
    def test_1d(self):
        arr = _create_nan_array(5)
        assert arr.shape == (5,)
        assert np.isnan(arr).all()

    def test_2d(self):
        arr = _create_nan_array(3, width=4)
        assert arr.shape == (3, 4)
        assert np.isnan(arr).all()


# ---------------------------------------------------------------------------
# _is_missing_indicator_result
# ---------------------------------------------------------------------------


class TestIsMissingIndicatorResult:
    def test_none_is_missing(self):
        assert _is_missing_indicator_result(None) is True

    def test_all_nan_series_is_missing(self, nan_series):
        assert _is_missing_indicator_result(nan_series) is True

    def test_valid_series_is_not_missing(self, sample_series):
        assert _is_missing_indicator_result(sample_series) is False

    def test_empty_series_is_missing(self):
        assert _is_missing_indicator_result(pd.Series(dtype=float)) is True

    def test_all_nan_dataframe_is_missing(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        assert _is_missing_indicator_result(df) is True

    def test_valid_dataframe_is_not_missing(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        assert _is_missing_indicator_result(df) is False

    def test_empty_ndarray_is_missing(self):
        assert _is_missing_indicator_result(np.array([])) is True

    def test_all_nan_ndarray_is_missing(self):
        assert _is_missing_indicator_result(np.array([np.nan, np.nan])) is True

    def test_valid_ndarray_is_not_missing(self):
        assert _is_missing_indicator_result(np.array([1.0, 2.0])) is False

    def test_empty_tuple_is_missing(self):
        assert _is_missing_indicator_result(()) is True

    def test_all_none_tuple_is_missing(self):
        assert _is_missing_indicator_result((None, None)) is True

    def test_valid_tuple_is_not_missing(self):
        assert _is_missing_indicator_result((1, 2)) is False

    def test_scalar_is_not_missing(self):
        assert _is_missing_indicator_result(42) is False


# ---------------------------------------------------------------------------
# create_nan_series_like
# ---------------------------------------------------------------------------


class TestCreateNanSeriesLike:
    def test_same_length_and_index(self, sample_series):
        result = create_nan_series_like(sample_series)
        assert len(result) == len(sample_series)
        assert result.index.equals(sample_series.index)
        assert result.isna().all()

    def test_custom_name(self, sample_series):
        result = create_nan_series_like(sample_series, name="custom")
        assert result.name == "custom"

    def test_preserves_original_name(self, sample_series):
        result = create_nan_series_like(sample_series)
        assert result.name == sample_series.name

    def test_custom_fill_value(self, sample_series):
        result = create_nan_series_like(sample_series, fill_value=0.0)
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# create_nan_series_bundle / create_nan_series_map
# ---------------------------------------------------------------------------


class TestCreateNanSeriesBundle:
    def test_count(self, sample_series):
        bundle = create_nan_series_bundle(sample_series, 3)
        assert len(bundle) == 3
        for s in bundle:
            assert len(s) == len(sample_series)
            assert s.isna().all()

    def test_independent_copies(self, sample_series):
        bundle = create_nan_series_bundle(sample_series, 2)
        bundle[0].iloc[0] = 999
        assert bundle[1].iloc[0] != 999


class TestCreateNanSeriesMap:
    def test_keys(self, sample_series):
        result = create_nan_series_map(sample_series, ["a", "b", "c"])
        assert set(result.keys()) == {"a", "b", "c"}
        for s in result.values():
            assert s.isna().all()
            assert len(s) == len(sample_series)


# ---------------------------------------------------------------------------
# normalize_non_finite
# ---------------------------------------------------------------------------


class TestNormalizeNonFinite:
    def test_inf_replaced(self):
        s = pd.Series([1.0, np.inf, -np.inf, np.nan, 2.0])
        result = normalize_non_finite(s)
        assert not np.isinf(result).any()

    def test_custom_fill(self):
        s = pd.Series([np.inf, -np.inf])
        result = normalize_non_finite(s, fill_value=0.0)
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# run_series_indicator / run_multi_series_indicator
# ---------------------------------------------------------------------------


class TestRunSeriesIndicator:
    def test_calls_factory_when_valid(self, sample_series):
        called = False

        def factory():
            nonlocal called
            called = True
            return sample_series * 2

        result = run_series_indicator(sample_series, 3, factory)
        assert called is True
        assert isinstance(result, pd.Series)

    def test_returns_nan_when_data_too_short(self, sample_series):
        result = run_series_indicator(
            sample_series, None, lambda: sample_series * 2, min_data_length=100
        )
        assert result.isna().all()

    def test_fallback_factory_used_when_result_missing(self, sample_series):
        def bad_factory():
            return None

        def fallback():
            return pd.Series([42.0] * len(sample_series), index=sample_series.index)

        result = run_series_indicator(
            sample_series, 3, bad_factory, fallback_factory=fallback
        )
        assert (result == 42.0).all()


class TestRunMultiSeriesIndicator:
    def test_calls_factory_when_valid(self, sample_series):
        called = False

        def factory():
            nonlocal called
            called = True
            return sample_series

        result = run_multi_series_indicator(
            {"a": sample_series, "b": sample_series}, 3, factory
        )
        assert called is True

    def test_returns_nan_when_data_too_short(self, sample_series):
        result = run_multi_series_indicator(
            {"a": sample_series},
            None,
            lambda: sample_series,
            min_data_length=100,
        )
        assert result.isna().all()


# ---------------------------------------------------------------------------
# validate_series_params / validate_multi_series_params
# ---------------------------------------------------------------------------


class TestValidateSeriesParams:
    def test_valid_returns_none(self, sample_series):
        assert validate_series_params(sample_series, length=3) is None

    def test_short_returns_nan(self, sample_series):
        result = validate_series_params(sample_series, min_data_length=100)
        assert result is not None
        assert result.isna().all()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            validate_series_params([1, 2, 3], length=3)


class TestValidateMultiSeriesParams:
    def test_valid_returns_none(self, sample_series):
        assert validate_multi_series_params({"a": sample_series}, length=3) is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            validate_multi_series_params(
                {"a": pd.Series([1, 2]), "b": pd.Series([1, 2, 3])}
            )


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_valid_data(self, sample_series):
        validate_input(sample_series, period=3)  # no raise

    def test_none_raises(self):
        with pytest.raises(PandasTAError, match="None"):
            validate_input(None, period=3)

    def test_not_series_raises(self):
        with pytest.raises(PandasTAError, match="pandas.Series"):
            validate_input([1, 2, 3], period=3)

    def test_empty_raises(self):
        with pytest.raises(PandasTAError, match="空"):
            validate_input(pd.Series(dtype=float), period=3)

    def test_negative_period_raises(self, sample_series):
        with pytest.raises(PandasTAError, match="正の整数"):
            validate_input(sample_series, period=-1)

    def test_period_larger_than_data_raises(self, sample_series):
        with pytest.raises(PandasTAError, match="短い"):
            validate_input(sample_series, period=100)

    def test_inf_data_raises(self):
        s = pd.Series([1.0, np.inf, 3.0])
        with pytest.raises(PandasTAError, match="無限大"):
            validate_input(s, period=1)


# ---------------------------------------------------------------------------
# handle_pandas_ta_errors
# ---------------------------------------------------------------------------


class TestHandlePandasTaErrors:
    def test_passes_valid_result(self, sample_series):
        @handle_pandas_ta_errors
        def good_func():
            return sample_series

        assert good_func() is sample_series

    def test_raises_on_none_result(self):
        @handle_pandas_ta_errors
        def bad_func():
            return None

        with pytest.raises(PandasTAError, match="None"):
            bad_func()

    def test_raises_on_empty_ndarray(self):
        @handle_pandas_ta_errors
        def bad_func():
            return np.array([])

        with pytest.raises(PandasTAError, match="空"):
            bad_func()

    def test_raises_on_all_nan_ndarray(self):
        @handle_pandas_ta_errors
        def bad_func():
            return np.array([np.nan, np.nan])

        with pytest.raises(PandasTAError, match="全てNaN"):
            bad_func()

    def test_raises_on_invalid_tuple(self):
        @handle_pandas_ta_errors
        def bad_func():
            return (None, None)

        with pytest.raises(PandasTAError, match="無効"):
            bad_func()

    def test_wraps_exception_as_pandas_ta_error(self):
        @handle_pandas_ta_errors
        def bad_func():
            raise RuntimeError("inner error")

        with pytest.raises(PandasTAError, match="inner error"):
            bad_func()

    def test_preserves_value_error(self):
        @handle_pandas_ta_errors
        def bad_func():
            raise ValueError("val error")

        with pytest.raises(ValueError, match="val error"):
            bad_func()

    def test_preserves_pandas_ta_error(self):
        @handle_pandas_ta_errors
        def bad_func():
            raise PandasTAError("pta error")

        with pytest.raises(PandasTAError, match="pta error"):
            bad_func()


# ---------------------------------------------------------------------------
# get_param_value
# ---------------------------------------------------------------------------


class TestGetParamValue:
    def test_finds_length(self):
        assert get_param_value({"length": 10}, ["length", "window"], 5) == 10

    def test_finds_window(self):
        assert get_param_value({"window": 20}, ["length", "window"], 5) == 20

    def test_default_when_missing(self):
        assert get_param_value({"other": 1}, ["length", "window"], 5) == 5

    def test_length_takes_priority_over_window(self):
        assert (
            get_param_value({"length": 10, "window": 20}, ["length", "window"], 5) == 10
        )


# ---------------------------------------------------------------------------
# _run_indicator_with_validation
# ---------------------------------------------------------------------------


class TestRunWithIndicatorValidation:
    def test_returns_validation_when_present(self, sample_series):
        nan_result = create_nan_series_like(sample_series)
        result = _run_indicator_with_validation(
            nan_result,
            lambda: sample_series,
        )
        assert result is nan_result

    def test_calls_factory_when_no_validation(self, sample_series):
        called = False

        def factory():
            nonlocal called
            called = True
            return sample_series

        result = _run_indicator_with_validation(None, factory)
        assert called is True
        assert result is sample_series

    def test_fallback_when_result_missing(self, sample_series):
        fallback = create_nan_series_like(sample_series, fill_value=0.0)
        result = _run_indicator_with_validation(
            None,
            lambda: None,
            fallback_factory=lambda: fallback,
        )
        assert result is fallback

    def test_nan_series_when_result_missing_and_reference(self, sample_series):
        result = _run_indicator_with_validation(
            None,
            lambda: None,
            reference_series=sample_series,
        )
        assert result.isna().all()
        assert len(result) == len(sample_series)
