"""
data_utilsモジュールのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.core.utils.data_utils import (
    ensure_series,
    ensure_numeric_series,
    validate_series_length,
    validate_series_data,
    create_series_with_index,
    safe_array_conversion,
    convert_to_series,
    _ensure_series,
    DataConversionError,
)


class TestEnsureSeries:
    """ensure_series関数のテスト"""

    def test_with_pandas_series(self):
        """pandas.Seriesの場合"""
        original = pd.Series([1, 2, 3], name="test")
        result = ensure_series(original)
        assert isinstance(result, pd.Series)
        assert result.equals(original)
        assert result.name == "test"

    def test_with_pandas_series_rename(self):
        """pandas.Seriesの名前変更"""
        original = pd.Series([1, 2, 3], name="original")
        result = ensure_series(original, name="new_name")
        assert result.name == "new_name"
        # データは同じだが、名前が違うのでオブジェクトは異なる
        assert result is not original

    def test_with_list(self):
        """リストの場合"""
        data = [1, 2, 3, 4, 5]
        result = ensure_series(data)
        assert isinstance(result, pd.Series)
        assert list(result.values) == data

    def test_with_numpy_array(self):
        """numpy配列の場合"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ensure_series(data)
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, data)

    def test_with_backtesting_array(self):
        """backtesting.pyの_Arrayオブジェクトの場合"""
        # _data属性を持つモックオブジェクト
        mock_array = Mock()
        mock_array._data = [1, 2, 3, 4, 5]

        result = ensure_series(mock_array)
        assert isinstance(result, pd.Series)
        assert list(result.values) == [1, 2, 3, 4, 5]

    def test_with_values_attribute(self):
        """valuesアトリビュートを持つオブジェクトの場合"""
        mock_obj = Mock()
        mock_obj.values = np.array([1, 2, 3])
        # _data属性がないことを明示的に設定
        del mock_obj._data

        result = ensure_series(mock_obj)
        assert isinstance(result, pd.Series)
        assert list(result.values) == [1, 2, 3]

    def test_with_scalar(self):
        """スカラー値の場合"""
        result = ensure_series(42)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == 42

    def test_with_invalid_type_raise_error(self):
        """サポートされていないデータ型（例外発生）"""
        # 辞書型など、明らかにサポートされていない型を使用
        invalid_data = {"key": "value"}
        with pytest.raises(DataConversionError):
            ensure_series(invalid_data, raise_on_error=True)

    def test_with_invalid_type_no_raise(self):
        """サポートされていないデータ型（例外なし）"""
        # 辞書型など、明らかにサポートされていない型を使用
        invalid_data = {"key": "value"}
        result = ensure_series(invalid_data, raise_on_error=False)
        assert isinstance(result, pd.Series)
        assert len(result) == 0


class TestEnsureNumericSeries:
    """ensure_numeric_series関数のテスト"""

    def test_with_numeric_data(self):
        """数値データの場合"""
        data = [1, 2, 3, 4, 5]
        result = ensure_numeric_series(data)
        assert isinstance(result, pd.Series)
        assert result.dtype in [np.int64, np.float64]

    def test_with_string_numbers(self):
        """文字列数値の場合"""
        data = ["1", "2", "3", "4", "5"]
        result = ensure_numeric_series(data, raise_on_error=False)
        assert isinstance(result, pd.Series)
        assert list(result.values) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_with_invalid_strings_raise_error(self):
        """無効な文字列（例外発生）"""
        data = ["1", "2", "invalid", "4", "5"]
        with pytest.raises(DataConversionError):
            ensure_numeric_series(data, raise_on_error=True)

    def test_with_invalid_strings_no_raise(self):
        """無効な文字列（例外なし）"""
        data = ["1", "2", "invalid", "4", "5"]
        result = ensure_numeric_series(data, raise_on_error=False)
        assert isinstance(result, pd.Series)
        assert pd.isna(result.iloc[2])  # "invalid"はNaNになる


class TestValidateSeriesLength:
    """validate_series_length関数のテスト"""

    def test_with_same_length(self):
        """同じ長さのSeriesの場合"""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([4, 5, 6])
        s3 = pd.Series([7, 8, 9])

        # 例外が発生しないことを確認
        validate_series_length(s1, s2, s3)

    def test_with_different_length(self):
        """異なる長さのSeriesの場合"""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([4, 5])

        with pytest.raises(DataConversionError):
            validate_series_length(s1, s2)

    def test_with_empty_series(self):
        """空のSeriesの場合"""
        s1 = pd.Series([])
        s2 = pd.Series([])

        with pytest.raises(DataConversionError):
            validate_series_length(s1, s2)

    def test_with_no_series(self):
        """Seriesが指定されていない場合"""
        with pytest.raises(DataConversionError):
            validate_series_length()


class TestValidateSeriesData:
    """validate_series_data関数のテスト"""

    def test_with_valid_data(self):
        """有効なデータの場合"""
        series = pd.Series([1, 2, 3, 4, 5])
        # 例外が発生しないことを確認
        validate_series_data(series)

    def test_with_none(self):
        """Noneの場合"""
        with pytest.raises(DataConversionError):
            validate_series_data(None)

    def test_with_short_data(self):
        """データが短い場合"""
        series = pd.Series([1, 2])
        with pytest.raises(DataConversionError):
            validate_series_data(series, min_length=5)

    def test_with_all_nan(self):
        """全てNaNの場合"""
        series = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(DataConversionError):
            validate_series_data(series)


class TestCreateSeriesWithIndex:
    """create_series_with_index関数のテスト"""

    def test_with_matching_length(self):
        """データとインデックスの長さが一致する場合"""
        data = [1, 2, 3]
        index = pd.date_range("2024-01-01", periods=3)

        result = create_series_with_index(data, index, "test")
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.name == "test"
        assert result.index.equals(index)

    def test_with_mismatched_length(self):
        """データとインデックスの長さが一致しない場合"""
        data = [1, 2, 3]
        index = pd.date_range("2024-01-01", periods=2)

        with pytest.raises(DataConversionError):
            create_series_with_index(data, index)

    def test_without_index(self):
        """インデックスなしの場合"""
        data = [1, 2, 3]
        result = create_series_with_index(data, name="test")
        assert isinstance(result, pd.Series)
        assert result.name == "test"


class TestSafeArrayConversion:
    """safe_array_conversion関数のテスト"""

    def test_with_numpy_array(self):
        """numpy配列の場合"""
        data = np.array([1, 2, 3])
        result = safe_array_conversion(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_with_pandas_series(self):
        """pandas.Seriesの場合"""
        data = pd.Series([1, 2, 3])
        result = safe_array_conversion(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data.values)

    def test_with_backtesting_array(self):
        """backtesting.pyの_Arrayオブジェクトの場合"""
        mock_array = Mock()
        mock_array._data = [1, 2, 3]

        result = safe_array_conversion(mock_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])


class TestBackwardCompatibility:
    """後方互換性関数のテスト"""

    def test_convert_to_series(self):
        """convert_to_series関数のテスト"""
        data = [1, 2, 3]
        result = convert_to_series(data)
        assert isinstance(result, pd.Series)
        assert list(result.values) == data

    def test_ensure_series_alias(self):
        """_ensure_series関数のテスト"""
        data = [1, 2, 3]
        result = _ensure_series(data)
        assert isinstance(result, pd.Series)
        assert list(result.values) == data

    def test_ensure_series_alias_with_error(self):
        """_ensure_series関数のエラーテスト"""
        # 辞書型など、明らかにサポートされていない型を使用
        invalid_data = {"key": "value"}
        with pytest.raises(DataConversionError):
            _ensure_series(invalid_data)
