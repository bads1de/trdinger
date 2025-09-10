"""DataConverterクラスのテストモジュール"""

import logging
import pytest
from app.services.auto_strategy.utils.data_converters import DataConverter

class TestDataConverter:
    """DataConverterクラスのテスト"""

    def test_ensure_float_valid_values(self):
        """有効な値のfloat変換テスト"""
        assert DataConverter.ensure_float(1) == 1.0
        assert DataConverter.ensure_float(1.5) == 1.5
        assert DataConverter.ensure_float("2.5") == 2.5
        assert DataConverter.ensure_float(False) == 0.0
        assert DataConverter.ensure_float(None, 5.0) == 5.0

    def test_ensure_float_edge_cases(self):
        """エッジケースのfloat変換テスト（バグ発見目的）"""
        # nan処理
        assert DataConverter.ensure_float(float('nan'), 10.0) == 10.0
        # inf処理
        assert DataConverter.ensure_float(float('inf'), 10.0) == 10.0
        assert DataConverter.ensure_float(float('-inf'), 10.0) == 10.0
        # 無効文字列
        assert DataConverter.ensure_float("abc", 5.0) == 5.0
        # 非数値
        assert DataConverter.ensure_float([], 5.0) == 5.0
        assert DataConverter.ensure_float({}, 5.0) == 5.0

    def test_ensure_int_valid_values(self):
        """有効な値のint変換テスト"""
        assert DataConverter.ensure_int(5) == 5
        assert DataConverter.ensure_int(5.7) == 5  # 切り捨て？
        assert DataConverter.ensure_int("42") == 42
        assert DataConverter.ensure_int(False) == 0
        assert DataConverter.ensure_int(None, 10) == 10

    def test_ensure_int_edge_cases(self):
        """エッジケースのint変換テスト"""
        # 浮動小数点
        assert DataConverter.ensure_int(3.14, 5) == 3
        # nan
        assert DataConverter.ensure_int(float('nan'), 5) == 5
        # inf
        assert DataConverter.ensure_int(float('inf'), 5) == 5
        # 無効文字列
        assert DataConverter.ensure_int("invalid", 5) == 5

    def test_ensure_list_valid_values(self):
        """有効な値のlist変換テスト"""
        assert DataConverter.ensure_list([1, 2, 3]) == [1, 2, 3]
        assert DataConverter.ensure_list(None) == []
        assert DataConverter.ensure_list(None, [4, 5]) == [4, 5]
        assert DataConverter.ensure_list("abc") == ["abc"]
        assert DataConverter.ensure_list(42) == [42]

    def test_ensure_list_edge_cases(self):
        """エッジケースのlist変換テスト"""
        # dict
        assert DataConverter.ensure_list({"key": "value"}) == [{"key": "value"}]
        # tuple
        assert DataConverter.ensure_list((1, 2)) == [(1, 2)]

    def test_ensure_dict_valid_values(self):
        """有効な値のdict変換テスト"""
        assert DataConverter.ensure_dict({"a": 1}) == {"a": 1}
        assert DataConverter.ensure_dict(None) == {}
        assert DataConverter.ensure_dict(None, {"default": True}) == {"default": True}
        assert DataConverter.ensure_dict("not_dict") == {}
        assert DataConverter.ensure_dict([1, 2]) == {}

    def test_ensure_dict_edge_cases(self):
        """エッジケースのdict変換テスト"""
        # list
        assert DataConverter.ensure_dict([1, 2, 3]) == {}
        # int
        assert DataConverter.ensure_dict(123) == {}

    def test_normalize_symbol_valid_cases(self):
        """有効なsymbolの正規化テスト"""
        assert DataConverter.normalize_symbol("btc/usdt") == "BTC/USDT"
        assert DataConverter.normalize_symbol("  eth-btc  ") == "ETH-BTC"
        assert DataConverter.normalize_symbol("BNB:USD") == "BNB:USD"

    def test_normalize_symbol_edge_cases(self):
        """エッジケースのsymbol正規化テスト"""
        assert DataConverter.normalize_symbol(None) == "BTC:USDT"
        assert DataConverter.normalize_symbol("") == "BTC:USDT"
        assert DataConverter.normalize_symbol("   ") == "BTC:USDT"
        assert DataConverter.normalize_symbol("btc") == "BTC"