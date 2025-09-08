"""DataConverterクラスのユニットテスト"""

import os
import sys
import unittest
from unittest.mock import patch
from typing import Any, Dict, List
import logging

# パス調整
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../app/services/auto_strategy/utils'))

from data_converters import DataConverter


class TestDataConverter(unittest.TestCase):
    """DataConverterクラスのテスト"""

    def test_ensure_float_valid_input(self):
        """ensure_float: 有効なfloat入力"""
        self.assertEqual(DataConverter.ensure_float(3.14), 3.14)
        self.assertEqual(DataConverter.ensure_float("2.5"), 2.5)
        self.assertEqual(DataConverter.ensure_float(10), 10.0)

    def test_ensure_float_invalid_input_with_default(self):
        """ensure_float: 無効な入力でデフォルト値使用"""
        with patch('backend.app.services.auto_strategy.utils.data_converters.logger') as mock_logger:
            self.assertEqual(DataConverter.ensure_float("invalid"), 0.0)
            mock_logger.warning.assert_called_once()

        with patch('data_converters.logger') as mock_logger:
            self.assertEqual(DataConverter.ensure_float(None, 5.5), 5.5)
            mock_logger.warning.assert_called_once()

    def test_ensure_int_valid_input(self):
        """ensure_int: 有効なint入力"""
        self.assertEqual(DataConverter.ensure_int(42), 42)
        self.assertEqual(DataConverter.ensure_int("123"), 123)
        self.assertEqual(DataConverter.ensure_int(3.14), 3)  # floatの切り捨て

    def test_ensure_int_invalid_input_with_default(self):
        """ensure_int: 無効な入力でデフォルト値使用"""
        with patch('backend.app.services.auto_strategy.utils.data_converters.logger') as mock_logger:
            self.assertEqual(DataConverter.ensure_int("invalid"), 0)
            mock_logger.warning.assert_called_once()

        with patch('data_converters.logger') as mock_logger:
            self.assertEqual(DataConverter.ensure_int([], 10), 10)
            mock_logger.warning.assert_called_once()

    def test_ensure_list_list_input(self):
        """ensure_list: リスト入力はそのまま返却"""
        input_list = [1, 2, 3]
        self.assertEqual(DataConverter.ensure_list(input_list), input_list)

    def test_ensure_list_none_input(self):
        """ensure_list: None入力でデフォルトを使用"""
        self.assertEqual(DataConverter.ensure_list(None), [])
        self.assertEqual(DataConverter.ensure_list(None, [1, 2, 3]), [1, 2, 3])

    def test_ensure_list_other_input(self):
        """ensure_list: リスト以外をリスト化"""
        self.assertEqual(DataConverter.ensure_list("test"), ["test"])
        self.assertEqual(DataConverter.ensure_list(42), [42])

    def test_ensure_dict_dict_input(self):
        """ensure_dict: 辞書入力はそのまま返却"""
        input_dict = {"key": "value"}
        self.assertEqual(DataConverter.ensure_dict(input_dict), input_dict)

    def test_ensure_dict_none_input(self):
        """ensure_dict: None入力でデフォルトを使用"""
        self.assertEqual(DataConverter.ensure_dict(None), {})
        self.assertEqual(DataConverter.ensure_dict(None, {"default": "value"}), {"default": "value"})

    def test_ensure_dict_other_input(self):
        """ensure_dict: 辞書以外はデフォルト返却"""
        self.assertEqual(DataConverter.ensure_dict("invalid"), {})
        self.assertEqual(DataConverter.ensure_dict([1, 2, 3]), {})

    def test_normalize_symbol_valid_input(self):
        """normalize_symbol: 有効なシンボル正規化"""
        self.assertEqual(DataConverter.normalize_symbol("btc:usdt"), "BTC:USDT")
        self.assertEqual(DataConverter.normalize_symbol(" eth/btc "), "ETH/BTC")
        self.assertEqual(DataConverter.normalize_symbol("BNB-USDT"), "BNB-USDT")

    def test_normalize_symbol_none_input(self):
        """normalize_symbol: None入力でデフォルト使用"""
        self.assertEqual(DataConverter.normalize_symbol(None), "BTC:USDT")

    def test_normalize_symbol_empty_input(self):
        """normalize_symbol: 空文字列入力でデフォルト使用"""
        self.assertEqual(DataConverter.normalize_symbol(""), "BTC:USDT")


if __name__ == "__main__":
    unittest.main()