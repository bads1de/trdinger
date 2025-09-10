"""compat_utils関数のテストモジュール"""

import pytest
from app.services.auto_strategy.utils.compat_utils import ensure_float, normalize_symbol, safe_execute

class TestCompatUtils:
    """compat_utilsのテスト"""

    def test_ensure_float_valid_values(self):
        """有効な値のfloat変換テスト"""
        assert ensure_float(1) == 1.0
        assert ensure_float(1.5) == 1.5
        assert ensure_float("2.5") == 2.5
        assert ensure_float(False) == 0.0
        assert ensure_float(None, 5.0) == 5.0

    def test_ensure_float_edge_cases(self):
        """エッジケースのfloat変換テスト"""
        # nan処理
        assert ensure_float(float('nan'), 10.0) == 10.0
        # inf処理
        assert ensure_float(float('inf'), 10.0) == 10.0
        # 無効文字列
        assert ensure_float("abc", 5.0) == 5.0
        # 非数値
        assert ensure_float([], 5.0) == 5.0

    def test_normalize_symbol_valid_cases(self):
        """有効なsymbolの正規化テスト"""
        assert normalize_symbol("btc/usdt") == "BTC/USDT"
        assert normalize_symbol("  eth-btc  ") == "ETH-BTC"

    def test_normalize_symbol_edge_cases(self):
        """エッジケースのsymbol正規化テスト"""
        assert normalize_symbol(None) == "BTC:USDT"
        assert normalize_symbol("") == "BTC:USDT"
        assert normalize_symbol("   ") == "BTC:USDT"
        assert normalize_symbol("btc") == "BTC"

    def test_safe_execute_basic(self):
        """safe_executeの基本テスト"""
        def sample_func(x):
            return x * 2

        result = safe_execute(lambda: sample_func(5))
        assert result == 10

    def test_safe_execute_with_exception(self):
        """safe_execute例外処理テスト"""
        def failing_func():
            raise ValueError("Test error")

        # 例外が発生してもsafe_executeはログ出力し、デフォルト値を返す（デフォルトはNone）
        result = safe_execute(lambda: failing_func())
        assert result is None