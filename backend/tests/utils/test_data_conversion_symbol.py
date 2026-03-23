"""
normalize_market_symbol ユーティリティ関数のテスト

data_conversion モジュールに追加される共通シンボル正規化関数のテスト。
BybitService._normalize_symbol_for_ccxt および
BaseDataCollectionOrchestrationService._normalize_derivative_symbol
の共通化をカバーする。
"""

import pytest

from app.utils.data_conversion import normalize_market_symbol


class TestNormalizeMarketSymbol:
    """normalize_market_symbol 関数のテスト"""

    def test_already_normalized_symbol_is_returned_as_is(self):
        """コロン付きシンボルはそのまま返す"""
        assert normalize_market_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
        assert normalize_market_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"
        assert normalize_market_symbol("BTC/USD:USD") == "BTC/USD:USD"

    def test_slash_usdt_symbol_is_appended_colon_usdt(self):
        """/USDT で終わるシンボルには :USDT を付加する"""
        assert normalize_market_symbol("BTC/USDT") == "BTC/USDT:USDT"
        assert normalize_market_symbol("ETH/USDT") == "ETH/USDT:USDT"

    def test_slash_usd_symbol_is_appended_colon_usd(self):
        """/USD で終わるシンボルには :USD を付加する"""
        assert normalize_market_symbol("BTC/USD") == "BTC/USD:USD"
        assert normalize_market_symbol("ETH/USD") == "ETH/USD:USD"

    def test_unknown_format_defaults_to_colon_usdt(self):
        """認識できない形式はデフォルトで :USDT を付加する"""
        # BybitService._normalize_symbol_for_ccxt / _normalize_derivative_symbol
        # の既存の挙動に合わせる
        assert normalize_market_symbol("BTCUSDT") == "BTCUSDT:USDT"

    def test_non_string_input_is_converted_to_string_first(self):
        """非文字列入力は str() 変換してから正規化する（既存の BybitService の挙動）"""
        result = normalize_market_symbol(123)  # type: ignore[arg-type]
        assert isinstance(result, str)
        assert ":USDT" in result
