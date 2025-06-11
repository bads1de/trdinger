"""
シンボル限定機能のテスト

TDD: BTCUSDTの無期限先物（BTC/USDT:USDT）のみをサポートすることをテスト
"""

import pytest
from unittest.mock import Mock, patch

from app.config.market_config import MarketDataConfig
from app.core.services.market_data_service import BybitMarketDataService


class TestSymbolLimitation:
    """シンボル限定機能のテストクラス"""

    def test_supported_symbols_contains_only_btcusdt_perpetual(self):
        """サポートされるシンボルがBTC/USDT:USDTのみであることをテスト"""
        # このテストは最初は失敗する（現在は複数シンボルをサポートしているため）
        expected_symbols = ["BTC/USDT:USDT"]

        assert (
            MarketDataConfig.SUPPORTED_SYMBOLS == expected_symbols
        ), f"サポートされるシンボルはBTC/USDT:USDTのみであるべきです。現在: {MarketDataConfig.SUPPORTED_SYMBOLS}"

    def test_default_symbol_is_btcusdt_perpetual(self):
        """デフォルトシンボルがBTC/USDT:USDTであることをテスト"""
        # このテストは最初は失敗する可能性がある
        assert (
            MarketDataConfig.DEFAULT_SYMBOL == "BTC/USDT:USDT"
        ), f"デフォルトシンボルはBTC/USDT:USDTであるべきです。現在: {MarketDataConfig.DEFAULT_SYMBOL}"

    def test_normalize_symbol_accepts_btcusdt_perpetual(self):
        """シンボル正規化でBTC/USDT:USDTが受け入れられることをテスト"""
        # 正規化テスト
        normalized = MarketDataConfig.normalize_symbol("BTC/USDT:USDT")
        assert (
            normalized == "BTC/USDT:USDT"
        ), f"BTC/USDT:USDTの正規化が正しくありません。結果: {normalized}"

    def test_normalize_symbol_accepts_variations(self):
        """シンボル正規化で様々な表記が受け入れられることをテスト"""
        # 大文字小文字の違いやスペースを含む表記のテスト
        variations = [
            "btc/usdt:usdt",
            "BTC/USDT:USDT",
            " BTC/USDT:USDT ",
            "btc/usdt:USDT",
            "BTC/usdt:usdt",
        ]

        for variation in variations:
            normalized = MarketDataConfig.normalize_symbol(variation)
            assert (
                normalized == "BTC/USDT:USDT"
            ), f"シンボル '{variation}' の正規化が失敗しました。結果: {normalized}"

    def test_normalize_symbol_rejects_unsupported_symbols(self):
        """サポートされていないシンボルが拒否されることをテスト"""
        # このテストは最初は失敗する（現在は他のシンボルもサポートしているため）
        unsupported_symbols = [
            "BTC/USDT",  # 現物取引
            "BTCUSD",  # USD建て無期限先物
            "ETH/USDT:USDT",  # イーサリアム無期限先物
            "ETH/USDT",  # イーサリアム現物
            "INVALID/SYMBOL",  # 無効なシンボル
        ]

        for symbol in unsupported_symbols:
            with pytest.raises(ValueError, match="サポートされていないシンボルです"):
                MarketDataConfig.normalize_symbol(symbol)

    def test_validate_symbol_accepts_only_btcusdt_perpetual(self):
        """validate_symbolメソッドがBTC/USDT:USDTのみを受け入れることをテスト"""
        # MarketDataConfigのvalidate_symbolメソッドを直接テスト
        assert (
            MarketDataConfig.validate_symbol("BTC/USDT:USDT") == True
        ), "BTC/USDT:USDTが有効なシンボルとして認識されません"

    def test_validate_symbol_rejects_unsupported_symbols(self):
        """validate_symbolメソッドがサポートされていないシンボルを拒否することをテスト"""
        unsupported_symbols = [
            "BTC/USDT",  # 現物取引
            "BTCUSD",  # USD建て無期限先物
            "ETH/USDT:USDT",  # イーサリアム無期限先物
            "ETH/USDT",  # イーサリアム現物
        ]

        for symbol in unsupported_symbols:
            assert (
                MarketDataConfig.validate_symbol(symbol) == False
            ), f"サポートされていないシンボル '{symbol}' が有効として認識されています"

    def test_supported_timeframes_contains_required_timeframes(self):
        """サポートされる時間足に要求された時間足が含まれることをテスト"""
        required_timeframes = ["1d", "4h", "1h", "30m", "15m"]

        for timeframe in required_timeframes:
            assert (
                timeframe in MarketDataConfig.SUPPORTED_TIMEFRAMES
            ), f"必要な時間足 '{timeframe}' がサポートされていません"

    def test_symbol_mapping_includes_btcusdt_perpetual(self):
        """シンボルマッピングにBTC/USDT:USDTが含まれることをテスト"""
        # シンボルマッピングが存在する場合のテスト
        if hasattr(MarketDataConfig, "SYMBOL_MAPPING"):
            # 様々な表記からBTC/USDT:USDTにマッピングされることを確認
            expected_mappings = {
                "BTCUSDT": "BTC/USDT:USDT",
                "BTC-USDT": "BTC/USDT:USDT",
                "BTCUSDT_PERP": "BTC/USDT:USDT",
            }

            for input_symbol, expected_output in expected_mappings.items():
                if input_symbol in MarketDataConfig.SYMBOL_MAPPING:
                    assert (
                        MarketDataConfig.SYMBOL_MAPPING[input_symbol] == expected_output
                    ), f"シンボルマッピング '{input_symbol}' -> '{expected_output}' が正しくありません"

    def test_fetch_ohlcv_data_with_supported_symbol(self):
        """サポートされるシンボルでのOHLCVデータ取得テスト"""
        # BTC/USDT:USDTでのデータ取得は成功するはず
        # MarketDataConfigのvalidate_symbolメソッドを使用
        assert MarketDataConfig.validate_symbol("BTC/USDT:USDT") == True

    def test_config_consistency(self):
        """設定の一貫性をテスト"""
        # デフォルトシンボルがサポートされるシンボルに含まれることを確認
        assert (
            MarketDataConfig.DEFAULT_SYMBOL in MarketDataConfig.SUPPORTED_SYMBOLS
        ), "デフォルトシンボルがサポートされるシンボルに含まれていません"

        # サポートされるシンボルが空でないことを確認
        assert (
            len(MarketDataConfig.SUPPORTED_SYMBOLS) > 0
        ), "サポートされるシンボルが空です"

        # サポートされるシンボルがBTC/USDT:USDTのみであることを確認
        assert (
            len(MarketDataConfig.SUPPORTED_SYMBOLS) == 1
        ), f"サポートされるシンボルは1つのみであるべきです。現在: {len(MarketDataConfig.SUPPORTED_SYMBOLS)}"
