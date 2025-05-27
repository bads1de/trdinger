"""
更新された市場設定のユニットテスト

BTCとETHのみに制限されたスポット・先物ペアが
正しく設定されているかをテストします。
"""

import pytest
from app.config.market_config import MarketDataConfig


class TestUpdatedMarketConfig:
    """更新された市場設定のテストクラス"""

    def test_target_currencies_spot_pairs(self):
        """対象通貨のスポットペアが全て含まれているかテスト（BTCとETHのみ）"""
        target_currencies = ['BTC', 'ETH']

        for currency in target_currencies:
            spot_symbol = f"{currency}/USDT"
            assert spot_symbol in MarketDataConfig.SUPPORTED_SYMBOLS, \
                f"{currency}のスポットペア {spot_symbol} がサポートされていません"

    def test_target_currencies_futures_pairs(self):
        """対象通貨の先物ペアが全て含まれているかテスト（BTCとETHのみ）"""
        # BTCとETHのみの先物ペア
        expected_futures = [
            "BTC/USDT:USDT",  # Bitcoin USDT永続契約
            "BTCUSD",         # Bitcoin USD永続契約
            "ETH/USDT:USDT",  # Ethereum USDT永続契約
            "ETHUSD",         # Ethereum USD永続契約
        ]

        for futures_symbol in expected_futures:
            assert futures_symbol in MarketDataConfig.SUPPORTED_SYMBOLS, \
                f"先物ペア {futures_symbol} がサポートされていません"

    def test_symbol_validation_for_target_pairs(self):
        """対象ペアのシンボル検証テスト（BTCとETHのみ）"""
        test_symbols = [
            # スポット
            "BTC/USDT", "ETH/USDT", "ETH/BTC",
            # 先物（BTCとETHのみ）
            "BTC/USDT:USDT", "BTCUSD", "ETH/USDT:USDT", "ETHUSD"
        ]

        for symbol in test_symbols:
            assert MarketDataConfig.validate_symbol(symbol), \
                f"シンボル {symbol} の検証に失敗しました"

    def test_symbol_normalization_spot(self):
        """スポットペアの正規化テスト（BTCとETHのみ）"""
        test_cases = [
            # スポットペア（そのまま）
            ("BTC/USDT", "BTC/USDT"),
            ("ETH/USDT", "ETH/USDT"),
            # ハイフン表記からスポットへ
            ("BTC-USDT", "BTC/USDT"),
            ("ETH-USDT", "ETH/USDT"),
            # ETH/BTCペア
            ("ETHBTC", "ETH/BTC"),
            ("ETH-BTC", "ETH/BTC"),
        ]

        for input_symbol, expected in test_cases:
            result = MarketDataConfig.normalize_symbol(input_symbol)
            assert result == expected, \
                f"スポット正規化失敗: {input_symbol} → {result} (期待値: {expected})"

    def test_symbol_normalization_futures(self):
        """先物ペアの正規化テスト（BTCとETHのみ）"""
        test_cases = [
            # USDT永続契約
            ("BTCUSDT", "BTC/USDT:USDT"),
            ("ETHUSDT", "ETH/USDT:USDT"),
            # 正規化確認
            ("BTC/USDT:USDT", "BTC/USDT:USDT"),
            ("ETH/USDT:USDT", "ETH/USDT:USDT"),
        ]

        for input_symbol, expected in test_cases:
            result = MarketDataConfig.normalize_symbol(input_symbol)
            assert result == expected, \
                f"先物正規化失敗: {input_symbol} → {result} (期待値: {expected})"

    def test_legacy_symbol_normalization(self):
        """レガシーUSD永続契約の正規化テスト"""
        test_cases = [
            ("BTCUSD", "BTCUSD"),   # Bitcoin USD永続契約
            ("ETHUSD", "ETHUSD"),   # Ethereum USD永続契約
        ]

        for input_symbol, expected in test_cases:
            result = MarketDataConfig.normalize_symbol(input_symbol)
            assert result == expected, \
                f"USD永続契約正規化失敗: {input_symbol} → {result} (期待値: {expected})"

    def test_unsupported_symbol_raises_error(self):
        """サポートされていないシンボルでエラーが発生するかテスト"""
        unsupported_symbols = [
            "DOGE/USDT",  # サポート対象外
            "INVALID/PAIR",  # 無効なペア
            "BTC/EUR",  # サポートされていない建て通貨
        ]

        for symbol in unsupported_symbols:
            with pytest.raises(ValueError):
                MarketDataConfig.normalize_symbol(symbol)

    def test_symbol_count(self):
        """シンボル数が期待値と一致するかテスト（BTCとETHのみ）"""
        # BTCとETHのみのペア数
        # BTC: 3ペア (BTC/USDT, BTC/USDT:USDT, BTCUSD)
        # ETH: 4ペア (ETH/USDT, ETH/USDT:USDT, ETH/BTC, ETHUSD)
        # 合計: 3 + 4 = 7ペア
        expected_count = 7
        actual_count = len(MarketDataConfig.SUPPORTED_SYMBOLS)

        assert actual_count == expected_count, \
            f"シンボル数が期待値と異なります: {actual_count} (期待値: {expected_count})"

    def test_default_symbol_is_supported(self):
        """デフォルトシンボルがサポートされているかテスト"""
        default_symbol = MarketDataConfig.DEFAULT_SYMBOL
        assert default_symbol in MarketDataConfig.SUPPORTED_SYMBOLS, \
            f"デフォルトシンボル {default_symbol} がサポートされていません"

    def test_all_target_currencies_have_both_markets(self):
        """全ての対象通貨でスポットと先物の両方が利用可能かテスト（BTCとETHのみ）"""
        target_currencies = ['BTC', 'ETH']

        for currency in target_currencies:
            # スポット市場の確認
            spot_found = any(
                symbol.startswith(f"{currency}/") and ":" not in symbol
                for symbol in MarketDataConfig.SUPPORTED_SYMBOLS
            )
            assert spot_found, f"{currency}のスポット市場が見つかりません"

            # 先物市場の確認（BTCとETHのみ）
            futures_found = any(
                f"{currency}/USDT:USDT" in symbol or
                symbol == f"{currency}USD"
                for symbol in MarketDataConfig.SUPPORTED_SYMBOLS
            )
            assert futures_found, f"{currency}の先物市場が見つかりません"
