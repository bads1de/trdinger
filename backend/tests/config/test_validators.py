
import pytest
from backend.app.config.validators import MarketDataValidator, MLConfigValidator

# テスト用の設定値
SUPPORTED_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h"]
SYMBOL_MAPPING = {
    "BTCUSDT": "BTC/USDT:USDT",
    "BTC-USDT": "BTC/USDT:USDT",
    "BTC/USDT": "BTC/USDT:USDT",
}

class TestMarketDataValidator:
    """
    MarketDataValidatorのテストクラス
    """

    def test_validate_symbol_supported(self):
        """サポートされているシンボルの場合、Trueを返すことをテスト"""
        assert MarketDataValidator.validate_symbol("BTC/USDT:USDT", SUPPORTED_SYMBOLS) is True

    def test_validate_symbol_unsupported(self):
        """サポートされていないシンボルの場合、Falseを返すことをテスト"""
        assert MarketDataValidator.validate_symbol("XRP/USDT:USDT", SUPPORTED_SYMBOLS) is False

    def test_validate_timeframe_supported(self):
        """サポートされている時間軸の場合、Trueを返すことをテスト"""
        assert MarketDataValidator.validate_timeframe("15m", SUPPORTED_TIMEFRAMES) is True

    def test_validate_timeframe_unsupported(self):
        """サポートされていない時間軸の場合、Falseを返すことをテスト"""
        assert MarketDataValidator.validate_timeframe("5m", SUPPORTED_TIMEFRAMES) is False

    @pytest.mark.parametrize(
        "input_symbol, expected_symbol",
        [
            ("BTCUSDT", "BTC/USDT:USDT"),
            ("btc-usdt", "BTC/USDT:USDT"),
            ("  BTC/USDT  ", "BTC/USDT:USDT"),
            ("ETH/USDT:USDT", "ETH/USDT:USDT"),
        ],
    )
    def test_normalize_symbol_valid(self, input_symbol, expected_symbol):
        """有効なシンボルの正規化をテスト"""
        normalized = MarketDataValidator.normalize_symbol(
            input_symbol, SYMBOL_MAPPING, SUPPORTED_SYMBOLS
        )
        assert normalized == expected_symbol

    def test_normalize_symbol_unsupported(self):
        """サポートされていないシンボルの正規化でValueErrorが発生することをテスト"""
        with pytest.raises(ValueError) as excinfo:
            MarketDataValidator.normalize_symbol(
                "XRPUSDT", SYMBOL_MAPPING, SUPPORTED_SYMBOLS
            )
        assert "サポートされていないシンボルです" in str(excinfo.value)

    def test_normalize_symbol_not_in_mapping_but_supported(self):
        """マッピングにないがサポートされているシンボルの正規化をテスト"""
        normalized = MarketDataValidator.normalize_symbol(
            "ETH/USDT:USDT", SYMBOL_MAPPING, SUPPORTED_SYMBOLS
        )
        assert normalized == "ETH/USDT:USDT"

class TestMLConfigValidator:
    """
    MLConfigValidatorのテストクラス
    """

    @pytest.mark.parametrize(
        "predictions, expected",
        [
            ({"up": 0.5, "down": 0.3, "range": 0.2}, True),
            ({"up": 0.4, "down": 0.4, "range": 0.2}, True),
        ],
    )
    def test_validate_predictions_valid(self, predictions, expected):
        """有効な予測値の検証をテスト"""
        assert MLConfigValidator.validate_predictions(predictions) is expected

    @pytest.mark.parametrize(
        "predictions, expected",
        [
            ({"up": 0.5, "down": 0.3}, False),  # キーが不足
            ({"up": 0.5, "down": 0.3, "range": "0.2"}, False),  # 値が文字列
            ({"up": 1.1, "down": 0.1, "range": 0.1}, False),  # 値が範囲外
            ({"up": -0.1, "down": 0.5, "range": 0.6}, False), # 値が範囲外
            ({"up": 0.2, "down": 0.2, "range": 0.2}, False),  # 合計が0.8未満
            ({"up": 0.6, "down": 0.6, "range": 0.1}, False),  # 合計が1.2超過
            (None, False), # 辞書でない
            ("not a dict", False), # 辞書でない
        ],
    )
    def test_validate_predictions_invalid(self, predictions, expected):
        """無効な予測値の検証をテスト"""
        assert MLConfigValidator.validate_predictions(predictions) is expected
