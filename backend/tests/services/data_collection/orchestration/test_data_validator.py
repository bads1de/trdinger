"""
data_collection/orchestration/data_validator モジュールのユニットテスト
"""

from unittest.mock import patch

import pytest

from app.services.data_collection.orchestration.data_validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


class TestDataValidator:
    @patch("app.services.data_collection.orchestration.data_validator.unified_config")
    def test_validate_valid_symbol_and_timeframe(self, mock_config, validator):
        mock_config.market.symbol_mapping = {}
        mock_config.market.supported_symbols = ["BTCUSDT", "ETHUSDT"]
        mock_config.market.supported_timeframes = ["1h", "4h", "1d"]

        result = validator.validate_symbol_and_timeframe("BTCUSDT", "1h")
        assert result == "BTCUSDT"

    @patch("app.services.data_collection.orchestration.data_validator.unified_config")
    def test_validate_with_symbol_mapping(self, mock_config, validator):
        mock_config.market.symbol_mapping = {"BTC/USDT": "BTCUSDT"}
        mock_config.market.supported_symbols = ["BTCUSDT"]
        mock_config.market.supported_timeframes = ["1h"]

        result = validator.validate_symbol_and_timeframe("BTC/USDT", "1h")
        assert result == "BTCUSDT"

    @patch("app.services.data_collection.orchestration.data_validator.unified_config")
    def test_validate_unsupported_symbol(self, mock_config, validator):
        mock_config.market.symbol_mapping = {}
        mock_config.market.supported_symbols = ["BTCUSDT"]
        mock_config.market.supported_timeframes = ["1h"]

        with pytest.raises(ValueError, match="サポートされていないシンボル"):
            validator.validate_symbol_and_timeframe("DOGEUSDT", "1h")

    @patch("app.services.data_collection.orchestration.data_validator.unified_config")
    def test_validate_invalid_timeframe(self, mock_config, validator):
        mock_config.market.symbol_mapping = {}
        mock_config.market.supported_symbols = ["BTCUSDT"]
        mock_config.market.supported_timeframes = ["1h", "4h"]

        with pytest.raises(ValueError, match="無効な時間軸"):
            validator.validate_symbol_and_timeframe("BTCUSDT", "2m")
