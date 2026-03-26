import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError
from app.services.backtest.config.backtest_config import BacktestConfig, BacktestConfigValidationError


class TestBacktestConfig:
    @pytest.fixture
    def valid_config(self):
        return {
            "strategy_name": "SMA",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2023, 1, 2),
            "initial_capital": 10000,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "MANUAL",
                "parameters": {},
            },
        }

    def test_validate_config_success(self, valid_config):
        # 例外が発生しないこと
        config = BacktestConfig(**valid_config)
        assert config.strategy_name == "SMA"

    def test_validate_missing_required_fields(self):
        config = {"symbol": "BTC/USDT"}
        with pytest.raises(ValidationError):
            BacktestConfig(**config)

    def test_validate_empty_strategy_name(self, valid_config):
        valid_config["strategy_name"] = ""
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_invalid_timeframe(self, valid_config):
        valid_config["timeframe"] = "invalid"
        with pytest.raises(ValidationError, match="timeframe"):
            BacktestConfig(**valid_config)

    def test_validate_dates_reversed(self, valid_config):
        valid_config["start_date"] = datetime(2023, 1, 2)
        valid_config["end_date"] = datetime(2023, 1, 1)
        with pytest.raises(BacktestConfigValidationError, match="start_dateはend_dateより前"):
            BacktestConfig(**valid_config)

    def test_validate_dates_future(self, valid_config):
        future_date = datetime.now() + timedelta(days=10)
        valid_config["end_date"] = future_date
        with pytest.raises(BacktestConfigValidationError, match="現在時刻より前"):
            BacktestConfig(**valid_config)

    def test_validate_zero_initial_capital(self, valid_config):
        valid_config["initial_capital"] = 0
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_negative_commission_rate(self, valid_config):
        valid_config["commission_rate"] = -0.1
        with pytest.raises(ValidationError):
            BacktestConfig(**valid_config)

    def test_validate_string_dates(self, valid_config):
        valid_config["start_date"] = "2023-01-01"
        valid_config["end_date"] = "2023-01-02"
        config = BacktestConfig(**valid_config)
        assert isinstance(config.start_date, datetime)
