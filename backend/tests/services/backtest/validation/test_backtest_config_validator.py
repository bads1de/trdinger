import pytest
from datetime import datetime, timedelta
from app.services.backtest.validation.backtest_config_validator import BacktestConfigValidator, BacktestConfigValidationError

class TestBacktestConfigValidator:
    @pytest.fixture
    def validator(self):
        return BacktestConfigValidator()

    @pytest.fixture
    def valid_config(self):
        return {
            "strategy_name": "SMA",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "initial_capital": 10000,
            "commission_rate": 0.001
        }

    def test_validate_config_success(self, validator, valid_config):
        # 例外が発生しないこと
        validator.validate_config(valid_config)

    def test_validate_required_fields_missing(self, validator):
        config = {"symbol": "BTC/USDT"}
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(config)
        assert "必須フィールド" in str(excinfo.value)

    def test_validate_field_values_invalid(self, validator, valid_config):
        # シンボル空
        valid_config["symbol"] = " "
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "symbol" in str(excinfo.value)

        # 無効なタイムフレーム
        valid_config["symbol"] = "BTC/USDT" # 戻す
        valid_config["timeframe"] = "invalid"
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "timeframe" in str(excinfo.value)

    def test_validate_dates_reversed(self, validator, valid_config):
        valid_config["start_date"] = "2023-01-02"
        valid_config["end_date"] = "2023-01-01"
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "start_dateはend_dateより前" in str(excinfo.value)

    def test_validate_dates_future(self, validator, valid_config):
        future_date = (datetime.now() + timedelta(days=10)).isoformat()
        valid_config["end_date"] = future_date
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "現在時刻より前" in str(excinfo.value)

    def test_validate_numeric_fields_invalid(self, validator, valid_config):
        # 資金0
        valid_config["initial_capital"] = 0
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "initial_capital" in str(excinfo.value)

        # 手数料マイナス
        valid_config["initial_capital"] = 1000
        valid_config["commission_rate"] = -0.1
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "commission_rate" in str(excinfo.value)

    def test_validate_strategy_config(self, validator, valid_config):
        valid_config["strategy_config"] = "not_a_dict"
        with pytest.raises(BacktestConfigValidationError) as excinfo:
            validator.validate_config(valid_config)
        assert "辞書である必要があります" in str(excinfo.value)
