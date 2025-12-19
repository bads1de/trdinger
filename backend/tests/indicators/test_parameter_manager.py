import pytest
from unittest.mock import MagicMock
from app.services.indicators.parameter_manager import IndicatorParameterManager
from app.services.indicators.config.indicator_config import IndicatorConfig, ParameterConfig

class TestIndicatorParameterManager:
    @pytest.fixture
    def manager(self):
        return IndicatorParameterManager()

    @pytest.fixture
    def rsi_config(self):
        config = MagicMock(spec=IndicatorConfig)
        config.indicator_name = "RSI"
        config.aliases = ["rsi"]
        
        param_config = MagicMock(spec=ParameterConfig)
        param_config.min_value = 2
        param_config.max_value = 100
        param_config.default_value = 14
        param_config.validate_value.side_effect = lambda v: 2 <= v <= 100
        
        config.parameters = {"period": param_config}
        return config

    def test_generate_parameters_success(self, manager, rsi_config):
        params = manager.generate_parameters("RSI", rsi_config)
        assert "period" in params
        assert 2 <= params["period"] <= 100

    def test_generate_parameters_mismatch_type(self, manager, rsi_config):
        with pytest.raises(Exception, match="指標タイプが一致しません"):
            manager.generate_parameters("MACD", rsi_config)

    def test_generate_parameters_no_params(self, manager):
        config = MagicMock(spec=IndicatorConfig)
        config.indicator_name = "TEST"
        config.aliases = []
        config.parameters = {}
        
        params = manager.generate_parameters("TEST", config)
        assert params == {}

    def test_validate_parameters_valid(self, manager, rsi_config):
        assert manager.validate_parameters("RSI", {"period": 14}, rsi_config) is True

    def test_validate_parameters_missing_param(self, manager, rsi_config):
        assert manager.validate_parameters("RSI", {}, rsi_config) is False

    def test_validate_parameters_out_of_range(self, manager, rsi_config):
        assert manager.validate_parameters("RSI", {"period": 1}, rsi_config) is False

    def test_validate_parameters_unexpected_param(self, manager, rsi_config):
        assert manager.validate_parameters("RSI", {"period": 14, "extra": 10}, rsi_config) is False

    def test_generate_standard_parameters_with_preset(self, manager, rsi_config):
        param_config = rsi_config.parameters["period"]
        param_config.get_range_for_preset.return_value = (5, 10)
        
        params = manager._generate_standard_parameters(rsi_config, preset="short_term")
        assert 5 <= params["period"] <= 10
        param_config.get_range_for_preset.assert_called_with("short_term")

    def test_generate_standard_parameters_default_no_range(self, manager):
        config = MagicMock(spec=IndicatorConfig)
        param_config = MagicMock(spec=ParameterConfig)
        param_config.min_value = None
        param_config.max_value = None
        param_config.default_value = 42
        config.parameters = {"p": param_config}
        
        params = manager._generate_standard_parameters(config)
        assert params["p"] == 42
