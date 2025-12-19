import pytest
from app.services.indicators.config.indicator_config import (
    ParameterConfig, IndicatorConfig, IndicatorConfigRegistry,
    IndicatorResultType, IndicatorScaleType, generate_positional_functions
)

class TestParameterConfig:
    def test_validate_value(self):
        config = ParameterConfig(name="test", default_value=10, min_value=5, max_value=20)
        assert config.validate_value(10) is True
        assert config.validate_value(5) is True
        assert config.validate_value(20) is True
        assert config.validate_value(4) is False
        assert config.validate_value(21) is False
        assert config.validate_value("not_a_number") is True # スキップされる仕様

    def test_get_range_for_preset(self):
        presets = {"short": (2, 5), "long": (20, 50)}
        config = ParameterConfig(name="test", default_value=10, min_value=1, max_value=100, presets=presets)
        
        assert config.get_range_for_preset("short") == (2, 5)
        assert config.get_range_for_preset("long") == (20, 50)
        assert config.get_range_for_preset("unknown") == (1, 100)

class TestIndicatorConfig:
    def test_validate_constraints_less_than(self):
        constraints = [{"type": "less_than", "param1": "fast", "param2": "slow"}]
        config = IndicatorConfig(indicator_name="TEST", parameter_constraints=constraints)
        
        assert config.validate_constraints({"fast": 10, "slow": 20}) == (True, [])
        is_valid, errors = config.validate_constraints({"fast": 20, "slow": 10})
        assert is_valid is False
        assert "より小さくなければなりません" in errors[0]

    def test_validate_constraints_greater_than(self):
        constraints = [{"type": "greater_than", "param1": "long", "param2": "short"}]
        config = IndicatorConfig(indicator_name="TEST", parameter_constraints=constraints)
        
        assert config.validate_constraints({"long": 50, "short": 10}) == (True, [])
        is_valid, errors = config.validate_constraints({"long": 10, "short": 50})
        assert is_valid is False
        assert "より大きくなければなりません" in errors[0]

    def test_validate_constraints_min_difference(self):
        constraints = [{"type": "min_difference", "param1": "a", "param2": "b", "min_diff": 5}]
        config = IndicatorConfig(indicator_name="TEST", parameter_constraints=constraints)
        
        assert config.validate_constraints({"a": 10, "b": 5}) == (True, [])
        is_valid, errors = config.validate_constraints({"a": 10, "b": 7})
        assert is_valid is False
        assert "最低 5 必要ですが" in errors[0]

    def test_normalize_params(self):
        param_map = {"p": "period"}
        parameters = {"period": ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)}
        config = IndicatorConfig(indicator_name="TEST", param_map=param_map, parameters=parameters)
        
        # エイリアスマッピング
        norm = config.normalize_params({"p": 20})
        assert norm == {"period": 20}
        
        # 範囲制限
        norm = config.normalize_params({"period": 1})
        assert norm["period"] == 2
        norm = config.normalize_params({"period": 200})
        assert norm["period"] == 100

class TestIndicatorConfigRegistry:
    def test_register_and_get(self):
        registry = IndicatorConfigRegistry()
        config = IndicatorConfig(indicator_name="RSI", aliases=["rsi"])
        registry.register(config)
        
        assert registry.get_indicator_config("RSI") == config
        assert registry.get_indicator_config("rsi") == config
        assert "RSI" in registry.list_indicators()
        assert "rsi" in registry.list_indicators()

    def test_generate_positional_functions(self):
        # グローバルなindicator_registryを使用するため、状態に依存する可能性があるが、
        # 基本的な動作を確認
        funcs = generate_positional_functions()
        assert isinstance(funcs, set)
        assert "rsi" in funcs
