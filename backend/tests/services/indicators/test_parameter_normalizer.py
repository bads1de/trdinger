"""
parameter_normalizer モジュールのユニットテスト
"""

import pytest

from app.services.indicators.parameter_normalizer import ParameterNormalizer


@pytest.fixture
def normalizer():
    return ParameterNormalizer()


class TestParameterNormalizer:
    def test_normalize_params_basic(self, normalizer):
        config = {
            "params": {"length": ["length", "period"]},
            "default_values": {"length": 14},
        }
        result = normalizer.normalize_params({"length": 20}, config)
        assert result["length"] == 20

    def test_normalize_params_alias(self, normalizer):
        config = {
            "params": {"length": ["length", "period", "window"]},
            "default_values": {"length": 14},
        }
        result = normalizer.normalize_params({"period": 30}, config)
        assert result["length"] == 30

    def test_normalize_params_default_value(self, normalizer):
        config = {
            "params": {"length": ["length"]},
            "default_values": {"length": 14},
        }
        result = normalizer.normalize_params({}, config)
        assert result["length"] == 14

    def test_normalize_params_multiple_params(self, normalizer):
        config = {
            "params": {
                "length": ["length", "period"],
                "signal": ["signal"],
            },
            "default_values": {"length": 14, "signal": 9},
        }
        result = normalizer.normalize_params({"length": 20}, config)
        assert result["length"] == 20
        assert result["signal"] == 9

    def test_min_length_guard_function(self, normalizer):
        config = {
            "params": {"length": ["length"]},
            "default_values": {"length": 14},
            "min_length": lambda p: max(p.get("length", 14), 5),
        }
        result = normalizer.normalize_params({"length": 3}, config)
        assert result["length"] == 5

    def test_min_length_guard_fixed_value(self, normalizer):
        config = {
            "params": {"length": ["length"]},
            "default_values": {"length": 14},
            "min_length": 10,
        }
        result = normalizer.normalize_params({"length": 5}, config)
        assert result["length"] == 10

    def test_min_length_guard_not_applied_to_non_length(self, normalizer):
        config = {
            "params": {"signal": ["signal"]},
            "default_values": {"signal": 9},
            "min_length": 10,
        }
        result = normalizer.normalize_params({"signal": 3}, config)
        assert result["signal"] == 3
