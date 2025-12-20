"""
インジケーター設定テスト

ParameterConfig, IndicatorConfig, IndicatorConfigRegistry のテスト
"""

import pytest

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    IndicatorConfigRegistry,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
)


class TestParameterConfig:
    """ParameterConfig のテスト"""

    def test_validate_value(self):
        """値の検証テスト"""
        config = ParameterConfig(
            name="test", default_value=10, min_value=5, max_value=20
        )
        assert config.validate_value(10) is True
        assert config.validate_value(5) is True
        assert config.validate_value(20) is True
        assert config.validate_value(4) is False
        assert config.validate_value(21) is False
        assert config.validate_value("not_a_number") is True  # スキップされる仕様

    def test_get_range_for_preset(self):
        """プリセット範囲取得テスト"""
        presets = {"short": (2, 5), "long": (20, 50)}
        config = ParameterConfig(
            name="test", default_value=10, min_value=1, max_value=100, presets=presets
        )

        assert config.get_range_for_preset("short") == (2, 5)
        assert config.get_range_for_preset("long") == (20, 50)
        assert config.get_range_for_preset("unknown") == (1, 100)


class TestIndicatorConfig:
    """IndicatorConfig のテスト"""

    def test_validate_constraints_less_than(self):
        """less_than 制約のテスト"""
        constraints = [{"type": "less_than", "param1": "fast", "param2": "slow"}]
        config = IndicatorConfig(
            indicator_name="TEST", parameter_constraints=constraints
        )

        assert config.validate_constraints({"fast": 10, "slow": 20}) == (True, [])
        is_valid, errors = config.validate_constraints({"fast": 20, "slow": 10})
        assert is_valid is False
        assert "より小さくなければなりません" in errors[0]

    def test_validate_constraints_greater_than(self):
        """greater_than 制約のテスト"""
        constraints = [{"type": "greater_than", "param1": "long", "param2": "short"}]
        config = IndicatorConfig(
            indicator_name="TEST", parameter_constraints=constraints
        )

        assert config.validate_constraints({"long": 50, "short": 10}) == (True, [])
        is_valid, errors = config.validate_constraints({"long": 10, "short": 50})
        assert is_valid is False
        assert "より大きくなければなりません" in errors[0]

    def test_validate_constraints_min_difference(self):
        """min_difference 制約のテスト"""
        constraints = [
            {"type": "min_difference", "param1": "a", "param2": "b", "min_diff": 5}
        ]
        config = IndicatorConfig(
            indicator_name="TEST", parameter_constraints=constraints
        )

        assert config.validate_constraints({"a": 10, "b": 5}) == (True, [])
        is_valid, errors = config.validate_constraints({"a": 10, "b": 7})
        assert is_valid is False
        assert "最低 5 必要ですが" in errors[0]

    def test_normalize_params(self):
        """パラメータ正規化テスト"""
        param_map = {"p": "period"}
        parameters = {
            "period": ParameterConfig(
                name="period", default_value=14, min_value=2, max_value=100
            )
        }
        config = IndicatorConfig(
            indicator_name="TEST", param_map=param_map, parameters=parameters
        )

        # エイリアスマッピング
        norm = config.normalize_params({"p": 20})
        assert norm == {"period": 20}

        # 範囲制限
        norm = config.normalize_params({"period": 1})
        assert norm["period"] == 2
        norm = config.normalize_params({"period": 200})
        assert norm["period"] == 100

    def test_generate_random_parameters(self):
        """ランダムパラメータ生成テスト"""
        parameters = {
            "length": ParameterConfig(
                name="length", default_value=14, min_value=5, max_value=50
            )
        }
        config = IndicatorConfig(indicator_name="RSI", parameters=parameters)

        # ランダム生成を複数回テスト
        for _ in range(10):
            params = config.generate_random_parameters()
            assert "length" in params
            assert 5 <= params["length"] <= 50

    def test_generate_random_parameters_with_preset(self):
        """プリセット付きランダムパラメータ生成テスト"""
        parameters = {
            "length": ParameterConfig(
                name="length",
                default_value=14,
                min_value=2,
                max_value=200,
                presets={"short_term": (5, 15), "long_term": (50, 100)},
            )
        }
        config = IndicatorConfig(indicator_name="RSI", parameters=parameters)

        # short_termプリセット
        for _ in range(10):
            params = config.generate_random_parameters(preset="short_term")
            assert 5 <= params["length"] <= 15

        # long_termプリセット
        for _ in range(10):
            params = config.generate_random_parameters(preset="long_term")
            assert 50 <= params["length"] <= 100


class TestIndicatorConfigRegistry:
    """IndicatorConfigRegistry のテスト"""

    def test_register_and_get(self):
        """登録と取得テスト"""
        registry = IndicatorConfigRegistry()
        config = IndicatorConfig(indicator_name="RSI", aliases=["rsi"])
        registry.register(config)

        assert registry.get_indicator_config("RSI") == config
        assert registry.get_indicator_config("rsi") == config
        assert "RSI" in registry.list_indicators()
        assert "rsi" in registry.list_indicators()

    def test_reset(self):
        """リセットテスト"""
        registry = IndicatorConfigRegistry()
        config = IndicatorConfig(indicator_name="TEST")
        registry.register(config)

        assert len(registry.list_indicators()) > 0
        registry.reset()
        assert len(registry.list_indicators()) == 0

    def test_generate_parameters_for_indicator(self):
        """レジストリからのパラメータ生成テスト"""
        registry = IndicatorConfigRegistry()
        parameters = {
            "length": ParameterConfig(
                name="length", default_value=14, min_value=5, max_value=50
            )
        }
        config = IndicatorConfig(indicator_name="TEST", parameters=parameters)
        registry.register(config)

        params = registry.generate_parameters_for_indicator("TEST")
        assert "length" in params
        assert 5 <= params["length"] <= 50

    def test_generate_parameters_for_unknown_indicator(self):
        """未知の指標に対するパラメータ生成テスト"""
        registry = IndicatorConfigRegistry()
        params = registry.generate_parameters_for_indicator("UNKNOWN")
        assert params == {}


class TestIndicatorScaleType:
    """IndicatorScaleType のテスト"""

    def test_scale_type_values(self):
        """スケールタイプの値が正しいことをテスト"""
        assert IndicatorScaleType.OSCILLATOR_0_100.value == "oscillator_0_100"
        assert (
            IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100.value
            == "oscillator_plus_minus_100"
        )
        assert (
            IndicatorScaleType.MOMENTUM_ZERO_CENTERED.value == "momentum_zero_centered"
        )
        assert IndicatorScaleType.PRICE_RATIO.value == "price_ratio"


class TestIndicatorResultType:
    """IndicatorResultType のテスト"""

    def test_result_type_values(self):
        """結果タイプの値が正しいことをテスト"""
        assert IndicatorResultType.SINGLE.value == "single"
        assert IndicatorResultType.COMPLEX.value == "complex"
