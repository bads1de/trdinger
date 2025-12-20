"""
IndicatorConfig パラメータ生成テスト

IndicatorConfig.generate_random_parameters() の機能をテストする。
（旧 IndicatorParameterManager のテストを統合）
"""

import random

import pytest

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    ParameterConfig,
)


class TestIndicatorConfigParameterGeneration:
    """IndicatorConfig パラメータ生成のテスト"""

    @pytest.fixture
    def rsi_config(self):
        """テスト用RSI設定"""
        param_config = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
        )

        return IndicatorConfig(
            indicator_name="RSI",
            aliases=["rsi"],
            parameters={"period": param_config},
        )

    def test_generate_random_parameters_success(self, rsi_config):
        """パラメータ生成成功テスト"""
        random.seed(42)
        params = rsi_config.generate_random_parameters()
        assert "period" in params
        assert 2 <= params["period"] <= 100

    def test_generate_random_parameters_no_params(self):
        """パラメータなし設定のテスト"""
        config = IndicatorConfig(
            indicator_name="TEST",
            aliases=[],
            parameters={},
        )

        params = config.generate_random_parameters()
        assert params == {}

    def test_validate_value_in_range(self, rsi_config):
        """パラメータ検証成功テスト"""
        param_config = rsi_config.parameters["period"]
        assert param_config.validate_value(14) is True
        assert param_config.validate_value(2) is True
        assert param_config.validate_value(100) is True

    def test_validate_value_out_of_range(self, rsi_config):
        """範囲外パラメータテスト"""
        param_config = rsi_config.parameters["period"]
        assert param_config.validate_value(1) is False
        assert param_config.validate_value(101) is False

    def test_generate_random_parameters_with_preset(self):
        """プリセット付きパラメータ生成テスト"""
        param_config = ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            presets={"short_term": (5, 10)},
        )

        config = IndicatorConfig(
            indicator_name="RSI",
            parameters={"period": param_config},
        )

        # 複数回テスト
        for _ in range(10):
            params = config.generate_random_parameters(preset="short_term")
            assert 5 <= params["period"] <= 10

    def test_generate_random_parameters_no_range(self):
        """範囲なしパラメータのデフォルト値テスト"""
        param_config = ParameterConfig(
            name="p",
            default_value=42,
            min_value=None,
            max_value=None,
        )

        config = IndicatorConfig(
            indicator_name="TEST",
            parameters={"p": param_config},
        )

        params = config.generate_random_parameters()
        assert params["p"] == 42


class TestIndicatorConfigParameterGenerationIntegration:
    """IndicatorConfig パラメータ生成統合テスト"""

    def test_generate_with_alias(self):
        """エイリアスを持つ設定からの生成テスト"""
        param_config = ParameterConfig(
            name="length",
            default_value=14,
            min_value=5,
            max_value=50,
        )

        config = IndicatorConfig(
            indicator_name="SMA",
            aliases=["sma", "simple_moving_average"],
            parameters={"length": param_config},
        )

        params = config.generate_random_parameters()
        assert "length" in params
        assert 5 <= params["length"] <= 50

    def test_generate_with_float_parameters(self):
        """浮動小数点パラメータの生成テスト"""
        param_config = ParameterConfig(
            name="multiplier",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
        )

        config = IndicatorConfig(
            indicator_name="TEST",
            parameters={"multiplier": param_config},
        )

        for _ in range(10):
            params = config.generate_random_parameters()
            assert "multiplier" in params
            assert 0.5 <= params["multiplier"] <= 5.0

    def test_normalize_params_clamps_values(self):
        """パラメータ正規化（範囲制限）テスト"""
        param_config = ParameterConfig(
            name="length",
            default_value=14,
            min_value=5,
            max_value=50,
        )

        config = IndicatorConfig(
            indicator_name="TEST",
            parameters={"length": param_config},
        )

        # 下限を下回る値
        normalized = config.normalize_params({"length": 1})
        assert normalized["length"] == 5

        # 上限を超える値
        normalized = config.normalize_params({"length": 100})
        assert normalized["length"] == 50

        # 範囲内の値
        normalized = config.normalize_params({"length": 20})
        assert normalized["length"] == 20
