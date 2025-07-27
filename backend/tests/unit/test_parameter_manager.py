"""
IndicatorParameterManagerのテスト

パラメータ生成とバリデーションの一元化システムのテスト
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
)
from app.services.indicators.parameter_manager import (
    IndicatorParameterManager,
    ParameterGenerationError,
)


class TestIndicatorParameterManager:
    """IndicatorParameterManagerのテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.manager = IndicatorParameterManager()

    def test_generate_period_parameter(self):
        """期間パラメータの生成テスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=2,
                max_value=100,
                description="RSI計算期間",
            )
        )

        # パラメータ生成
        params = self.manager.generate_parameters("RSI", rsi_config)

        # 検証
        assert "period" in params
        assert isinstance(params["period"], int)
        assert 2 <= params["period"] <= 100

    def test_generate_macd_parameters(self):
        """MACDパラメータの生成テスト（制約エンジン使用）"""
        # MACD設定を作成
        macd_config = IndicatorConfig(
            indicator_name="MACD",
            required_data=["close"],
            result_type=IndicatorResultType.COMPLEX,
        )
        macd_config.add_parameter(
            ParameterConfig(
                name="fast_period",
                default_value=12,
                min_value=2,
                max_value=50,
                description="短期期間",
            )
        )
        macd_config.add_parameter(
            ParameterConfig(
                name="slow_period",
                default_value=26,
                min_value=2,
                max_value=100,
                description="長期期間",
            )
        )
        macd_config.add_parameter(
            ParameterConfig(
                name="signal_period",
                default_value=9,
                min_value=2,
                max_value=50,
                description="シグナル期間",
            )
        )

        # 複数回生成して制約エンジンの動作を確認
        for _ in range(10):
            params = self.manager.generate_parameters("MACD", macd_config)

            # 検証
            assert "fast_period" in params
            assert "slow_period" in params
            assert "signal_period" in params
            # 制約エンジンによってfast_period < slow_periodが保証される
            assert params["fast_period"] < params["slow_period"]
            assert 2 <= params["fast_period"] <= 50
            # slow_periodは制約エンジンによって調整される可能性がある
            assert params["slow_period"] >= params["fast_period"] + 1
            assert 2 <= params["signal_period"] <= 50

    def test_generate_bollinger_bands_parameters(self):
        """Bollinger Bandsパラメータの生成テスト"""
        # BB設定を作成
        bb_config = IndicatorConfig(
            indicator_name="BB",
            required_data=["close"],
            result_type=IndicatorResultType.COMPLEX,
        )
        bb_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=20,
                min_value=2,
                max_value=100,
                description="移動平均期間",
            )
        )
        bb_config.add_parameter(
            ParameterConfig(
                name="std_dev",
                default_value=2.0,
                min_value=1.0,
                max_value=3.0,
                description="標準偏差倍率",
            )
        )

        # パラメータ生成
        params = self.manager.generate_parameters("BB", bb_config)

        # 検証
        assert "period" in params
        assert "std_dev" in params
        assert isinstance(params["period"], int)
        assert isinstance(params["std_dev"], float)
        assert 2 <= params["period"] <= 100
        assert 1.0 <= params["std_dev"] <= 3.0

    def test_validate_parameters_success(self):
        """パラメータバリデーション成功テスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(indicator_name="RSI")
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        # 有効なパラメータでバリデーション
        params = {"period": 20}
        result = self.manager.validate_parameters("RSI", params, rsi_config)

        assert result is True

    def test_validate_parameters_failure(self):
        """パラメータバリデーション失敗テスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(indicator_name="RSI")
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        # 無効なパラメータでバリデーション（範囲外）
        params = {"period": 150}
        result = self.manager.validate_parameters("RSI", params, rsi_config)

        assert result is False

    def test_validate_parameters_missing_parameter(self):
        """必須パラメータ不足のバリデーションテスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(indicator_name="RSI")
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        # パラメータ不足でバリデーション
        params = {}
        result = self.manager.validate_parameters("RSI", params, rsi_config)

        assert result is False

    def test_get_parameter_ranges(self):
        """パラメータ範囲取得テスト"""
        # MACD設定を作成
        macd_config = IndicatorConfig(indicator_name="MACD")
        macd_config.add_parameter(
            ParameterConfig(
                name="fast_period", default_value=12, min_value=2, max_value=50
            )
        )
        macd_config.add_parameter(
            ParameterConfig(
                name="slow_period", default_value=26, min_value=2, max_value=100
            )
        )

        # 範囲取得
        ranges = self.manager.get_parameter_ranges("MACD", macd_config)

        # 検証
        assert "fast_period" in ranges
        assert "slow_period" in ranges
        assert ranges["fast_period"] == {"min": 2, "max": 50, "default": 12}
        assert ranges["slow_period"] == {"min": 2, "max": 100, "default": 26}

    def test_generate_parameters_no_params_indicator(self):
        """パラメータ不要な指標のテスト（OBVなど）"""
        # OBV設定を作成（パラメータなし）
        obv_config = IndicatorConfig(
            indicator_name="OBV",
            required_data=["close", "volume"],
            result_type=IndicatorResultType.SINGLE,
        )

        # パラメータ生成
        params = self.manager.generate_parameters("OBV", obv_config)

        # 検証
        assert params == {}

    def test_generate_parameters_unknown_indicator(self):
        """未知の指標に対するエラーハンドリングテスト"""
        # 指標タイプと設定名が一致しない場合
        unknown_config = IndicatorConfig(indicator_name="UNKNOWN")

        # エラーが発生することを確認（指標タイプ不一致）
        with pytest.raises(ParameterGenerationError):
            self.manager.generate_parameters("DIFFERENT_TYPE", unknown_config)

def test_parameter_generation_consistency(self):
        """パラメータ生成の一貫性テスト"""
        # 同じ設定で複数回生成しても、すべて有効な範囲内であることを確認
        rsi_config = IndicatorConfig(indicator_name="RSI")
        rsi_config.add_parameter(
            ParameterConfig(name="period", default_value=14, min_value=2, max_value=100)
        )

        generated_params = []
        for _ in range(100):
            params = self.manager.generate_parameters("RSI", rsi_config)
            generated_params.append(params["period"])
            assert self.manager.validate_parameters("RSI", params, rsi_config)

        # 生成された値に多様性があることを確認
        unique_values = set(generated_params)
        assert len(unique_values) > 1  # 少なくとも2つ以上の異なる値が生成される

    def test_indicator_config_integration(self):
        """IndicatorConfigとの統合テスト"""
        # RSI設定を作成
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
        )
        rsi_config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=14,
                min_value=2,
                max_value=100,
                description="RSI計算期間",
            )
        )

        # IndicatorConfigの新しいメソッドをテスト
        assert rsi_config.has_parameters() is True

        # パラメータ範囲取得
        ranges = rsi_config.get_parameter_ranges()
        assert "period" in ranges
        assert ranges["period"]["min"] == 2
        assert ranges["period"]["max"] == 100
        assert ranges["period"]["default"] == 14

        # ランダムパラメータ生成
        params = rsi_config.generate_random_parameters()
        assert "period" in params
        assert 2 <= params["period"] <= 100

        # パラメータなしの指標
        obv_config = IndicatorConfig(indicator_name="OBV")
        assert obv_config.has_parameters() is False
        assert obv_config.get_parameter_ranges() == {}
        assert obv_config.generate_random_parameters() == {}
