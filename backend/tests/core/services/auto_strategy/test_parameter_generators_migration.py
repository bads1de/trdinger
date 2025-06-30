"""
parameter_generators.py移行テスト

新しいIndicatorParameterManagerシステムへの移行をテストします。
"""

import pytest
from unittest.mock import patch, Mock

from app.core.services.auto_strategy.utils.parameter_generators import (
    generate_indicator_parameters,
    ParameterGenerator,
)
from app.core.services.indicators.config.indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
)


class TestParameterGeneratorsMigration:
    """parameter_generators.py移行のテストクラス"""

    def test_generate_indicator_parameters_with_registry(self):
        """レジストリに登録された指標のパラメータ生成テスト"""
        # RSI設定をモック
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

        # indicator_registryをモック
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = rsi_config
            
            # パラメータ生成
            params = generate_indicator_parameters("RSI")
            
            # 検証
            assert "period" in params
            assert isinstance(params["period"], int)
            assert 2 <= params["period"] <= 100
            mock_registry.get_config.assert_called_once_with("RSI")

    def test_generate_indicator_parameters_fallback_to_legacy(self):
        """レジストリに未登録の指標のフォールバックテスト"""
        # indicator_registryをモック（Noneを返す）
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = None
            
            # 期間のみの指標（SMA）
            params = generate_indicator_parameters("SMA")
            
            # 検証：従来のロジックが使用される
            assert "period" in params
            assert isinstance(params["period"], int)
            assert 5 <= params["period"] <= 50  # 従来の範囲

    def test_generate_indicator_parameters_fallback_on_error(self):
        """新システムでエラーが発生した場合のフォールバックテスト"""
        # indicator_registryをモック（例外を発生）
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.side_effect = Exception("Registry error")
            
            # パラメータ生成
            params = generate_indicator_parameters("EMA")
            
            # 検証：フォールバックが動作
            assert "period" in params
            assert isinstance(params["period"], int)
            assert 5 <= params["period"] <= 50

    def test_generate_indicator_parameters_no_params_indicator(self):
        """パラメータ不要な指標のテスト"""
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = None
            
            # OBV（パラメータ不要）
            params = generate_indicator_parameters("OBV")
            
            # 検証
            assert params == {}

    def test_generate_indicator_parameters_special_indicators(self):
        """特別な処理が必要な指標のフォールバックテスト"""
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = None
            
            # MACD（特別な処理が必要）
            params = generate_indicator_parameters("MACD")
            
            # 検証
            assert "fast_period" in params
            assert "slow_period" in params
            assert "signal_period" in params
            assert params["fast_period"] < params["slow_period"]

    def test_backward_compatibility_with_existing_code(self):
        """既存コードとの後方互換性テスト"""
        # 複数の指標タイプで従来と同じ結果が得られることを確認
        test_indicators = ["SMA", "EMA", "RSI", "CCI", "ADX", "ATR", "OBV"]
        
        for indicator_type in test_indicators:
            with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
                mock_registry.get_config.return_value = None
                
                # パラメータ生成
                params = generate_indicator_parameters(indicator_type)
                
                # 基本的な検証
                assert isinstance(params, dict)
                
                if indicator_type == "OBV":
                    assert params == {}
                else:
                    assert "period" in params
                    assert isinstance(params["period"], int)
                    assert 5 <= params["period"] <= 50

    def test_legacy_parameter_generator_methods(self):
        """従来のParameterGeneratorメソッドのテスト"""
        # 期間パラメータ生成
        params = ParameterGenerator.generate_period_parameter()
        assert "period" in params
        assert 5 <= params["period"] <= 50

        # MACD パラメータ生成
        macd_params = ParameterGenerator.generate_macd_parameters()
        assert "fast_period" in macd_params
        assert "slow_period" in macd_params
        assert "signal_period" in macd_params

        # Bollinger Bands パラメータ生成
        bb_params = ParameterGenerator.generate_bollinger_bands_parameters()
        assert "period" in bb_params
        assert "std_dev" in bb_params
        assert 15 <= bb_params["period"] <= 25
        assert 1.5 <= bb_params["std_dev"] <= 2.5

        # Stochastic パラメータ生成
        stoch_params = ParameterGenerator.generate_stochastic_parameters()
        assert "k_period" in stoch_params
        assert "d_period" in stoch_params
        assert "slow_k_period" in stoch_params

    def test_migration_consistency(self):
        """移行の一貫性テスト"""
        # 新システムと従来システムで同じ指標タイプに対して
        # 適切な範囲のパラメータが生成されることを確認
        
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
                min_value=5,  # 従来システムと同じ範囲
                max_value=50,
                description="RSI計算期間",
            )
        )

        # 新システムでの生成
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = rsi_config
            new_params = generate_indicator_parameters("RSI")

        # 従来システムでの生成
        with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
            mock_registry.get_config.return_value = None
            legacy_params = generate_indicator_parameters("RSI")

        # 両方とも同じ範囲内であることを確認
        assert 5 <= new_params["period"] <= 50
        assert 5 <= legacy_params["period"] <= 50
