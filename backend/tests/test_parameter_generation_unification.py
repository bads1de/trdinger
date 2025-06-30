"""
ランダムパラメータ生成ロジック統一のテスト

TDDアプローチで、ParameterManagerにインジケーターのパラメータ生成と検証の責務を
集中させ、StrategyGeneがParameterManagerを利用することを検証するテスト
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.indicators.parameter_manager import IndicatorParameterManager
from app.core.services.indicators.config.indicator_config import IndicatorConfig, ParameterConfig
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.core.services.auto_strategy.models.gene_encoding import GeneEncoding
from app.core.services.auto_strategy.utils.parameter_generators import generate_indicator_parameters


class TestParameterGenerationUnification:
    """ランダムパラメータ生成ロジック統一のテストクラス"""

    @pytest.fixture
    def mock_parameter_manager(self):
        """モックされたParameterManagerを作成"""
        return Mock(spec=IndicatorParameterManager)

    @pytest.fixture
    def sample_indicator_config(self):
        """テスト用のIndicatorConfigを作成"""
        return IndicatorConfig(
            indicator_name="SMA",
            parameters={
                "period": ParameterConfig(
                    name="period",
                    default_value=20,
                    min_value=5,
                    max_value=200,
                    description="移動平均の期間"
                )
            },
            description="単純移動平均",
            category="trend"
        )

    @pytest.fixture
    def sample_macd_config(self):
        """テスト用のMACDのIndicatorConfigを作成"""
        return IndicatorConfig(
            indicator_name="MACD",
            parameters={
                "fast_period": ParameterConfig(
                    name="fast_period",
                    default_value=12,
                    min_value=5,
                    max_value=20,
                    description="高速期間"
                ),
                "slow_period": ParameterConfig(
                    name="slow_period",
                    default_value=26,
                    min_value=20,
                    max_value=50,
                    description="低速期間"
                ),
                "signal_period": ParameterConfig(
                    name="signal_period",
                    default_value=9,
                    min_value=5,
                    max_value=15,
                    description="シグナル期間"
                )
            },
            description="MACD",
            category="momentum"
        )

    def test_parameter_manager_generates_valid_parameters(self, sample_indicator_config):
        """ParameterManagerが有効なパラメータを生成することを検証"""
        manager = IndicatorParameterManager()
        
        # パラメータ生成を複数回実行して範囲内であることを確認
        for _ in range(10):
            params = manager.generate_parameters("SMA", sample_indicator_config)
            
            assert "period" in params
            assert isinstance(params["period"], int)
            assert 5 <= params["period"] <= 200

    def test_parameter_manager_validates_generated_parameters(self, sample_indicator_config):
        """ParameterManagerが生成したパラメータを検証することを確認"""
        manager = IndicatorParameterManager()
        
        params = manager.generate_parameters("SMA", sample_indicator_config)
        
        # 生成されたパラメータが検証を通ることを確認
        is_valid = manager.validate_parameters("SMA", params, sample_indicator_config)
        assert is_valid

    def test_parameter_manager_handles_macd_parameters(self, sample_macd_config):
        """ParameterManagerがMACDの複雑なパラメータを正しく処理することを検証"""
        manager = IndicatorParameterManager()
        
        params = manager.generate_parameters("MACD", sample_macd_config)
        
        assert "fast_period" in params
        assert "slow_period" in params
        assert "signal_period" in params
        
        # fast_period < slow_periodの制約が守られていることを確認
        assert params["fast_period"] < params["slow_period"]
        
        # 各パラメータが範囲内であることを確認
        assert 5 <= params["fast_period"] <= 20
        assert 20 <= params["slow_period"] <= 50
        assert 5 <= params["signal_period"] <= 15

    def test_random_gene_generator_uses_parameter_manager(self, mock_parameter_manager):
        """RandomGeneGeneratorがParameterManagerを使用することを検証"""
        # ParameterManagerのモック設定
        mock_parameter_manager.generate_parameters.return_value = {"period": 20}
        
        with patch('app.core.services.auto_strategy.generators.random_gene_generator.IndicatorParameterManager') as mock_manager_class:
            mock_manager_class.return_value = mock_parameter_manager
            
            with patch('app.core.services.auto_strategy.generators.random_gene_generator.indicator_registry') as mock_registry:
                # indicator_registryのモック設定
                mock_config = Mock()
                mock_registry.get_config.return_value = mock_config
                
                # RandomGeneGeneratorのインスタンス作成
                generator = RandomGeneGenerator(available_indicators=["SMA"])
                
                # _generate_random_indicatorsメソッドを呼び出し
                indicators = generator._generate_random_indicators()
                
                # ParameterManagerが使用されたことを確認
                mock_manager_class.assert_called()
                mock_parameter_manager.generate_parameters.assert_called()

    def test_gene_encoding_uses_parameter_manager(self, mock_parameter_manager):
        """GeneEncodingがParameterManagerを使用することを検証"""
        mock_parameter_manager.generate_parameters.return_value = {"period": 15}
        
        with patch('app.core.services.auto_strategy.models.gene_encoding.IndicatorParameterManager') as mock_manager_class:
            mock_manager_class.return_value = mock_parameter_manager
            
            with patch('app.core.services.auto_strategy.models.gene_encoding.indicator_registry') as mock_registry:
                mock_config = Mock()
                mock_registry.get_config.return_value = mock_config
                
                # GeneEncodingのインスタンス作成
                encoding = GeneEncoding()
                
                # _generate_indicator_parametersメソッドを呼び出し
                params = encoding._generate_indicator_parameters("SMA", 0.5)
                
                # ParameterManagerが使用されたことを確認
                mock_manager_class.assert_called()
                mock_parameter_manager.generate_parameters.assert_called()
                assert params == {"period": 15}

    def test_generate_indicator_parameters_function_uses_parameter_manager(self, mock_parameter_manager):
        """generate_indicator_parameters関数がParameterManagerを使用することを検証"""
        mock_parameter_manager.generate_parameters.return_value = {"period": 25}
        
        with patch('app.core.services.auto_strategy.utils.parameter_generators.IndicatorParameterManager') as mock_manager_class:
            mock_manager_class.return_value = mock_parameter_manager
            
            with patch('app.core.services.auto_strategy.utils.parameter_generators.indicator_registry') as mock_registry:
                mock_config = Mock()
                mock_registry.get_config.return_value = mock_config
                
                # generate_indicator_parameters関数を呼び出し
                params = generate_indicator_parameters("SMA")
                
                # ParameterManagerが使用されたことを確認
                mock_manager_class.assert_called()
                mock_parameter_manager.generate_parameters.assert_called()
                assert params == {"period": 25}

    def test_no_parameter_generator_class_usage(self):
        """ParameterGeneratorクラスが使用されていないことを検証"""
        import inspect
        from app.core.services.auto_strategy.generators import random_gene_generator
        from app.core.services.auto_strategy.models import gene_encoding
        from app.core.services.auto_strategy.utils import parameter_generators
        
        modules = [random_gene_generator, gene_encoding, parameter_generators]
        
        for module in modules:
            source = inspect.getsource(module)
            # ParameterGeneratorクラスの直接使用がないことを確認
            assert 'ParameterGenerator.' not in source, f"{module.__name__} still uses ParameterGenerator class"

    def test_parameter_manager_usage_in_auto_strategy(self):
        """auto_strategy関連モジュールでParameterManagerが使用されていることを検証"""
        import inspect
        from app.core.services.auto_strategy.generators import random_gene_generator
        from app.core.services.auto_strategy.models import gene_encoding
        from app.core.services.auto_strategy.utils import parameter_generators
        
        modules = [random_gene_generator, gene_encoding, parameter_generators]
        
        for module in modules:
            source = inspect.getsource(module)
            # ParameterManagerの使用があることを確認
            assert 'IndicatorParameterManager' in source, f"{module.__name__} does not use IndicatorParameterManager"

    def test_parameter_consistency_across_modules(self, sample_indicator_config):
        """異なるモジュールで同じ指標に対して一貫したパラメータが生成されることを検証"""
        # 同じシードを使用して一貫性を確認
        import random
        random.seed(42)
        
        manager1 = IndicatorParameterManager()
        params1 = manager1.generate_parameters("SMA", sample_indicator_config)
        
        random.seed(42)
        manager2 = IndicatorParameterManager()
        params2 = manager2.generate_parameters("SMA", sample_indicator_config)
        
        # 同じシードで同じ結果が得られることを確認
        assert params1 == params2

    def test_parameter_validation_consistency(self, sample_indicator_config):
        """パラメータ検証が一貫していることを確認"""
        manager = IndicatorParameterManager()
        
        # 有効なパラメータ
        valid_params = {"period": 20}
        assert manager.validate_parameters("SMA", valid_params, sample_indicator_config)
        
        # 無効なパラメータ（範囲外）
        invalid_params = {"period": 300}
        assert not manager.validate_parameters("SMA", invalid_params, sample_indicator_config)
        
        # 無効なパラメータ（キー不足）
        incomplete_params = {}
        assert not manager.validate_parameters("SMA", incomplete_params, sample_indicator_config)

    def test_fallback_mechanism_removed(self):
        """フォールバック機能が削除されていることを検証"""
        import inspect
        from app.core.services.auto_strategy.utils import parameter_generators
        
        source = inspect.getsource(parameter_generators)
        # 旧システムのフォールバックロジックが削除されていることを確認
        assert 'PARAMETER_GENERATORS' not in source, "Old fallback system still exists"
        assert 'generate_period_parameter' not in source, "Old parameter generation methods still exist"
