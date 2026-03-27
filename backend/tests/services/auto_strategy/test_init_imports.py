"""
AutoStrategy パッケージの遅延インポートテスト
"""

import pytest

from app.services.auto_strategy import (
    AutoStrategyService,
    StrategyGene,
    GAConfig,
    TPSLService,
    PositionSizingService,
)


class TestAutoStrategyImports:
    """遅延インポートのテスト"""

    def test_auto_strategy_service_import(self):
        """AutoStrategyServiceがインポートできること"""
        assert AutoStrategyService is not None

    def test_strategy_gene_import(self):
        """StrategyGeneがインポートできること"""
        assert StrategyGene is not None

    def test_ga_config_import(self):
        """GAConfigがインポートできること"""
        assert GAConfig is not None

    def test_tpsl_service_import(self):
        """TPSLServiceがインポートできること"""
        assert TPSLService is not None

    def test_position_sizing_service_import(self):
        """PositionSizingServiceがインポートできること"""
        assert PositionSizingService is not None

    def test_invalid_attribute_raises_error(self):
        """無効な属性でAttributeErrorが発生すること"""
        import app.services.auto_strategy as auto_strategy
        
        with pytest.raises(AttributeError):
            _ = auto_strategy.InvalidAttribute
