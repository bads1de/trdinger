"""
auto_strategyパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__, __dir__）をテストします。
"""

import pytest

import app.services.auto_strategy as auto_strategy_package


class TestAutoStrategyInitExports:
    """auto_strategy/__init__.pyのエクスポートテスト"""

    def test_auto_strategy_service_lazy_load(self):
        """AutoStrategyServiceが遅延ロードされる"""
        from app.services.auto_strategy.services.auto_strategy_service import (
            AutoStrategyService,
        )

        service = getattr(auto_strategy_package, "AutoStrategyService")

        assert service is AutoStrategyService

    def test_strategy_gene_lazy_load(self):
        """StrategyGeneが遅延ロードされる"""
        from app.services.auto_strategy.genes import StrategyGene

        gene = getattr(auto_strategy_package, "StrategyGene")

        assert gene is StrategyGene

    def test_ga_config_lazy_load(self):
        """GAConfigが遅延ロードされる"""
        from app.services.auto_strategy.config import GAConfig

        config = getattr(auto_strategy_package, "GAConfig")

        assert config is GAConfig

    def test_tpsl_service_lazy_load(self):
        """TPSLServiceが遅延ロードされる"""
        from app.services.auto_strategy.tpsl import TPSLService

        service = getattr(auto_strategy_package, "TPSLService")

        assert service is TPSLService

    def test_position_sizing_service_lazy_load(self):
        """PositionSizingServiceが遅延ロードされる"""
        from app.services.auto_strategy.positions.position_sizing_service import (
            PositionSizingService,
        )

        service = getattr(auto_strategy_package, "PositionSizingService")

        assert service is PositionSizingService

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = auto_strategy_package.NonExistentAttribute

    def test_dir_includes_exports(self):
        """__dir__にエクスポートが含まれる"""
        dir_result = dir(auto_strategy_package)

        assert "AutoStrategyService" in dir_result
        assert "StrategyGene" in dir_result
        assert "GAConfig" in dir_result
        assert "TPSLService" in dir_result
        assert "PositionSizingService" in dir_result

    def test_dir_returns_list(self):
        """__dir__がリストを返す"""
        dir_result = dir(auto_strategy_package)

        assert isinstance(dir_result, list)
        assert len(dir_result) > 0

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "AutoStrategyService",
            "StrategyGene",
            "GAConfig",
            "TPSLService",
            "PositionSizingService",
        ]

        for item in expected_items:
            assert item in auto_strategy_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(auto_strategy_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert auto_strategy_package.__doc__ is not None
        assert len(auto_strategy_package.__doc__) > 0
