"""
generatorsパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.generators as generators_package


class TestAutoStrategyGeneratorsInitExports:
    """generators/__init__.pyのエクスポートテスト"""

    def test_random_gene_generator_exported(self):
        """RandomGeneGeneratorがエクスポートされている"""
        assert hasattr(generators_package, "RandomGeneGenerator")

    def test_condition_generator_exported(self):
        """ConditionGeneratorがエクスポートされている"""
        assert hasattr(generators_package, "ConditionGenerator")

    def test_seed_strategy_factory_exported(self):
        """SeedStrategyFactoryがエクスポートされている"""
        assert hasattr(generators_package, "SeedStrategyFactory")

    def test_inject_seeds_into_population_exported(self):
        """inject_seeds_into_populationがエクスポートされている"""
        assert hasattr(generators_package, "inject_seeds_into_population")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "RandomGeneGenerator",
            "ConditionGenerator",
            "SeedStrategyFactory",
            "inject_seeds_into_population",
        ]

        for item in expected_items:
            assert item in generators_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(generators_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert generators_package.__doc__ is not None
        assert len(generators_package.__doc__) > 0
