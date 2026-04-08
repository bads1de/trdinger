"""
genesパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.genes as genes_package


class TestAutoStrategyGenesInitExports:
    """genes/__init__.pyのエクスポートテスト"""

    def test_strategy_gene_exported(self):
        """StrategyGeneがエクスポートされている"""
        assert hasattr(genes_package, "StrategyGene")

    def test_indicator_gene_exported(self):
        """IndicatorGeneがエクスポートされている"""
        assert hasattr(genes_package, "IndicatorGene")

    def test_condition_exported(self):
        """Conditionがエクスポートされている"""
        assert hasattr(genes_package, "Condition")

    def test_condition_group_exported(self):
        """ConditionGroupがエクスポートされている"""
        assert hasattr(genes_package, "ConditionGroup")

    def test_stateful_condition_exported(self):
        """StatefulConditionがエクスポートされている"""
        assert hasattr(genes_package, "StatefulCondition")

    def test_state_tracker_exported(self):
        """StateTrackerがエクスポートされている"""
        assert hasattr(genes_package, "StateTracker")

    def test_entry_direction_exported(self):
        """EntryDirectionがエクスポートされている"""
        assert hasattr(genes_package, "EntryDirection")

    def test_tpsl_gene_exported(self):
        """TPSLGeneがエクスポートされている"""
        assert hasattr(genes_package, "TPSLGene")

    def test_position_sizing_gene_exported(self):
        """PositionSizingGeneがエクスポートされている"""
        assert hasattr(genes_package, "PositionSizingGene")

    def test_tpsl_result_exported(self):
        """TPSLResultがエクスポートされている"""
        assert hasattr(genes_package, "TPSLResult")

    def test_entry_gene_exported(self):
        """EntryGeneがエクスポートされている"""
        assert hasattr(genes_package, "EntryGene")

    def test_tool_gene_exported(self):
        """ToolGeneがエクスポートされている"""
        assert hasattr(genes_package, "ToolGene")

    def test_position_sizing_method_exported(self):
        """PositionSizingMethodがエクスポートされている"""
        assert hasattr(genes_package, "PositionSizingMethod")

    def test_tpsl_method_exported(self):
        """TPSLMethodがエクスポートされている"""
        assert hasattr(genes_package, "TPSLMethod")

    def test_entry_type_exported(self):
        """EntryTypeがエクスポートされている"""
        assert hasattr(genes_package, "EntryType")

    def test_gene_validator_exported(self):
        """GeneValidatorがエクスポートされている"""
        assert hasattr(genes_package, "GeneValidator")

    def test_create_random_position_sizing_gene_exported(self):
        """create_random_position_sizing_geneがエクスポートされている"""
        assert hasattr(genes_package, "create_random_position_sizing_gene")

    def test_create_random_tpsl_gene_exported(self):
        """create_random_tpsl_geneがエクスポートされている"""
        assert hasattr(genes_package, "create_random_tpsl_gene")

    def test_create_random_entry_gene_exported(self):
        """create_random_entry_geneがエクスポートされている"""
        assert hasattr(genes_package, "create_random_entry_gene")

    def test_generate_random_indicators_exported(self):
        """generate_random_indicatorsがエクスポートされている"""
        assert hasattr(genes_package, "generate_random_indicators")

    def test_create_random_indicator_gene_exported(self):
        """create_random_indicator_geneがエクスポートされている"""
        assert hasattr(genes_package, "create_random_indicator_gene")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "StrategyGene",
            "IndicatorGene",
            "Condition",
            "ConditionGroup",
            "StatefulCondition",
            "StateTracker",
            "EntryDirection",
            "TPSLGene",
            "PositionSizingGene",
            "TPSLResult",
            "EntryGene",
            "ToolGene",
            "PositionSizingMethod",
            "TPSLMethod",
            "EntryType",
            "GeneValidator",
            "create_random_position_sizing_gene",
            "create_random_tpsl_gene",
            "create_random_entry_gene",
            "generate_random_indicators",
            "create_random_indicator_gene",
        ]

        for item in expected_items:
            assert item in genes_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(genes_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert genes_package.__doc__ is not None
        assert len(genes_package.__doc__) > 0
