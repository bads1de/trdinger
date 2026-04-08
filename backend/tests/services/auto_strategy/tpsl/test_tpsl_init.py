"""
tpslパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.tpsl as tpsl_package


class TestAutoStrategyTPSLInitExports:
    """tpsl/__init__.pyのエクスポートテスト"""

    def test_tpsl_service_exported(self):
        """TPSLServiceがエクスポートされている"""
        assert hasattr(tpsl_package, "TPSLService")

    def test_tpsl_gene_exported(self):
        """TPSLGeneがエクスポートされている"""
        assert hasattr(tpsl_package, "TPSLGene")

    def test_tpsl_method_exported(self):
        """TPSLMethodがエクスポートされている"""
        assert hasattr(tpsl_package, "TPSLMethod")

    def test_tpsl_result_exported(self):
        """TPSLResultがエクスポートされている"""
        assert hasattr(tpsl_package, "TPSLResult")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "TPSLService",
            "TPSLGene",
            "TPSLMethod",
            "TPSLResult",
        ]

        for item in expected_items:
            assert item in tpsl_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(tpsl_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert tpsl_package.__doc__ is not None
        assert len(tpsl_package.__doc__) > 0
