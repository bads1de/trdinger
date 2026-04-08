"""
positionsパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.positions as positions_package


class TestAutoStrategyPositionsInitExports:
    """positions/__init__.pyのエクスポートテスト"""

    def test_position_sizing_service_exported(self):
        """PositionSizingServiceがエクスポートされている"""
        assert hasattr(positions_package, "PositionSizingService")

    def test_position_sizing_result_exported(self):
        """PositionSizingResultがエクスポートされている"""
        assert hasattr(positions_package, "PositionSizingResult")

    def test_entry_executor_exported(self):
        """EntryExecutorがエクスポートされている"""
        assert hasattr(positions_package, "EntryExecutor")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "PositionSizingService",
            "PositionSizingResult",
            "EntryExecutor",
        ]

        for item in expected_items:
            assert item in positions_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(positions_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert positions_package.__doc__ is not None
        assert len(positions_package.__doc__) > 0
