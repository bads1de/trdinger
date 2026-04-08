"""
mlパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__, __dir__）をテストします。
"""

import pytest

import app.services.ml as ml_package


class TestMLInitExports:
    """ml/__init__.pyのエクスポートテスト"""

    def test_ml_training_service_lazy_load(self):
        """MLTrainingServiceが遅延ロードされる"""
        from app.services.ml.orchestration.ml_training_orchestration_service import (
            MLTrainingService,
        )

        service = getattr(ml_package, "MLTrainingService")

        assert service is MLTrainingService

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = ml_package.NonExistentAttribute

    def test_dir_includes_exports(self):
        """__dir__にエクスポートが含まれる"""
        dir_result = dir(ml_package)

        assert "MLTrainingService" in dir_result

    def test_dir_returns_list(self):
        """__dir__がリストを返す"""
        dir_result = dir(ml_package)

        assert isinstance(dir_result, list)
        assert len(dir_result) > 0

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = ["MLTrainingService"]

        for item in expected_items:
            assert item in ml_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(ml_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert ml_package.__doc__ is not None
        assert len(ml_package.__doc__) > 0
