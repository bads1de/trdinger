"""
base_resource_manager モジュールのユニットテスト
"""

import pytest

from app.services.ml.common.base_resource_manager import (
    BaseResourceManager,
    CleanupLevel,
)


class ConcreteResourceManager(BaseResourceManager):
    def _cleanup_temporary_files(self, level: CleanupLevel):
        pass

    def _cleanup_cache(self, level: CleanupLevel):
        pass

    def _cleanup_models(self, level: CleanupLevel):
        pass


class FailingResourceManager(BaseResourceManager):
    def _cleanup_temporary_files(self, level: CleanupLevel):
        raise RuntimeError("temp files error")

    def _cleanup_cache(self, level: CleanupLevel):
        raise RuntimeError("cache error")

    def _cleanup_models(self, level: CleanupLevel):
        raise RuntimeError("models error")


class TestCleanupLevel:
    def test_standard(self):
        assert CleanupLevel.STANDARD.value == "standard"

    def test_thorough(self):
        assert CleanupLevel.THOROUGH.value == "thorough"


class TestBaseResourceManager:
    def test_initialization(self):
        mgr = ConcreteResourceManager()
        assert mgr._cleanup_level == CleanupLevel.STANDARD
        assert mgr._is_cleaned_up is False

    def test_cleanup_resources(self):
        mgr = ConcreteResourceManager()
        stats = mgr.cleanup_resources()

        assert stats["level"] == "standard"
        assert "memory_before" in stats
        assert "memory_after" in stats
        assert stats["errors"] == []
        assert "temporary_files" in stats["cleaned"]
        assert "cache" in stats["cleaned"]
        assert "models" in stats["cleaned"]
        assert mgr._is_cleaned_up is True

    def test_cleanup_already_done(self):
        mgr = ConcreteResourceManager()
        mgr.cleanup_resources()
        result = mgr.cleanup_resources()
        assert result["status"] == "already_cleaned"

    def test_cleanup_with_thorough_level(self):
        mgr = ConcreteResourceManager()
        stats = mgr.cleanup_resources(level=CleanupLevel.THOROUGH)
        assert stats["level"] == "thorough"

    def test_cleanup_reports_errors(self):
        mgr = FailingResourceManager()
        stats = mgr.cleanup_resources()
        assert len(stats["errors"]) == 3
        assert mgr._is_cleaned_up is True

    def test_abstract_methods_required(self):
        with pytest.raises(TypeError):
            BaseResourceManager()
