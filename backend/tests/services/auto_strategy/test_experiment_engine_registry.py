"""
ExperimentEngineRegistry のテスト
"""

from unittest.mock import Mock

from app.services.auto_strategy.services.experiment_engine_registry import (
    ExperimentEngineRegistry,
)


class TestExperimentEngineRegistry:
    """ExperimentEngineRegistry のテストクラス"""

    def setup_method(self):
        self.registry = ExperimentEngineRegistry()

    def test_register_and_get_engine(self):
        engine = Mock()

        self.registry.register("exp-001", engine)

        assert self.registry.get("exp-001") is engine

    def test_release_respects_engine_identity(self):
        original_engine = Mock()
        other_engine = Mock()
        self.registry.register("exp-001", original_engine)

        self.registry.release("exp-001", other_engine)

        assert self.registry.get("exp-001") is original_engine

    def test_clear_removes_all_engines(self):
        self.registry.register("exp-001", Mock())
        self.registry.register("exp-002", Mock())

        self.registry.clear()

        assert self.registry.get("exp-001") is None
        assert self.registry.get("exp-002") is None
