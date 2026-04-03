"""ml パッケージの lazy import を検証する。"""

from __future__ import annotations

import importlib
import sys


class TestMLPackageLazyImports:
    """循環依存を避けるための lazy import を検証する。"""

    def test_orchestration_package_defers_training_service_import(self) -> None:
        """orchestration パッケージは属性アクセスまで MLTrainingService を読み込まない。"""
        sys.modules.pop("app.services.ml.orchestration", None)

        module = importlib.import_module("app.services.ml.orchestration")

        assert "MLTrainingService" not in module.__dict__

        exported = module.MLTrainingService

        assert exported.__name__ == "MLTrainingService"
        assert module.__dict__["MLTrainingService"] is exported

    def test_trainers_package_defers_base_trainer_import(self) -> None:
        """trainers パッケージは属性アクセスまで BaseMLTrainer を読み込まない。"""
        sys.modules.pop("app.services.ml.trainers", None)

        module = importlib.import_module("app.services.ml.trainers")

        assert "BaseMLTrainer" not in module.__dict__

        exported = module.BaseMLTrainer

        assert exported.__name__ == "BaseMLTrainer"
        assert module.__dict__["BaseMLTrainer"] is exported
