"""ml パッケージの lazy import を検証する。"""

from __future__ import annotations

import importlib
import sys


class TestMLPackageLazyImports:
    """循環依存を避けるための lazy import を検証する。"""

    def test_ml_package_defers_training_service_import_and_caches_export(self) -> None:
        """ml パッケージは属性アクセス後に MLTrainingService をキャッシュする。"""
        sys.modules.pop("app.services.ml", None)

        module = importlib.import_module("app.services.ml")

        assert "MLTrainingService" not in module.__dict__

        exported = module.MLTrainingService

        assert exported.__name__ == "MLTrainingService"
        assert module.__dict__["MLTrainingService"] is exported

    def test_orchestration_package_defers_training_service_import(self) -> None:
        """orchestration パッケージは属性アクセスまで MLTrainingService を読み込まない。"""
        sys.modules.pop("app.services.ml.orchestration", None)

        module = importlib.import_module("app.services.ml.orchestration")

        assert "MLTrainingService" not in module.__dict__

        exported = module.MLTrainingService

        assert exported.__name__ == "MLTrainingService"
        assert module.__dict__["MLTrainingService"] is exported

    def test_orchestration_package_defers_management_service_import(self) -> None:
        """orchestration パッケージは属性アクセスまで MLManagementOrchestrationService を読み込まない。"""
        sys.modules.pop("app.services.ml.orchestration", None)

        module = importlib.import_module("app.services.ml.orchestration")

        assert "MLManagementOrchestrationService" not in module.__dict__

        exported = module.MLManagementOrchestrationService

        assert exported.__name__ == "MLManagementOrchestrationService"
        assert module.__dict__["MLManagementOrchestrationService"] is exported

    def test_trainers_package_defers_base_trainer_import(self) -> None:
        """trainers パッケージは属性アクセスまで BaseMLTrainer を読み込まない。"""
        sys.modules.pop("app.services.ml.trainers", None)

        module = importlib.import_module("app.services.ml.trainers")

        assert "BaseMLTrainer" not in module.__dict__

        exported = module.BaseMLTrainer

        assert exported.__name__ == "BaseMLTrainer"
        assert module.__dict__["BaseMLTrainer"] is exported

    def test_trainers_package_defers_volatility_trainer_import(self) -> None:
        """trainers パッケージは属性アクセスまで VolatilityRegressionTrainer を読み込まない。"""
        sys.modules.pop("app.services.ml.trainers", None)

        module = importlib.import_module("app.services.ml.trainers")

        assert "VolatilityRegressionTrainer" not in module.__dict__

        exported = module.VolatilityRegressionTrainer

        assert exported.__name__ == "VolatilityRegressionTrainer"
        assert module.__dict__["VolatilityRegressionTrainer"] is exported
