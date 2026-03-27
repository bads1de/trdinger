import importlib

import pytest

from app.services.auto_strategy import core as auto_strategy_core


def test_core_attribute_exports_remain_available():
    assert auto_strategy_core.IndividualEvaluator is not None
    assert auto_strategy_core.GeneticAlgorithmEngine is not None
    assert auto_strategy_core.ParallelEvaluator is not None


def test_legacy_core_module_aliases_are_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.auto_strategy.core.individual_evaluator")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.auto_strategy.core.ga_engine")
