import importlib


def test_utils_lazy_exports():
    module = importlib.import_module("app.services.auto_strategy.utils")

    assert module.NormalizationUtils is not None
    assert module.create_default_strategy_gene is not None


def test_operand_grouping_not_exported():
    module = importlib.import_module("app.services.auto_strategy.utils")

    assert not hasattr(module, "OperandGroupingSystem")
    assert not hasattr(module, "operand_grouping_system")
