import importlib


def test_utils_lazy_exports():
    module = importlib.import_module("app.services.auto_strategy.utils")

    assert module.NormalizationUtils is not None


def test_removed_symbols_not_exported():
    module = importlib.import_module("app.services.auto_strategy.utils")

    assert not hasattr(module, "create_default_strategy_gene")
    assert not hasattr(module, "normalize_parameter")
    assert not hasattr(module, "OperandGroupingSystem")
    assert not hasattr(module, "operand_grouping_system")
