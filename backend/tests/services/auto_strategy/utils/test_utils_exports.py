import importlib


def test_utils_lazy_exports():
    module = importlib.import_module("app.services.auto_strategy.utils")

    assert module.NormalizationUtils is not None
    assert module.create_default_strategy_gene is not None
    assert module.OperandGroup is not None
    assert module.OperandGroupingSystem is not None
    assert module.operand_grouping_system is not None
