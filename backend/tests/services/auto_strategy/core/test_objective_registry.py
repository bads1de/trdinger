from app.services.auto_strategy.core import objective_registry


def test_minimize_objectives_are_centralized():
    assert objective_registry.MINIMIZE_OBJECTIVES == frozenset(
        {"max_drawdown", "ulcer_index", "trade_frequency_penalty"}
    )
    assert objective_registry.DYNAMIC_SCALAR_OBJECTIVES == frozenset(
        {"max_drawdown", "ulcer_index", "trade_frequency_penalty"}
    )


def test_is_minimize_objective_defaults_to_maximize():
    assert objective_registry.is_minimize_objective("max_drawdown") is True
    assert objective_registry.is_minimize_objective("ulcer_index") is True
    assert objective_registry.is_minimize_objective("trade_frequency_penalty") is True
    assert objective_registry.is_minimize_objective("total_return") is False
    assert objective_registry.is_minimize_objective("unknown_objective") is False


def test_to_selection_space_uses_registry(monkeypatch):
    monkeypatch.setattr(
        objective_registry,
        "is_minimize_objective",
        lambda objective: objective == "custom_loss",
    )

    assert objective_registry.to_selection_space(1.25, "custom_loss") == -1.25
    assert objective_registry.to_selection_space(1.25, "total_return") == 1.25
