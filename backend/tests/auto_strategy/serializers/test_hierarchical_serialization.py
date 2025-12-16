from app.services.auto_strategy.genes.conditions import Condition, ConditionGroup
from app.services.auto_strategy.serializers.serialization import DictConverter


def test_hierarchical_serialization():
    converter = DictConverter(enable_smart_generation=False)

    cond_a = Condition(left_operand="A", operator=">", right_operand=10)
    cond_b = Condition(left_operand="B", operator="<", right_operand=5)
    cond_c = Condition(left_operand="C", operator="==", right_operand=100)

    # (A AND B) OR C
    group_and = ConditionGroup(operator="AND", conditions=[cond_a, cond_b])
    group_or = ConditionGroup(operator="OR", conditions=[group_and, cond_c])

    # Serialize
    data = converter.condition_or_group_to_dict(group_or)

    # Check JSON structure
    assert data["type"] == "GROUP"
    assert data["operator"] == "OR"
    assert len(data["conditions"]) == 2

    # Verify first child is GROUP (AND)
    child1 = data["conditions"][0]
    assert child1["type"] == "GROUP"
    assert child1["operator"] == "AND"
    assert len(child1["conditions"]) == 2

    # Deserialize
    # We need to simulate the context where this is used (parse_condition_or_group is internal to dict_to_strategy_gene)
    # But we can expose or access the logic.
    # Actually checking internal logic is hard.
    # Let's use dict_to_strategy_gene but that requires a full strategy dict.

    # Alternatively, create a public deserialization helper in DictConverter?
    # Or just test that we update valid code.
    # For TDD on this unit, I'd prefer if `dict_converter` had public method for condition tree.
    # It doesn't. `dict_to_condition` only does single condition.

    # I will rely on `dict_to_strategy_gene` (integration test style) or modify code first.
    pass
