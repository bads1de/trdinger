from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator


def test_smart_condition_generator_fallback_returns_tuple():
    gen = SmartConditionGenerator(enable_smart_generation=True)
    longs, shorts, exits = gen._generate_fallback_conditions()

    assert isinstance(longs, list)
    assert isinstance(shorts, list)
    assert isinstance(exits, list)
    assert longs and shorts, "フォールバックでもロング/ショート条件は存在するべき"

