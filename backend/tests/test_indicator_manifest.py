from app.services.indicators.config.indicator_config import indicator_registry


def test_manifest_registration_creates_rsi_config():
    from app.services.indicators import manifest

    indicator_registry.reset()
    manifest.register_indicator_manifest()

    config = indicator_registry.get_indicator_config("RSI")
    assert config is not None
    assert config.pandas_function == "rsi"
    assert config.scale_type.value == "oscillator_0_100"


def test_manifest_yaml_export_contains_thresholds():
    from app.services.indicators import manifest

    yaml_data = manifest.manifest_to_yaml_dict()

    rsi = yaml_data["indicators"]["RSI"]
    assert rsi["thresholds"]["normal"]["long_gt"] == 75
    assert rsi["conditions"]["long"] == "{left_operand} > {threshold}"
