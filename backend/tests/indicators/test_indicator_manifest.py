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


def test_manifest_registers_sixty_indicators_with_new_entries():
    from app.services.indicators import manifest

    indicator_registry.reset()
    manifest.register_indicator_manifest()

    manifest_keys = set(manifest.MANIFEST.keys())
    # 新しいインジケーターを追加したため、数が増えている
    assert len(manifest_keys) >= 60
    for indicator in {"APO", "LINREG", "NATR", "KVO", "GRI", "PRIME_OSC", "FIBO_CYCLE"}:
        assert indicator in manifest_keys


def test_indicator_settings_and_service_support_new_indicators():
    import numpy as np
    import pandas as pd

    from app.services.auto_strategy.config.indicators import IndicatorSettings
    from app.services.indicators import TechnicalIndicatorService

    indicator_registry.reset()
    from app.services.indicators import manifest

    manifest.register_indicator_manifest()

    settings = IndicatorSettings()
    for indicator in ["APO", "LINREG", "NATR", "KVO"]:
        assert indicator in settings.get_all_indicators()

    service = TechnicalIndicatorService()
    rows = 200  # KVOの計算に必要な十分なデータ長
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 150, rows),
            "high": np.linspace(101, 151, rows),
            "low": np.linspace(99, 149, rows),
            "close": np.linspace(100, 150, rows) + np.sin(np.linspace(0, 6, rows)),
            "volume": np.linspace(1_000, 2_000, rows),
        }
    )

    apo_result = service.calculate_indicator(df, "APO", {})
    assert len(apo_result) == rows

    linreg_result = service.calculate_indicator(df, "LINREG", {})
    assert len(linreg_result) == rows

    natr_result = service.calculate_indicator(df, "NATR", {})
    assert len(natr_result) == rows

    kvo_result = service.calculate_indicator(df, "KVO", {})
    # KVOはタプル（kvo_line, signal_line）を返すため、タプルであることを確認
    assert isinstance(kvo_result, tuple)
    assert len(kvo_result) == 2
    # 各要素が適切な長さであることを確認
    kvo_line, signal_line = kvo_result
    assert len(kvo_line) >= rows - 50
    assert len(signal_line) >= rows - 50
