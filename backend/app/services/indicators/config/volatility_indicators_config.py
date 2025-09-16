"""
ボラティリティ系インジケーターの設定
"""

from app.services.indicators.technical_indicators.volatility import (
    VolatilityIndicators,
)

from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)


def setup_volatility_indicators():
    """ボラティリティ系インジケーターの設定"""

    # ATR
    atr_config = IndicatorConfig(
        indicator_name="ATR",
        adapter_function=VolatilityIndicators.atr,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    atr_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="ATR計算期間",
        )
    )
    atr_config.param_map = {"period": "length"}
    indicator_registry.register(atr_config)

    # BB (Bollinger Bands)
    bb_config = IndicatorConfig(
        indicator_name="BB",
        adapter_function=VolatilityIndicators.bbands,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
        output_names=["BB_Upper", "BB_Middle", "BB_Lower"],
        default_output="BB_Middle",
    )
    bb_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=20,
            min_value=5,
            max_value=100,
            description="ボリンジャーバンド期間",
        )
    )
    bb_config.add_parameter(
        ParameterConfig(
            name="std",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
            description="標準偏差倍数",
        )
    )
    bb_config.param_map = {"close": "data", "period": "length", "std": "std"}
    indicator_registry.register(bb_config)

    # ACCBANDS
    accbands_config = IndicatorConfig(
        indicator_name="ACCBANDS",
        adapter_function=VolatilityIndicators.accbands,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
        output_names=["ACCBANDS_Upper", "ACCBANDS_Middle", "ACCBANDS_Lower"],
        default_output="ACCBANDS_Middle",
    )
    accbands_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=100,
            description="ACCBANDS計算期間",
        )
    )
    accbands_config.param_map = {"length": "length"}
    indicator_registry.register(accbands_config)

    # UI
    ui_config = IndicatorConfig(
        indicator_name="UI",
        adapter_function=VolatilityIndicators.ui,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="volatility",
    )
    ui_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="UI計算期間",
        )
    )
    ui_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(ui_config)

    kc_config = IndicatorConfig(
        indicator_name="KELTNER",
        adapter_function=VolatilityIndicators.keltner,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    kc_config.add_parameter(
        ParameterConfig(name="length", default_value=20, min_value=2, max_value=200)
    )
    kc_config.add_parameter(
        ParameterConfig(name="scalar", default_value=2.0, min_value=0.5, max_value=5.0)
    )
    indicator_registry.register(kc_config)

    donch_config = IndicatorConfig(
        indicator_name="DONCHIAN",
        adapter_function=VolatilityIndicators.donchian,
        required_data=["high", "low"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    donch_config.add_parameter(
        ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
    )
    donch_config.param_map = {"period": "length"}
    indicator_registry.register(donch_config)

    supertrend_config = IndicatorConfig(
        indicator_name="SUPERTREND",
        adapter_function=VolatilityIndicators.supertrend,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    supertrend_config.add_parameter(
        ParameterConfig(name="period", default_value=10, min_value=2, max_value=200)
    )
    supertrend_config.add_parameter(
        ParameterConfig(
            name="multiplier", default_value=3.0, min_value=1.0, max_value=10.0
        )
    )
    supertrend_config.param_map = {"period": "length"}
    indicator_registry.register(supertrend_config)