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
        # pandas-ta設定
        pandas_function="atr",
        multi_column=True,
        data_columns=["High", "Low", "Close"],
        returns="single",
        default_values={"length": 14},
    )
    atr_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="ATR計算期間",
        )
    )
    atr_config.param_map = {"high": "high", "low": "low", "close": "close", "length": "length"}
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
        # pandas-ta設定
        pandas_function="bbands",
        data_column="Close",
        returns="multiple",
        return_cols=["BBL", "BBM", "BBU"],
        default_values={"length": 20, "std": 2.0},
        min_length_func=lambda params: params.get("length", 20),
    )
    bb_config.add_parameter(
        ParameterConfig(
            name="length",
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
    bb_config.param_map = {"close": "data", "length": "length", "std": "std"}
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
        # pandas-ta設定
        pandas_function="kc",
        multi_column=True,
        data_columns=["High", "Low", "Close"],
        returns="multiple",
        return_cols=["KC_LB", "KC_MID", "KC_UB"],
        default_values={"length": 20, "multiplier": 2.0},
    )
    kc_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=100,
            description="Keltner Channels期間",
        )
    )
    kc_config.add_parameter(
        ParameterConfig(
            name="multiplier",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
            description="ATR倍数",
        )
    )
    kc_config.param_map = {"high": "high", "low": "low", "close": "close", "length": "length", "multiplier": "multiplier"}
    indicator_registry.register(kc_config)

    donch_config = IndicatorConfig(
        indicator_name="DONCHIAN",
        adapter_function=VolatilityIndicators.donchian,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
        # pandas-ta設定
        pandas_function="donchian",
        multi_column=True,
        data_columns=["High", "Low", "Close"],
        returns="multiple",
        return_cols=["DC_LB", "DC_MB", "DC_UB"],
        default_values={"length": 20},
    )
    donch_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=200,
            description="Donchian Channels期間",
        )
    )
    donch_config.param_map = {"high": "high", "low": "low", "close": "close", "length": "length"}
    indicator_registry.register(donch_config)

    supertrend_config = IndicatorConfig(
        indicator_name="SUPERTREND",
        adapter_function=VolatilityIndicators.supertrend,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
        # pandas-ta設定
        pandas_function="supertrend",
        multi_column=True,
        data_columns=["High", "Low", "Close"],
        returns="complex",
        return_cols=["ST", "D"],
        default_values={"length": 10, "multiplier": 3.0},
        min_length_func=lambda params: params.get("length", 10) + 10,
    )
    supertrend_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=10,
            min_value=2,
            max_value=200,
            description="ATR期間",
        )
    )
    supertrend_config.add_parameter(
        ParameterConfig(
            name="multiplier",
            default_value=3.0,
            min_value=1.0,
            max_value=10.0,
            description="ATR倍数",
        )
    )
    supertrend_config.param_map = {"high": "high", "low": "low", "close": "close", "length": "length", "multiplier": "multiplier"}
    indicator_registry.register(supertrend_config)