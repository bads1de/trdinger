"""
トレンド系インジケーターの設定
"""

from app.services.indicators.technical_indicators.trend import TrendIndicators

from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)


def setup_trend_indicators():
    """トレンド系インジケーターの設定"""

    # SMA (Simple Moving Average)
    sma_config = IndicatorConfig(
        indicator_name="SMA",
        adapter_function=TrendIndicators.sma,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="sma",
        data_column="Close",
        returns="single",
        default_values={"length": 20},
        min_length_func=lambda params: max(2, params.get("length", 20)),
    )
    sma_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=200,
            description="SMA計算期間",
        )
    )
    sma_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(sma_config)

    # EMA (Exponential Moving Average)
    ema_config = IndicatorConfig(
        indicator_name="EMA",
        adapter_function=TrendIndicators.ema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="ema",
        data_column="Close",
        returns="single",
        default_values={"length": 20},
        min_length_func=lambda params: max(2, params.get("length", 20)),
    )
    ema_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=200,
            description="EMA計算期間",
        )
    )
    ema_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(ema_config)

    # WMA (Weighted Moving Average)
    wma_config = IndicatorConfig(
        indicator_name="WMA",
        adapter_function=TrendIndicators.wma,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="wma",
        data_column="Close",
        returns="single",
        default_values={"length": 20},
        min_length_func=lambda params: max(2, params.get("length", 20)),
    )
    wma_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=200,
            description="加重移動平均期間",
        )
    )
    wma_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(wma_config)

    # DEMA
    dema_config = IndicatorConfig(
        indicator_name="DEMA",
        adapter_function=TrendIndicators.dema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="dema",
        data_column="Close",
        returns="single",
        default_values={"length": 14},
    )
    dema_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="二重指数移動平均期間",
        )
    )
    dema_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(dema_config)

    # TEMA
    tema_config = IndicatorConfig(
        indicator_name="TEMA",
        adapter_function=TrendIndicators.tema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="tema",
        data_column="Close",
        returns="single",
        default_values={"length": 14},
        min_length_func=lambda params: max(3, params.get("length", 14) // 2),
    )
    tema_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="三重指数移動平均期間",
        )
    )
    tema_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(tema_config)

    # T3
    t3_config = IndicatorConfig(
        indicator_name="T3",
        adapter_function=TrendIndicators.t3,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="t3",
        data_column="Close",
        returns="single",
        default_values={"length": 5, "a": 0.7},
    )
    t3_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=5,
            min_value=2,
            max_value=50,
            description="T3移動平均期間",
        )
    )
    t3_config.add_parameter(
        ParameterConfig(
            name="a",
            default_value=0.7,
            min_value=0.1,
            max_value=1.0,
            description="T3スムージングファクター",
        )
    )
    # Add vfactor parameter mapping to a (for pandas-ta compatibility)
    t3_config.add_parameter(
        ParameterConfig(
            name="vfactor",
            default_value=0.7,
            min_value=0.1,
            max_value=1.0,
            description="V-Factor for pandas-ta compatibility (maps to a parameter)",
        )
    )
    t3_config.param_map = {
        "close": "data",
        "length": "length",
        "a": "a",
        "vfactor": "a",  # Map vfactor to a parameter for pandas-ta compatibility
    }
    indicator_registry.register(t3_config)

    # KAMA
    kama_config = IndicatorConfig(
        indicator_name="KAMA",
        adapter_function=TrendIndicators.kama,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
        # pandas-ta設定
        pandas_function="kama",
        data_column="Close",
        returns="single",
        default_values={"length": 30},
    )
    kama_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=30,
            min_value=2,
            max_value=200,
            description="KAMA期間",
        )
    )
    kama_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(kama_config)

    # SAR
    sar_config = IndicatorConfig(
        indicator_name="SAR",
        adapter_function=TrendIndicators.sar,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
        # pandas-ta設定
        pandas_function="psar",
        multi_column=True,
        data_columns=["High", "Low"],
        returns="single",
        default_values={"af": 0.02, "max_af": 0.2},
    )
    sar_config.add_parameter(
        ParameterConfig(
            name="af",
            default_value=0.02,
            min_value=0.01,
            max_value=0.1,
            description="加速因子",
        )
    )
    sar_config.add_parameter(
        ParameterConfig(
            name="max_af",
            default_value=0.2,
            min_value=0.1,
            max_value=1.0,
            description="最大加速因子",
        )
    )
    sar_config.param_map = {"high": "high", "low": "low", "af": "af", "max_af": "max_af"}
    indicator_registry.register(sar_config)

