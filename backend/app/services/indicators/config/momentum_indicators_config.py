"""
モメンタム系インジケーターの設定
"""

from app.services.indicators.technical_indicators.momentum import (
    MomentumIndicators,
)

from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)


def setup_momentum_indicators():
    """モメンタム系インジケーターの設定"""

    # STOCH
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        adapter_function=MomentumIndicators.stoch,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
        output_names=["STOCH_0", "STOCH_1"],  # %K, %D
        default_output="STOCH_0",  # %K
        aliases=["STOCH"],
        # pandas-ta設定
        pandas_function="stoch",
        multi_column=True,
        data_columns=["High", "Low", "Close"],
        returns="multiple",
        return_cols=["STOCHk", "STOCHd"],
        default_values={"k_length": 14, "smooth_k": 3, "d_length": 3},
        min_length_func=lambda params: params.get("k_length", 14) + params.get("d_length", 3) + params.get("smooth_k", 3),
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="k_length",
            default_value=14,
            min_value=1,
            max_value=30,
            description="K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="smooth_k",
            default_value=3,
            min_value=1,
            max_value=10,
            description="K平滑化期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="d_length",
            default_value=3,
            min_value=1,
            max_value=10,
            description="D期間",
        )
    )
    stoch_config.param_map = {
        "high": "high",
        "low": "low",
        "close": "close",
        "k_length": "k",
        "smooth_k": "smooth_k",
        "d_length": "d",
    }
    indicator_registry.register(stoch_config)

    # QQE
    qqe_config = IndicatorConfig(
        indicator_name="QQE",
        adapter_function=MomentumIndicators.qqe,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
        # pandas-ta設定
        pandas_function="qqe",
        data_column="Close",
        returns="multiple",
        return_cols=["QQE", "QQE_SIGNAL"],
        default_values={"length": 14, "smooth": 5},
    )
    qqe_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="QQE計算期間",
        )
    )
    qqe_config.add_parameter(
        ParameterConfig(
            name="smooth",
            default_value=5,
            min_value=1,
            max_value=50,
            description="QQE平滑化期間",
        )
    )
    qqe_config.param_map = {"close": "data", "length": "length", "smooth": "smooth"}
    indicator_registry.register(qqe_config)

    # RSI (Relative Strength Index)
    rsi_config = IndicatorConfig(
        indicator_name="RSI",
        adapter_function=MomentumIndicators.rsi,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
        # pandas-ta設定
        pandas_function="rsi",
        data_column="Close",
        returns="single",
        default_values={"length": 14},
        min_length_func=lambda params: max(2, params.get("length", 14)),
    )
    rsi_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="RSI計算期間",
        )
    )
    rsi_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(rsi_config)

    # MACD (Moving Average Convergence Divergence)
    macd_config = IndicatorConfig(
        indicator_name="MACD",
        adapter_function=MomentumIndicators.macd,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
        output_names=["MACD_0", "MACD_1", "MACD_2"],
        default_output="MACD_0",
        # pandas-ta設定
        pandas_function="macd",
        data_column="Close",
        returns="multiple",
        return_cols=["MACD", "Signal", "Histogram"],
        default_values={"fast": 12, "slow": 26, "signal": 9},
        min_length_func=lambda params: params.get("slow", 26) + params.get("signal", 9) + 5,
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=12,
            min_value=2,
            max_value=100,
            description="高速移動平均期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=26,
            min_value=5,
            max_value=200,
            description="低速移動平均期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="signal",
            default_value=9,
            min_value=2,
            max_value=50,
            description="シグナル線期間",
        )
    )
    macd_config.param_map = {"close": "data", "fast": "fast", "slow": "slow", "signal": "signal"}
    indicator_registry.register(macd_config)

    # CCI
    cci_config = IndicatorConfig(
        indicator_name="CCI",
        adapter_function=MomentumIndicators.cci,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    cci_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=5,
            max_value=50,
            description="CCI計算期間",
        )
    )
    cci_config.param_map = {"period": "length"}
    indicator_registry.register(cci_config)

    # MOM (Momentum)
    mom_config = IndicatorConfig(
        indicator_name="MOM",
        adapter_function=MomentumIndicators.mom,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    mom_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=10,
            min_value=2,
            max_value=50,
            description="モメンタム計算期間",
        )
    )
    mom_config.param_map = {"close": "data", "period": "length"}
    mom_config.aliases = ["MOMENTUM"]
    indicator_registry.register(mom_config)

    # ADX
    adx_config = IndicatorConfig(
        indicator_name="ADX",
        adapter_function=MomentumIndicators.adx,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    adx_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="ADX計算期間",
        )
    )
    adx_config.param_map = {"period": "length"}
    indicator_registry.register(adx_config)

    # WILLR
    willr_config = IndicatorConfig(
        indicator_name="WILLR",
        adapter_function=MomentumIndicators.willr,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    willr_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Williams %R計算期間",
        )
    )
    willr_config.param_map = {"period": "length"}
    indicator_registry.register(willr_config)

    # ROC
    roc_config = IndicatorConfig(
        indicator_name="ROC",
        adapter_function=MomentumIndicators.roc,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    roc_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROC計算期間",
        )
    )
    roc_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(roc_config)

    # SQUEEZE
    squeeze_config = IndicatorConfig(
        indicator_name="SQUEEZE",
        adapter_function=MomentumIndicators.squeeze,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
        output_names=["SQZ"],  # Will return the primary SQZ column
        default_output="SQZ",
        aliases=["SQUEEZE"],
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="bb_length",
            default_value=20,
            min_value=5,
            max_value=100,
            description="Bollinger Bands length",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="bb_std",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
            description="Bollinger Bands standard deviation",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="kc_length",
            default_value=20,
            min_value=5,
            max_value=100,
            description="Keltner Channels length",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="kc_scalar",
            default_value=1.5,
            min_value=0.1,
            max_value=5.0,
            description="Keltner Channels scalar",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="mom_length",
            default_value=12,
            min_value=2,
            max_value=50,
            description="Momentum length",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="mom_smooth",
            default_value=6,
            min_value=1,
            max_value=20,
            description="Momentum smoothing",
        )
    )
    squeeze_config.add_parameter(
        ParameterConfig(
            name="use_tr",
            default_value=True,
            description="Use True Range for calculations",
        )
    )
    squeeze_config.param_map = {
        "bb_length": "bb_length",
        "bb_std": "bb_std",
        "kc_length": "kc_length",
        "kc_scalar": "kc_scalar",
        "mom_length": "mom_length",
        "mom_smooth": "mom_smooth",
        "use_tr": "use_tr",
    }
    indicator_registry.register(squeeze_config)