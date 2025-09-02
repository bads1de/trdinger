"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

from app.services.indicators.technical_indicators.momentum import (
    MomentumIndicators,
)
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import (
    VolatilityIndicators,
)
from app.services.indicators.technical_indicators.volume import VolumeIndicators


from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)


def setup_momentum_indicators():
    """モメンタム系インジケーターの設定"""

    # MAD (Mean Absolute Deviation)
    mad_config = IndicatorConfig(
        indicator_name="MAD",
        adapter_function=MomentumIndicators.mad,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    mad_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="MAD計算期間",
        )
    )
    mad_config.param_map = {"close": "data", "period": "period"}
    indicator_registry.register(mad_config)

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
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="fastk_period",
            default_value=5,
            min_value=1,
            max_value=30,
            description="Fast %K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="d_length",
            default_value=3,
            min_value=1,
            max_value=10,
            description="Slow K期間 (smooth_kとしても使用)",
        )
    )
    stoch_config.param_map = {
        "fastk_period": "k",
        "d_length": "smooth_k",
        "slowd_period": "d",
    }
    indicator_registry.register(stoch_config)

    # AO
    ao_config = IndicatorConfig(
        indicator_name="AO",
        adapter_function=MomentumIndicators.ao,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    indicator_registry.register(ao_config)

    # KDJ
    kdj_config = IndicatorConfig(
        indicator_name="KDJ",
        adapter_function=MomentumIndicators.kdj,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
        output_names=["KDJ_K", "KDJ_D", "KDJ_J"],
        default_output="KDJ_K",
    )
    kdj_config.add_parameter(
        ParameterConfig(
            name="k",
            default_value=14,
            min_value=2,
            max_value=100,
            description="K値計算期間",
        )
    )
    kdj_config.add_parameter(
        ParameterConfig(
            name="d",
            default_value=3,
            min_value=1,
            max_value=50,
            description="D値計算期間",
        )
    )
    kdj_config.param_map = {
        "high": "high",
        "low": "low",
        "close": "close",
        "k": "k",
        "d": "d",
    }
    indicator_registry.register(kdj_config)

    # RVGI
    rvi_config = IndicatorConfig(
        indicator_name="RVI",
        adapter_function=MomentumIndicators.rvi,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    rvi_config.add_parameter(
        ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
    )
    rvi_config.param_map = {"open": "open_", "length": "length"}
    indicator_registry.register(rvi_config)

    # QQE
    qqe_config = IndicatorConfig(
        indicator_name="QQE",
        adapter_function=MomentumIndicators.qqe,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    qqe_config.add_parameter(
        ParameterConfig(name="length", default_value=14, min_value=2, max_value=200)
    )
    indicator_registry.register(qqe_config)

    # SMI
    smi_config = IndicatorConfig(
        indicator_name="SMI",
        adapter_function=MomentumIndicators.smi,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    smi_config.add_parameter(
        ParameterConfig(name="fast", default_value=13, min_value=2, max_value=50)
    )
    smi_config.add_parameter(
        ParameterConfig(name="slow", default_value=25, min_value=3, max_value=100)
    )
    smi_config.add_parameter(
        ParameterConfig(name="signal", default_value=2, min_value=1, max_value=20)
    )
    indicator_registry.register(smi_config)

    # KST
    kst_config = IndicatorConfig(
        indicator_name="KST",
        adapter_function=MomentumIndicators.kst,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
        output_names=["KST_0", "KST_1"],
        default_output="KST_0",
    )
    kst_config.add_parameter(
        ParameterConfig(
            name="roc1",
            default_value=10,
            min_value=2,
            max_value=100,
            description="ROC1期間",
        )
    )
    kst_config.add_parameter(
        ParameterConfig(
            name="roc2",
            default_value=15,
            min_value=2,
            max_value=100,
            description="ROC2期間",
        )
    )
    kst_config.add_parameter(
        ParameterConfig(
            name="roc3",
            default_value=20,
            min_value=2,
            max_value=100,
            description="ROC3期間",
        )
    )
    kst_config.add_parameter(
        ParameterConfig(
            name="roc4",
            default_value=30,
            min_value=2,
            max_value=100,
            description="ROC4期間",
        )
    )
    kst_config.param_map = {
        "close": "data",
        "roc1": "roc1",
        "roc2": "roc2",
        "roc3": "roc3",
        "roc4": "roc4",
    }
    indicator_registry.register(kst_config)


    # RSI (Relative Strength Index)
    rsi_config = IndicatorConfig(
        indicator_name="RSI",
        adapter_function=MomentumIndicators.rsi,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
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
    rsi_config.param_map = {"data": "data"}
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
    macd_config.param_map = {"data": "data"}
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
    # CMO
    cmo_config = IndicatorConfig(
        indicator_name="CMO",
        adapter_function=MomentumIndicators.cmo,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    cmo_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=5,
            max_value=50,
            description="CMO計算期間",
        )
    )
    cmo_config.param_map = {"close": "data", "period": "length"}
    indicator_registry.register(cmo_config)

    # UO (Ultimate Oscillator)
    uo_config = IndicatorConfig(
        indicator_name="UO",
        adapter_function=MomentumIndicators.uo,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    uo_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=7,
            min_value=2,
            max_value=20,
            description="UO高速期間",
        )
    )
    uo_config.add_parameter(
        ParameterConfig(
            name="medium",
            default_value=14,
            min_value=5,
            max_value=50,
            description="UO中速期間",
        )
    )
    uo_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=28,
            min_value=10,
            max_value=100,
            description="UO低速期間",
        )
    )
    indicator_registry.register(uo_config)

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

    # MFI
    mfi_config = IndicatorConfig(
        indicator_name="MFI",
        adapter_function=VolumeIndicators.mfi,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="volume",
    )
    mfi_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="MFI計算期間",
        )
    )
    mfi_config.param_map = {"period": "length"}
    indicator_registry.register(mfi_config)

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

    # AROON
    aroon_config = IndicatorConfig(
        indicator_name="AROON",
        adapter_function=MomentumIndicators.aroon,
        required_data=["high", "low"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    aroon_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Aroon計算期間",
        )
    )
    aroon_config.param_map = {"period": "length"}
    indicator_registry.register(aroon_config)

    # AROONOSC
    aroonosc_config = IndicatorConfig(
        indicator_name="AROONOSC",
        adapter_function=MomentumIndicators.aroonosc,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    aroonosc_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="AroonOsc計算期間",
        )
    )
    aroonosc_config.param_map = {"period": "length"}
    indicator_registry.register(aroonosc_config)

    # DX
    dx_config = IndicatorConfig(
        indicator_name="DX",
        adapter_function=MomentumIndicators.dx,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    dx_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="DX計算期間",
        )
    )
    indicator_registry.register(dx_config)

    # PLUS_DI
    plus_di_config = IndicatorConfig(
        indicator_name="PLUS_DI",
        adapter_function=MomentumIndicators.plus_di,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    plus_di_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Plus DI計算期間",
        )
    )
    plus_di_config.param_map = {"period": "length"}
    indicator_registry.register(plus_di_config)

    # MINUS_DI
    minus_di_config = IndicatorConfig(
        indicator_name="MINUS_DI",
        adapter_function=MomentumIndicators.minus_di,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    minus_di_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Minus DI計算期間",
        )
    )
    minus_di_config.param_map = {"period": "length"}
    indicator_registry.register(minus_di_config)


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


    # TRIX
    trix_config = IndicatorConfig(
        indicator_name="TRIX",
        adapter_function=MomentumIndicators.trix,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    trix_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=30,
            min_value=1,
            max_value=100,
            description="TRIX計算期間",
        )
    )
    indicator_registry.register(trix_config)

    # ULTOSC
    ultosc_config = IndicatorConfig(
        indicator_name="ULTOSC",
        adapter_function=MomentumIndicators.ultosc,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    ultosc_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=7,
            min_value=1,
            max_value=100,
            description="高速期間",
        )
    )
    ultosc_config.add_parameter(
        ParameterConfig(
            name="medium",
            default_value=14,
            min_value=1,
            max_value=100,
            description="中間期間",
        )
    )
    ultosc_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=28,
            min_value=1,
            max_value=100,
            description="低速期間",
        )
    )
    indicator_registry.register(ultosc_config)

    # BOP
    bop_config = IndicatorConfig(
        indicator_name="BOP",
        adapter_function=MomentumIndicators.bop,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    bop_config.param_map = {"open": "open_"}
    indicator_registry.register(bop_config)

    # APO
    apo_config = IndicatorConfig(
        indicator_name="APO",
        adapter_function=MomentumIndicators.apo,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    apo_config.add_parameter(
        ParameterConfig(
            name="fastperiod",
            default_value=12,
            min_value=2,
            max_value=100,
            description="高速期間",
        )
    )
    apo_config.add_parameter(
        ParameterConfig(
            name="slowperiod",
            default_value=26,
            min_value=2,
            max_value=200,
            description="低速期間",
        )
    )
    apo_config.add_parameter(
        ParameterConfig(
            name="matype",
            default_value=0,
            min_value=0,
            max_value=8,
            description="移動平均タイプ",
        )
    )
    indicator_registry.register(apo_config)

    # ADXR
    adxr_config = IndicatorConfig(
        indicator_name="ADXR",
        adapter_function=MomentumIndicators.adxr,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    adxr_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="ADXR計算期間",
        )
    )
    indicator_registry.register(adxr_config)


    # STOCHRSI
    stochrsi_config = IndicatorConfig(
        indicator_name="STOCHRSI",
        adapter_function=MomentumIndicators.stochrsi,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    stochrsi_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="RSI期間",
        )
    )
    stochrsi_config.add_parameter(
        ParameterConfig(
            name="fastk_period",
            default_value=5,
            min_value=1,
            max_value=100,
            description="FastK期間",
        )
    )
    stochrsi_config.add_parameter(
        ParameterConfig(
            name="fastd_period",
            default_value=3,
            min_value=1,
            max_value=100,
            description="FastD期間",
        )
    )
    stochrsi_config.add_parameter(
        ParameterConfig(
            name="fastd_matype",
            default_value=0,
            min_value=0,
            max_value=8,
            description="FastD MA型",
        )
    )
    stochrsi_config.param_map = {
        "data": "close",
        "period": "length",
        "fastk_period": "k_period",
        "fastd_period": "d_period",
        "fastd_matype": None,  # 無視するパラメータ
    }
    indicator_registry.register(stochrsi_config)

    # ROCP
    rocp_config = IndicatorConfig(
        indicator_name="ROCP",
        adapter_function=MomentumIndicators.rocp,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    rocp_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCP計算期間",
        )
    )
    rocp_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(rocp_config)

    # ROCR
    rocr_config = IndicatorConfig(
        indicator_name="ROCR",
        adapter_function=MomentumIndicators.rocr,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    rocr_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCR計算期間",
        )
    )
    rocr_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(rocr_config)

    # ROCR100
    rocr100_config = IndicatorConfig(
        indicator_name="ROCR100",
        adapter_function=MomentumIndicators.rocr100,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    rocr100_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCR100計算期間",
        )
    )
    rocr100_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(rocr100_config)

    # PLUS_DM
    plus_dm_config = IndicatorConfig(
        indicator_name="PLUS_DM",
        adapter_function=MomentumIndicators.plus_dm,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="momentum",
    )
    plus_dm_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Plus DM計算期間",
        )
    )
    indicator_registry.register(plus_dm_config)

    # MINUS_DM
    minus_dm_config = IndicatorConfig(
        indicator_name="MINUS_DM",
        adapter_function=MomentumIndicators.minus_dm,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="momentum",
    )
    minus_dm_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Minus DM計算期間",
        )
    )
    indicator_registry.register(minus_dm_config)


def setup_trend_indicators():
    """トレンド系インジケーターの設定（オートストラテジー最適化版）"""

    # SMA (Simple Moving Average) - 明示的実装
    sma_config = IndicatorConfig(
        indicator_name="SMA",
        adapter_function=TrendIndicators.sma,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
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

    # EMA (Exponential Moving Average) - 明示的実装
    ema_config = IndicatorConfig(
        indicator_name="EMA",
        adapter_function=TrendIndicators.ema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
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
    )
    wma_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
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
    )
    dema_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="二重指数移動平均期間",
        )
    )
    dema_config.param_map = {"close": "data", "period": "length"}
    indicator_registry.register(dema_config)

    # TEMA
    tema_config = IndicatorConfig(
        indicator_name="TEMA",
        adapter_function=TrendIndicators.tema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
    )
    tema_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="三重指数移動平均期間",
        )
    )
    tema_config.param_map = {"close": "data", "period": "length"}
    indicator_registry.register(tema_config)

    # T3
    t3_config = IndicatorConfig(
        indicator_name="T3",
        adapter_function=TrendIndicators.t3,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
    )
    t3_config.add_parameter(
        ParameterConfig(
            name="period",
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
    # Add vfactor parameter mapping to a (for TA-Lib compatibility)
    t3_config.add_parameter(
        ParameterConfig(
            name="vfactor",
            default_value=0.7,
            min_value=0.1,
            max_value=1.0,
            description="V-Factor for TA-Lib compatibility (maps to a parameter)",
        )
    )
    t3_config.param_map = {
        "close": "data",
        "period": "length",
        "a": "a",
        "vfactor": "a"  # Map vfactor to a parameter for pandas-ta compatibility
    }
    indicator_registry.register(t3_config)

    # TRIMA
    trima_config = IndicatorConfig(
        indicator_name="TRIMA",
        adapter_function=TrendIndicators.trima,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
    )
    trima_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="三角移動平均期間",
        )
    )
    trima_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(trima_config)

    # KAMA
    kama_config = IndicatorConfig(
        indicator_name="KAMA",
        adapter_function=TrendIndicators.kama,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
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
    indicator_registry.register(kama_config)

    # SAR
    sar_config = IndicatorConfig(
        indicator_name="SAR",
        adapter_function=TrendIndicators.sar,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
    )
    sar_config.add_parameter(
        ParameterConfig(
            name="acceleration",
            default_value=0.02,
            min_value=0.01,
            max_value=0.1,
            description="加速因子",
        )
    )
    sar_config.add_parameter(
        ParameterConfig(
            name="maximum",
            default_value=0.2,
            min_value=0.1,
            max_value=1.0,
            description="最大値",
        )
    )
    indicator_registry.register(sar_config)

    # MA
    ma_config = IndicatorConfig(
        indicator_name="MA",
        adapter_function=TrendIndicators.ma,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
    )
    ma_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=30,
            min_value=2,
            max_value=200,
            description="移動平均期間",
        )
    )
    ma_config.add_parameter(
        ParameterConfig(
            name="matype",
            default_value=0,
            min_value=0,
            max_value=8,
            description="移動平均タイプ",
        )
    )
    ma_config.param_map = {"close": "data", "period": "length", "matype": "matype"}
    indicator_registry.register(ma_config)
    
    # LINREG (Linear Regression Moving Average)
    linreg_config = IndicatorConfig(
        indicator_name="LINREG",
        adapter_function=TrendIndicators.linreg,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="trend",
    )
    linreg_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="線形回帰計算期間",
        )
    )
    linreg_config.param_map = {"close": "data", "timeperiod": "length", "length": "length", "period": "length"}
    indicator_registry.register(linreg_config)
    
    # LINREG_SLOPE (Linear Regression Slope)
    linreg_slope_config = IndicatorConfig(
        indicator_name="LINREG_SLOPE",
        adapter_function=TrendIndicators.linreg_slope,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="trend",
        aliases=["LINREGSLOPE"],
    )
    linreg_slope_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="線形回帰傾き計算期間",
        )
    )
    linreg_slope_config.param_map = {"close": "data", "length": "length", "timeperiod": "length"}
    indicator_registry.register(linreg_slope_config)
    
    # LINREG_INTERCEPT (Linear Regression Intercept)
    linreg_intercept_config = IndicatorConfig(
        indicator_name="LINREG_INTERCEPT",
        adapter_function=TrendIndicators.linreg_intercept,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
    )
    linreg_intercept_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="線形回帰切片計算期間",
        )
    )
    linreg_intercept_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(linreg_intercept_config)
    
    # LINREG_ANGLE (Linear Regression Angle)
    linreg_angle_config = IndicatorConfig(
        indicator_name="LINREG_ANGLE",
        adapter_function=TrendIndicators.linreg_angle,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="trend",
        aliases=["LINREGANGLE"],
    )
    linreg_angle_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="線形回帰角度計算期間",
        )
    )
    linreg_angle_config.add_parameter(
        ParameterConfig(
            name="degrees",
            default_value=False,
            min_value=False,
            max_value=True,
            description="度数法で角度を出力するか",
        )
    )
    linreg_angle_config.param_map = {"close": "data", "length": "length", "degrees": "degrees", "timeperiod": "length"}
    indicator_registry.register(linreg_angle_config)

    # MIDPOINT
    midpoint_config = IndicatorConfig(
        indicator_name="MIDPOINT",
        adapter_function=TrendIndicators.midpoint,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
    )
    midpoint_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="中点期間",
        )
    )
    midpoint_config.param_map = {"period": "period"}
    indicator_registry.register(midpoint_config)

    # MIDPRICE
    midprice_config = IndicatorConfig(
        indicator_name="MIDPRICE",
        adapter_function=TrendIndicators.midprice,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
    )
    midprice_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="中値価格期間",
        )
    )
    # MIN - Minimum Value Rolling Moving Average
    midprice_config.param_map = {"high": "high", "low": "low", "period": "length"}
    indicator_registry.register(midprice_config)

    min_config = IndicatorConfig(
        indicator_name="MIN",
        adapter_function=TrendIndicators.min,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="statistics",
    )
    min_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="計算期間",
        )
    )
    min_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(min_config)

    # MAX - Maximum Value Rolling Moving Average
    max_config = IndicatorConfig(
        indicator_name="MAX",
        adapter_function=TrendIndicators.max,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="statistics",
    )
    max_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="計算期間",
        )
    )
    max_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(max_config)

    # RANGE - Range (High - Low) within rolling period
    range_config = IndicatorConfig(
        indicator_name="RANGE",
        adapter_function=TrendIndicators.range_func,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    range_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=200,
            description="レンジ計算期間",
        )
    )
    range_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(range_config)
    midprice_config.param_map = {"high": "high", "low": "low", "period": "length"}
    indicator_registry.register(midprice_config)


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

    # NATR
    natr_config = IndicatorConfig(
        indicator_name="NATR",
        adapter_function=VolatilityIndicators.natr,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    natr_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="NATR計算期間",
        )
    )
    natr_config.param_map = {"period": "length"}
    indicator_registry.register(natr_config)

    # TRANGE
    trange_config = IndicatorConfig(
        indicator_name="TRANGE",
        adapter_function=VolatilityIndicators.trange,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volatility",
    )
    indicator_registry.register(trange_config)

    # BBANDS (Bollinger Bands)
    bbands_config = IndicatorConfig(
        indicator_name="BBANDS",
        adapter_function=VolatilityIndicators.bbands,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
        output_names=["BBANDS_Upper", "BBANDS_Middle", "BBANDS_Lower"],
        default_output="BBANDS_Middle",
        aliases=["BB"],
    )
    bbands_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=20,
            min_value=5,
            max_value=100,
            description="ボリンジャーバンド期間",
        )
    )
    bbands_config.add_parameter(
        ParameterConfig(
            name="std",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
            description="標準偏差倍数",
        )
    )
    bbands_config.param_map = {"close": "data", "period": "length", "std": "std"}
    indicator_registry.register(bbands_config)


    # BB - BBANDSのエイリアスとして別途登録
    bb_config = IndicatorConfig(
        indicator_name="BB",
        adapter_function=VolatilityIndicators.bbands,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
        output_names=["BBANDS_Upper", "BBANDS_Middle", "BBANDS_Lower"],
        default_output="BBANDS_Middle",
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


    # MASSI
    massi_config = IndicatorConfig(
        indicator_name="MASSI",
        adapter_function=VolatilityIndicators.massi,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="volatility",
    )
    massi_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=9,
            min_value=2,
            max_value=50,
            description="MASSI高速期間",
        )
    )
    massi_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=25,
            min_value=5,
            max_value=100,
            description="MASSI低速期間",
        )
    )
    massi_config.param_map = {"fast": "fast", "slow": "slow"}
    indicator_registry.register(massi_config)

    # PDIST
    pdist_config = IndicatorConfig(
        indicator_name="PDIST",
        adapter_function=VolatilityIndicators.pdist,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    pdist_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=21,
            min_value=2,
            max_value=100,
            description="PDIST計算期間",
        )
    )
    pdist_config.param_map = {"length": "length"}
    indicator_registry.register(pdist_config)

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

    # VAR (Variance)
    var_config = IndicatorConfig(
        indicator_name="VAR",
        adapter_function=VolatilityIndicators.variance,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    var_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Variance calculation period",
        )
    )
    var_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(var_config)

    # CV (Coefficient of Variation)
    cv_config = IndicatorConfig(
        indicator_name="CV",
        adapter_function=VolatilityIndicators.coefficient_of_variation,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    cv_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Coefficient of Variation period",
        )
    )
    cv_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(cv_config)

    # IRM (Implied Risk Measure)
    irm_config = IndicatorConfig(
        indicator_name="IRM",
        adapter_function=VolatilityIndicators.implied_risk_measure,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    irm_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Implied Risk Measure period",
        )
    )
    irm_config.param_map = {"length": "length"}
    indicator_registry.register(irm_config)


def setup_volume_indicators():
    """出来高系インジケーターの設定"""

    # OBV
    obv_config = IndicatorConfig(
        indicator_name="OBV",
        adapter_function=VolumeIndicators.obv,
        required_data=["close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.VOLUME,
        category="volume",
    )
    indicator_registry.register(obv_config)

    # AD
    ad_config = IndicatorConfig(
        indicator_name="AD",
        adapter_function=VolumeIndicators.ad,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.VOLUME,
        category="volume",
    )
    indicator_registry.register(ad_config)

    # ADOSC
    adosc_config = IndicatorConfig(
        indicator_name="ADOSC",
        adapter_function=VolumeIndicators.adosc,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="volume",
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="fastperiod",
            default_value=3,
            min_value=2,
            max_value=50,
            description="高速期間",
        )
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="slowperiod",
            default_value=10,
            min_value=2,
            max_value=100,
            description="低速期間",
        )
    )
    # パラメータマッピング: pandas-taのADOSC関数では"fast"和"slow"を使用
    adosc_config.param_map = {
        "fastperiod": "fast",
        "slowperiod": "slow",
    }
    indicator_registry.register(adosc_config)

    # VP (Volume Price Confirmation)
    vp_config = IndicatorConfig(
        indicator_name="VP",
        adapter_function=VolumeIndicators.vp,
        required_data=["close", "volume"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.VOLUME,
        category="volume",
        output_names=[
            "VP_LOW",
            "VP_MEAN",
            "VP_HIGH",
            "VP_POS_VOL",
            "VP_NEG_VOL",
            "VP_TOTAL_VOL",
        ],
        default_output="VP_TOTAL_VOL",
    )
    vp_config.add_parameter(
        ParameterConfig(
            name="width",
            default_value=10,
            min_value=2,
            max_value=50,
            description="VP価格範囲数",
        )
    )
    vp_config.param_map = {"period": "width"}
    indicator_registry.register(vp_config)

    # 統計指標の設定は削除済み


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()


# モジュール読み込み時に初期化
initialize_all_indicators()

# ---- Append new pandas-ta indicators and custom ones ----

# Trend additions
hma_config = IndicatorConfig(
    indicator_name="HMA",
    adapter_function=TrendIndicators.hma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
hma_config.add_parameter(
    ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
)
hma_config.param_map = {"period": "length"}
indicator_registry.register(hma_config)

zlma_config = IndicatorConfig(
    indicator_name="ZLMA",
    adapter_function=TrendIndicators.zlma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
zlma_config.add_parameter(
    ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
)
zlma_config.param_map = {"close": "close", "period": "length"}
indicator_registry.register(zlma_config)

vwma_config = IndicatorConfig(
    indicator_name="VWMA",
    adapter_function=TrendIndicators.vwma,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
vwma_config.add_parameter(
    ParameterConfig(name="length", default_value=20, min_value=2, max_value=200)
)
vwma_config.param_map = {"close": "data", "volume": "volume", "length": "length"}
indicator_registry.register(vwma_config)

swma_config = IndicatorConfig(
    indicator_name="SWMA",
    adapter_function=TrendIndicators.swma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
swma_config.add_parameter(
    ParameterConfig(name="period", default_value=10, min_value=2, max_value=200)
)
swma_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(swma_config)

alma_config = IndicatorConfig(
    indicator_name="ALMA",
    adapter_function=TrendIndicators.alma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
alma_config.add_parameter(
    ParameterConfig(name="period", default_value=9, min_value=2, max_value=200)
)
alma_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(alma_config)

rma_config = IndicatorConfig(
    indicator_name="RMA",
    adapter_function=TrendIndicators.rma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
rma_config.add_parameter(
    ParameterConfig(name="period", default_value=14, min_value=2, max_value=200)
)
indicator_registry.register(rma_config)

e1_config = IndicatorConfig(
    indicator_name="ICHIMOKU",
    adapter_function=TrendIndicators.ichimoku,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
)
e1_config.add_parameter(
    ParameterConfig(name="tenkan", default_value=9, min_value=2, max_value=100)
)
e1_config.add_parameter(
    ParameterConfig(name="kijun", default_value=26, min_value=2, max_value=200)
)
e1_config.add_parameter(
    ParameterConfig(name="senkou", default_value=52, min_value=2, max_value=300)
)
indicator_registry.register(e1_config)

# Volatility additions
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
    ParameterConfig(name="multiplier", default_value=3.0, min_value=1.0, max_value=10.0)
)
supertrend_config.param_map = {"period": "length"}
indicator_registry.register(supertrend_config)

# Volume additions
nvi_config = IndicatorConfig(
    indicator_name="NVI",
    adapter_function=VolumeIndicators.nvi,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
indicator_registry.register(nvi_config)

pvi_config = IndicatorConfig(
    indicator_name="PVI",
    adapter_function=VolumeIndicators.pvi,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
indicator_registry.register(pvi_config)

vwap_config = IndicatorConfig(
    indicator_name="VWAP",
    adapter_function=VolumeIndicators.vwap,
    required_data=["high", "low", "close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="volume",
)
indicator_registry.register(vwap_config)

# Momentum additions
tsi_config = IndicatorConfig(
    indicator_name="TSI",
    adapter_function=MomentumIndicators.tsi,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="momentum",
)
tsi_config.add_parameter(
    ParameterConfig(name="fastperiod", default_value=13, min_value=2, max_value=100)
)
tsi_config.add_parameter(
    ParameterConfig(name="slowperiod", default_value=25, min_value=2, max_value=200)
)
indicator_registry.register(tsi_config)

rvi_config = IndicatorConfig(
    indicator_name="RVI",
    adapter_function=MomentumIndicators.rvi,
    required_data=["open", "high", "low", "close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.OSCILLATOR_0_100,
    category="momentum",
)
rvi_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
indicator_registry.register(rvi_config)


cfo_config = IndicatorConfig(
    indicator_name="CFO",
    adapter_function=MomentumIndicators.cfo,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="momentum",
)
cfo_config.add_parameter(
    ParameterConfig(name="period", default_value=9, min_value=2, max_value=200)
)
cfo_config.param_map = {"period": "length"}
indicator_registry.register(cfo_config)

cti_config = IndicatorConfig(
    indicator_name="CTI",
    adapter_function=MomentumIndicators.cti,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="momentum",
)
cti_config.add_parameter(
    ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
)
cti_config.param_map = {"period": "length"}
indicator_registry.register(cti_config)

# Custom originals
sma_slope_config = IndicatorConfig(
    indicator_name="SMA_SLOPE",
    adapter_function=TrendIndicators.sma_slope,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="trend",
)
sma_slope_config.add_parameter(
    ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
)
sma_slope_config.param_map = {"period": "length"}
indicator_registry.register(sma_slope_config)

price_ema_ratio_config = IndicatorConfig(
    indicator_name="PRICE_EMA_RATIO",
    adapter_function=TrendIndicators.price_ema_ratio,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="trend",
)
price_ema_ratio_config.add_parameter(
    ParameterConfig(name="period", default_value=20, min_value=2, max_value=200)
)
price_ema_ratio_config.param_map = {"period": "length"}
indicator_registry.register(price_ema_ratio_config)

# MAMA configuration removed due to pandas-ta compatibility issues
# MAXINDEX and MININDEX configurations also removed due to missing implementations
#
# maxindex_config = IndicatorConfig(
#     indicator_name="MAXINDEX",
#     adapter_function=TrendIndicators.maxindex,
#     required_data=["close"],
#     result_type=IndicatorResultType.SINGLE,
#     scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
#     category="statistics",
# )
# maxindex_config.add_parameter(
#     ParameterConfig(
#         name="period",
#         default_value=14,
#         min_value=2,
#         max_value=100,
#         description="最大値インデックス期間",
#     )
# )
# maxindex_config.param_map = {"close": "data", "period": "length"}
# indicator_registry.register(maxindex_config)
#
# minindex_config = IndicatorConfig(
#     indicator_name="MININDEX",
#     adapter_function=TrendIndicators.minindex,
#     required_data=["close"],
#     result_type=IndicatorResultType.SINGLE,
#     scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
#     category="statistics",
# )
# minindex_config.add_parameter(
#     ParameterConfig(
#         name="period",
#         default_value=14,
#         min_value=2,
#         max_value=100,
#         description="最小値インデックス期間",
#     )
# )
# minindex_config.param_map = {"close": "data", "period": "length"}
# indicator_registry.register(minindex_config)

# MINMAX indicators removed due to implementation issues
# These have been commented out to prevent invalid indicator errors
#
# minmax_config = IndicatorConfig(
#     indicator_name="MINMAX",
#     adapter_function=TrendIndicators.minmax,
#     required_data=["close"],
#     result_type=IndicatorResultType.COMPLEX,
#     scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
#     category="statistics",
#     output_names=["MINMAX_MIN", "MINMAX_MAX"],
#     default_output="MINMAX_MIN",
# )
# minmax_config.add_parameter(
#     ParameterConfig(
#         name="period",
#         default_value=14,
#         min_value=2,
#         max_value=100,
#         description="最小最大期間",
#     )
# )
# minmax_config.param_map = {"close": "data", "period": "length"}
# indicator_registry.register(minmax_config)
#
# minmaxindex_config = IndicatorConfig(
#     indicator_name="MINMAXINDEX",
#     adapter_function=TrendIndicators.minmaxindex,
#     required_data=["close"],
#     result_type=IndicatorResultType.COMPLEX,
#     scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
#     category="statistics",
#     output_names=["MINMAXINDEX_MIN", "MINMAXINDEX_MAX"],
#     default_output="MINMAXINDEX_MIN",
# )
# minmaxindex_config.add_parameter(
#     ParameterConfig(
#         name="period",
#         default_value=14,
#         min_value=2,
#         max_value=100,
#         description="最小最大インデックス期間",
#     )
# )
# minmaxindex_config.param_map = {"close": "data", "period": "length"}
# indicator_registry.register(minmaxindex_config)

rsi_ema_cross_config = IndicatorConfig(
    indicator_name="RSI_EMA_CROSS",
    adapter_function=MomentumIndicators.rsi_ema_cross,
    required_data=[],  # data will be passed as parameter
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.OSCILLATOR_0_100,
    category="momentum",
)
rsi_ema_cross_config.add_parameter(
    ParameterConfig(name="rsi_length", default_value=14, min_value=2, max_value=200)
)
rsi_ema_cross_config.add_parameter(
    ParameterConfig(name="ema_length", default_value=9, min_value=2, max_value=200)
)
rsi_ema_cross_config.param_map = {
    "close": "data",
    "rsi_length": "rsi_length",
    "ema_length": "ema_length",
}
indicator_registry.register(rsi_ema_cross_config)

# Additional momentum registrations
rmi_config = IndicatorConfig(
    indicator_name="RMI",
    adapter_function=MomentumIndicators.rmi,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.OSCILLATOR_0_100,
    category="momentum",
)
rmi_config.add_parameter(
    ParameterConfig(name="length", default_value=20, min_value=2, max_value=200)
)
rmi_config.add_parameter(
    ParameterConfig(name="mom", default_value=20, min_value=1, max_value=100)
)
indicator_registry.register(rmi_config)

dpo_cfg = IndicatorConfig(
    indicator_name="DPO",
    adapter_function=MomentumIndicators.dpo,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="momentum",
)
dpo_cfg.add_parameter(
    ParameterConfig(name="length", default_value=20, min_value=2, max_value=200)
)
indicator_registry.register(dpo_cfg)

chop_cfg = IndicatorConfig(
    indicator_name="CHOP",
    adapter_function=MomentumIndicators.chop,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.OSCILLATOR_0_100,
    category="momentum",
)
chop_cfg.add_parameter(
    ParameterConfig(name="length", default_value=14, min_value=2, max_value=200)
)
chop_cfg.param_map = {
    "high": "high",
    "low": "low",
    "close": "close",
    "length": "length",
}
indicator_registry.register(chop_cfg)

vi_cfg = IndicatorConfig(
    indicator_name="VORTEX",
    adapter_function=MomentumIndicators.vortex,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
    category="momentum",
)
vi_cfg.add_parameter(
    ParameterConfig(name="length", default_value=14, min_value=2, max_value=200)
)
vi_cfg.param_map = {"high": "high", "low": "low", "close": "close", "length": "length"}
indicator_registry.register(vi_cfg)

# Additional volume registrations
eom_cfg = IndicatorConfig(
    indicator_name="EOM",
    adapter_function=VolumeIndicators.eom,
    required_data=["high", "low", "close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
eom_cfg.add_parameter(
    ParameterConfig(name="length", default_value=14, min_value=2, max_value=200)
)
eom_cfg.add_parameter(
    ParameterConfig(
        name="divisor", default_value=100000000, min_value=1, max_value=1000000000
    )
)
eom_cfg.add_parameter(
    ParameterConfig(name="drift", default_value=1, min_value=1, max_value=10)
)
eom_cfg.param_map = {
    "divisor": "divisor",
    "drift": "drift",
}
indicator_registry.register(eom_cfg)


pvt_cfg = IndicatorConfig(
    indicator_name="PVT",
    adapter_function=VolumeIndicators.pvt,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
indicator_registry.register(pvt_cfg)

cmf_cfg = IndicatorConfig(
    indicator_name="CMF",
    adapter_function=VolumeIndicators.cmf,
    required_data=["high", "low", "close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
cmf_cfg.add_parameter(
    ParameterConfig(name="length", default_value=20, min_value=2, max_value=200)
)
cmf_cfg.param_map = {
    "length": "length",
}
indicator_registry.register(cmf_cfg)

# AOBV (Archer On-Balance Volume)
aobv_config = IndicatorConfig(
    indicator_name="AOBV",
    adapter_function=VolumeIndicators.aobv,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
aobv_config.add_parameter(
    ParameterConfig(name="fast", default_value=5, min_value=2, max_value=50)
)
aobv_config.add_parameter(
    ParameterConfig(name="slow", default_value=10, min_value=2, max_value=100)
)
aobv_config.add_parameter(
    ParameterConfig(name="max_lookback", default_value=2, min_value=1, max_value=10)
)
aobv_config.add_parameter(
    ParameterConfig(name="min_lookback", default_value=2, min_value=1, max_value=10)
)
aobv_config.add_parameter(ParameterConfig(name="mamode", default_value="ema"))
indicator_registry.register(aobv_config)

# EFI (Elder's Force Index)
efi_config = IndicatorConfig(
    indicator_name="EFI",
    adapter_function=VolumeIndicators.efi,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="volume",
)
efi_config.add_parameter(
    ParameterConfig(name="length", default_value=13, min_value=2, max_value=100)
)
efi_config.add_parameter(ParameterConfig(name="mamode", default_value="ema"))
efi_config.add_parameter(
    ParameterConfig(name="drift", default_value=1, min_value=1, max_value=10)
)
indicator_registry.register(efi_config)

# PVOL (Price-Volume)
pvol_config = IndicatorConfig(
    indicator_name="PVOL",
    adapter_function=VolumeIndicators.pvol,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
pvol_config.add_parameter(
    ParameterConfig(name="signed", default_value=True, min_value=False, max_value=True)
)
# PVOL指標のパラメータマッピング（lengthパラメータ拒否）
pvol_config.param_map = {
    "signed": "signed",
    # periodパラメータを無効化（lengthへ変換させない）
    "period": None,
    "length": None,
}
indicator_registry.register(pvol_config)

# PVR (Price Volume Rank)
pvr_config = IndicatorConfig(
    indicator_name="PVR",
    adapter_function=VolumeIndicators.pvr,
    required_data=["close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.OSCILLATOR_0_100,
    category="volume",
)
indicator_registry.register(pvr_config)

# Additional trend indicators
# FWMA (Fibonacci's Weighted Moving Average)
fwma_config = IndicatorConfig(
    indicator_name="FWMA",
    adapter_function=TrendIndicators.fwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
fwma_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
fwma_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(fwma_config)

# HILO (Gann High-Low Activator)
hilo_config = IndicatorConfig(
    indicator_name="HILO",
    adapter_function=TrendIndicators.hilo,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
    output_names=["HILO_0", "HILO_1", "HILO_2"],
    default_output="HILO_0",
)
hilo_config.add_parameter(
    ParameterConfig(
        name="high_length",
        default_value=13,
        min_value=2,
        max_value=100,
        description="High period length",
    )
)
hilo_config.add_parameter(
    ParameterConfig(
        name="low_length",
        default_value=21,
        min_value=2,
        max_value=100,
        description="Low period length",
    )
)
hilo_config.param_map = {
    "high_length": "high_length",
    "low_length": "low_length",
    "high": "high",
    "low": "low",
    "close": "close",
}
indicator_registry.register(hilo_config)

# HL2 (High-Low Average)
hl2_config = IndicatorConfig(
    indicator_name="HL2",
    adapter_function=TrendIndicators.hl2,
    required_data=["high", "low"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
)
indicator_registry.register(hl2_config)

# HLC3 (High-Low-Close Average)
hlc3_config = IndicatorConfig(
    indicator_name="HLC3",
    adapter_function=TrendIndicators.hlc3,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
)
indicator_registry.register(hlc3_config)

# HWMA (Holt-Winter Moving Average)
hwma_config = IndicatorConfig(
    indicator_name="HWMA",
    adapter_function=TrendIndicators.hwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
hwma_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
hwma_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(hwma_config)

# JMA (Jurik Moving Average)
jma_config = IndicatorConfig(
    indicator_name="JMA",
    adapter_function=TrendIndicators.jma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
jma_config.add_parameter(
    ParameterConfig(name="length", default_value=7, min_value=2, max_value=50)
)
jma_config.add_parameter(
    ParameterConfig(name="phase", default_value=0.0, min_value=-100.0, max_value=100.0)
)
jma_config.add_parameter(
    ParameterConfig(name="power", default_value=2.0, min_value=1.0, max_value=10.0)
)
jma_config.param_map = {
    "close": "data",
    "length": "length",
    "phase": "phase",
    "power": "power",
}
indicator_registry.register(jma_config)

# MCGD (McGinley Dynamic)
mcgd_config = IndicatorConfig(
    indicator_name="MCGD",
    adapter_function=TrendIndicators.mcgd,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
mcgd_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
mcgd_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(mcgd_config)

# OHLC4 (Open-High-Low-Close Average)
ohlc4_config = IndicatorConfig(
    indicator_name="OHLC4",
    adapter_function=TrendIndicators.ohlc4,
    required_data=["open", "high", "low", "close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
)
ohlc4_config.param_map = {
    "open": "open_",
    "high": "high",
    "low": "low",
    "close": "close",
}
indicator_registry.register(ohlc4_config)

# PWMA (Pascal's Weighted Moving Average)
pwma_config = IndicatorConfig(
    indicator_name="PWMA",
    adapter_function=TrendIndicators.pwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
pwma_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
pwma_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(pwma_config)

# SINWMA (Sine Weighted Moving Average)
sinwma_config = IndicatorConfig(
    indicator_name="SINWMA",
    adapter_function=TrendIndicators.sinwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
sinwma_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
sinwma_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(sinwma_config)

# SSF (Ehler's Super Smoother Filter)
ssf_config = IndicatorConfig(
    indicator_name="SSF",
    adapter_function=TrendIndicators.ssf,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
ssf_config.add_parameter(
    ParameterConfig(name="length", default_value=10, min_value=2, max_value=200)
)
ssf_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(ssf_config)

# VIDYA (Variable Index Dynamic Average)
vidya_config = IndicatorConfig(
    indicator_name="VIDYA",
    adapter_function=TrendIndicators.vidya,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
vidya_config.add_parameter(
    ParameterConfig(name="length", default_value=14, min_value=2, max_value=200)
)
vidya_config.add_parameter(
    ParameterConfig(name="adjust", default_value=True, min_value=False, max_value=True)
)
vidya_config.param_map = {"close": "data", "length": "length", "adjust": "adjust"}
indicator_registry.register(vidya_config)

# WCP (Weighted Closing Price)
wcp_config = IndicatorConfig(
    indicator_name="WCP",
    adapter_function=TrendIndicators.wcp,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="trend",
)
wcp_config.param_map = {"close": "data"}
indicator_registry.register(wcp_config)

# TLB (Trend Line Break)
tlb_config = IndicatorConfig(
    indicator_name="TLB",
    adapter_function=TrendIndicators.tlb,
    required_data=["high", "low", "close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="trend",
)
tlb_config.add_parameter(
    ParameterConfig(
        name="length",
        default_value=3,
        min_value=2,
        max_value=20,
        description="Trend break lookback length",
    )
)
tlb_config.param_map = {
    "high": "high",
    "low": "low",
    "close": "close",
    "length": "length"
}
indicator_registry.register(tlb_config)

# CWMA (Central Weighted Moving Average)
cwma_config = IndicatorConfig(
    indicator_name="CWMA",
    adapter_function=TrendIndicators.cwma,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
)
cwma_config.add_parameter(
    ParameterConfig(
        name="length",
        default_value=10,
        min_value=2,
        max_value=200,
        description="Central Weighted Moving Average period",
    )
)
cwma_config.param_map = {"close": "data", "length": "length"}
indicator_registry.register(cwma_config)
