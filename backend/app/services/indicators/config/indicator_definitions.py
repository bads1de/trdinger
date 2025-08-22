"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

from app.services.indicators.technical_indicators.momentum import (
    MomentumIndicators,
)
from app.services.indicators.technical_indicators.pattern_recognition import (
    PatternRecognitionIndicators,
)
from app.services.indicators.technical_indicators.statistics import (
    StatisticsIndicators,
)
from app.services.indicators.technical_indicators.hilbert_transform import (
    HilbertTransformIndicators,
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
    """モメンタム系インジケーターの設定（オートストラテジー最適化版）"""

    # STOCH
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        adapter_function=MomentumIndicators.stoch,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="stoch_handler",
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
            name="slowk_period",
            default_value=3,
            min_value=1,
            max_value=10,
            description="Slow %K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowd_period",
            default_value=3,
            min_value=1,
            max_value=10,
            description="Slow %D期間",
        )
    )
    stoch_config.param_map = {
        "fastk_period": "k",
        "slowk_period": "smooth_k",
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
    )
    kdj_config.add_parameter(
        ParameterConfig(name="k", default_value=14, min_value=2, max_value=100)
    )
    kdj_config.add_parameter(
        ParameterConfig(name="d", default_value=3, min_value=1, max_value=50)
    )
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
        required_data=[],  # data will be passed as parameter
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="momentum",
    )
    kst_config.param_map = {"close": "data"}
    indicator_registry.register(kst_config)

    # STC
    stc_config = IndicatorConfig(
        indicator_name="STC",
        adapter_function=MomentumIndicators.stc,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    stc_config.add_parameter(
        ParameterConfig(
            name="tclength",
            default_value=10,
            min_value=2,
            max_value=100,
            description="STCシグナルライン長",
        )
    )
    stc_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=23,
            min_value=2,
            max_value=100,
            description="STC高速期間",
        )
    )
    stc_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=50,
            min_value=5,
            max_value=200,
            description="STC低速期間",
        )
    )
    stc_config.add_parameter(
        ParameterConfig(
            name="factor",
            default_value=0.5,
            min_value=0.1,
            max_value=1.0,
            description="STCスムージングファクター",
        )
    )
    stc_config.param_map = {"data": "close"}
    indicator_registry.register(stc_config)

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

    # RVGI (Relative Vigor Index)
    rvgi_config = IndicatorConfig(
        indicator_name="RVGI",
        adapter_function=MomentumIndicators.rvgi,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    rvgi_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=10,
            min_value=2,
            max_value=50,
            description="RVGI計算期間",
        )
    )
    rvgi_config.param_map = {"open": "open_", "period": "length"}
    indicator_registry.register(rvgi_config)

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
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="ADX計算期間",
        )
    )
    indicator_registry.register(adx_config)

    # MFI
    mfi_config = IndicatorConfig(
        indicator_name="MFI",
        adapter_function=MomentumIndicators.mfi,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    mfi_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="MFI計算期間",
        )
    )
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
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Williams %R計算期間",
        )
    )
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
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Aroon計算期間",
        )
    )
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
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="AroonOsc計算期間",
        )
    )
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

    # PPO
    ppo_config = IndicatorConfig(
        indicator_name="PPO",
        adapter_function=MomentumIndicators.ppo,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    ppo_config.add_parameter(
        ParameterConfig(
            name="fastperiod",
            default_value=12,
            min_value=2,
            max_value=100,
            description="高速期間",
        )
    )
    ppo_config.add_parameter(
        ParameterConfig(
            name="slowperiod",
            default_value=26,
            min_value=2,
            max_value=200,
            description="低速期間",
        )
    )
    ppo_config.add_parameter(
        ParameterConfig(
            name="matype",
            default_value=0,
            min_value=0,
            max_value=8,
            description="移動平均タイプ",
        )
    )
    ppo_config.param_map = {
        "data": "close",
        "fastperiod": "fast",
        "slowperiod": "slow",
        "matype": "signal",
    }
    indicator_registry.register(ppo_config)

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
    roc_config.param_map = {"data": "close"}
    indicator_registry.register(roc_config)

    # STOCHF
    stochf_config = IndicatorConfig(
        indicator_name="STOCHF",
        adapter_function=MomentumIndicators.stochf,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
    )
    stochf_config.add_parameter(
        ParameterConfig(
            name="fastk_period",
            default_value=5,
            min_value=1,
            max_value=100,
            description="FastK期間",
        )
    )
    stochf_config.add_parameter(
        ParameterConfig(
            name="fastd_period",
            default_value=3,
            min_value=1,
            max_value=100,
            description="FastD期間",
        )
    )
    stochf_config.param_map = {"fastk_period": "k", "fastd_period": "d"}
    indicator_registry.register(stochf_config)

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

    # MACDEXT
    macdext_config = IndicatorConfig(
        indicator_name="MACDEXT",
        adapter_function=MomentumIndicators.macdext,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=12,
            min_value=2,
            max_value=100,
            description="高速期間",
        )
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="fast_ma_type",
            default_value=0,
            min_value=0,
            max_value=8,
            description="高速MA型",
        )
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=26,
            min_value=2,
            max_value=200,
            description="低速期間",
        )
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="slow_ma_type",
            default_value=0,
            min_value=0,
            max_value=8,
            description="低速MA型",
        )
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="signal_period",
            default_value=9,
            min_value=2,
            max_value=100,
            description="シグナル期間",
        )
    )
    macdext_config.add_parameter(
        ParameterConfig(
            name="signal_ma_type",
            default_value=0,
            min_value=0,
            max_value=8,
            description="シグナルMA型",
        )
    )
    indicator_registry.register(macdext_config)

    # MACDFIX
    macdfix_config = IndicatorConfig(
        indicator_name="MACDFIX",
        adapter_function=MomentumIndicators.macdfix,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    macdfix_config.add_parameter(
        ParameterConfig(
            name="signal_period",
            default_value=9,
            min_value=2,
            max_value=100,
            description="シグナル期間",
        )
    )
    indicator_registry.register(macdfix_config)

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
    rocp_config.param_map = {"data": "close"}
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
    rocr_config.param_map = {"data": "close", "period": "length"}
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
    rocr100_config.param_map = {"data": "close"}
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
    sma_config.param_map = {"data": "data", "length": "length"}
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
    ema_config.param_map = {"data": "data", "length": "length"}
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
    wma_config.param_map = {"data": "close", "length": "length"}
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
    t3_config.param_map = {"close": "data", "period": "length", "a": "a"}
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
    trima_config.param_map = {"data": "close", "length": "length"}
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

    # HT_TRENDLINE
    ht_trendline_config = IndicatorConfig(
        indicator_name="HT_TRENDLINE",
        adapter_function=TrendIndicators.ht_trendline,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="trend",
    )
    indicator_registry.register(ht_trendline_config)

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
    ma_config.param_map = {"data": "close", "period": "length", "matype": "matype"}
    indicator_registry.register(ma_config)

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
    indicator_registry.register(adosc_config)


def setup_statistics_indicators():
    """統計系インジケーターの設定"""

    # BETA
    beta_config = IndicatorConfig(
        indicator_name="BETA",
        adapter_function=StatisticsIndicators.beta,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="statistics",
    )
    beta_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=5,
            min_value=2,
            max_value=100,
            description="BETA計算期間",
        )
    )
    beta_config.param_map = {"length": "length"}
    indicator_registry.register(beta_config)

    # CORREL
    correl_config = IndicatorConfig(
        indicator_name="CORREL",
        adapter_function=StatisticsIndicators.correl,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="statistics",
    )
    correl_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=30,
            min_value=2,
            max_value=200,
            description="相関係数計算期間",
        )
    )
    correl_config.param_map = {"data0": "high", "data1": "low", "period": "length"}
    indicator_registry.register(correl_config)

    # LINEARREG
    linearreg_config = IndicatorConfig(
        indicator_name="LINEARREG",
        adapter_function=StatisticsIndicators.linearreg,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="statistics",
    )
    linearreg_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰期間",
        )
    )
    linearreg_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(linearreg_config)

    # STDDEV
    stddev_config = IndicatorConfig(
        indicator_name="STDDEV",
        adapter_function=StatisticsIndicators.stddev,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="statistics",
    )
    stddev_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=5,
            min_value=2,
            max_value=100,
            description="標準偏差期間",
        )
    )
    stddev_config.add_parameter(
        ParameterConfig(
            name="nbdev",
            default_value=1.0,
            min_value=0.1,
            max_value=5.0,
            description="偏差数",
        )
    )
    stddev_config.param_map = {"close": "data", "length": "length", "nbdev": "nbdev"}
    indicator_registry.register(stddev_config)

    # TSF
    tsf_config = IndicatorConfig(
        indicator_name="TSF",
        adapter_function=StatisticsIndicators.tsf,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="statistics",
    )
    tsf_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="時系列予測期間",
        )
    )
    tsf_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(tsf_config)

    # VAR
    var_config = IndicatorConfig(
        indicator_name="VAR",
        adapter_function=StatisticsIndicators.var,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="statistics",
    )
    var_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=5,
            min_value=2,
            max_value=100,
            description="分散期間",
        )
    )
    var_config.add_parameter(
        ParameterConfig(
            name="nbdev",
            default_value=1.0,
            min_value=0.1,
            max_value=5.0,
            description="偏差数",
        )
    )
    var_config.param_map = {"close": "data", "nbdev": "nbdev"}
    indicator_registry.register(var_config)

    # LINEARREG_ANGLE
    linearreg_angle_config = IndicatorConfig(
        indicator_name="LINEARREG_ANGLE",
        adapter_function=StatisticsIndicators.linearreg_angle,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="statistics",
    )
    linearreg_angle_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰角度期間",
        )
    )
    linearreg_angle_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(linearreg_angle_config)

    # LINEARREG_INTERCEPT
    linearreg_intercept_config = IndicatorConfig(
        indicator_name="LINEARREG_INTERCEPT",
        adapter_function=StatisticsIndicators.linearreg_intercept,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="statistics",
    )
    linearreg_intercept_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰切片期間",
        )
    )
    linearreg_intercept_config.param_map = {"close": "data", "length": "length"}
    indicator_registry.register(linearreg_intercept_config)

    # LINEARREG_SLOPE
    linearreg_slope_config = IndicatorConfig(
        indicator_name="LINEARREG_SLOPE",
        adapter_function=StatisticsIndicators.linearreg_slope,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="statistics",
    )
    linearreg_slope_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰傾き期間",
        )
    )
    linearreg_slope_config.param_map = {"close": "data", "period": "length"}
    indicator_registry.register(linearreg_slope_config)


def setup_pattern_recognition_indicators():
    """パターン認識系インジケーターの設定"""

    # CDL_HANGING_MAN - ハンギングマン
    hanging_man_config = IndicatorConfig(
        indicator_name="CDL_HANGING_MAN",
        adapter_function=PatternRecognitionIndicators.cdl_hanging_man,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(hanging_man_config)

    # CDL_SHOOTING_STAR - 流れ星
    shooting_star_config = IndicatorConfig(
        indicator_name="CDL_SHOOTING_STAR",
        adapter_function=PatternRecognitionIndicators.cdl_shooting_star,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(shooting_star_config)

    # CDL_HARAMI - はらみ足
    harami_config = IndicatorConfig(
        indicator_name="CDL_HARAMI",
        adapter_function=PatternRecognitionIndicators.cdl_harami,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(harami_config)

    # CDL_PIERCING - 明けの明星
    piercing_config = IndicatorConfig(
        indicator_name="CDL_PIERCING",
        adapter_function=PatternRecognitionIndicators.cdl_piercing,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(piercing_config)

    # CDL_DARK_CLOUD_COVER - 宵の明星
    dark_cloud_config = IndicatorConfig(
        indicator_name="CDL_DARK_CLOUD_COVER",
        adapter_function=PatternRecognitionIndicators.cdl_dark_cloud_cover,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(dark_cloud_config)

    # CDL_THREE_BLACK_CROWS - 三羽烏
    three_black_crows_config = IndicatorConfig(
        indicator_name="CDL_THREE_BLACK_CROWS",
        adapter_function=PatternRecognitionIndicators.cdl_three_black_crows,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(three_black_crows_config)

    # CDL_THREE_WHITE_SOLDIERS - 三兵
    three_white_soldiers_config = IndicatorConfig(
        indicator_name="CDL_THREE_WHITE_SOLDIERS",
        adapter_function=PatternRecognitionIndicators.cdl_three_white_soldiers,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(three_white_soldiers_config)

    # CDL_MARUBOZU - 丸坊主
    marubozu_config = IndicatorConfig(
        indicator_name="CDL_MARUBOZU",
        adapter_function=PatternRecognitionIndicators.cdl_marubozu,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(marubozu_config)

    # CDL_SPINNING_TOP - コマ
    spinning_top_config = IndicatorConfig(
        indicator_name="CDL_SPINNING_TOP",
        adapter_function=PatternRecognitionIndicators.cdl_spinning_top,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(spinning_top_config)

    # DOJI - 同事
    doji_config = IndicatorConfig(
        indicator_name="CDL_DOJI",
        adapter_function=PatternRecognitionIndicators.doji,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(doji_config)

    # HAMMER - ハンマー
    hammer_config = IndicatorConfig(
        indicator_name="HAMMER",
        adapter_function=PatternRecognitionIndicators.hammer,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(hammer_config)

    # ENGULFING_PATTERN - 包み足
    engulfing_config = IndicatorConfig(
        indicator_name="ENGULFING_PATTERN",
        adapter_function=PatternRecognitionIndicators.engulfing_pattern,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    indicator_registry.register(engulfing_config)

    # MORNING_STAR - 明けの明星
    morning_star_config = IndicatorConfig(
        indicator_name="MORNING_STAR",
        adapter_function=PatternRecognitionIndicators.morning_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    morning_star_config.param_map = {"open": "open_data"}
    indicator_registry.register(morning_star_config)

    # EVENING_STAR - 宵の明星
    evening_star_config = IndicatorConfig(
        indicator_name="EVENING_STAR",
        adapter_function=PatternRecognitionIndicators.evening_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    evening_star_config.param_map = {"open": "open_data"}
    indicator_registry.register(evening_star_config)
    # CDL_ENGULFING - 包み足
    engulfing_config = IndicatorConfig(
        indicator_name="CDL_ENGULFING",
        adapter_function=PatternRecognitionIndicators.cdl_engulfing,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    engulfing_config.param_map = {"open": "open_data"}
    indicator_registry.register(engulfing_config)

    # CDL_HAMMER - ハンマー
    hammer_config = IndicatorConfig(
        indicator_name="CDL_HAMMER",
        adapter_function=PatternRecognitionIndicators.cdl_hammer,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    hammer_config.param_map = {"open": "open_data"}
    indicator_registry.register(hammer_config)

    # CDL_MORNING_STAR - 明けの明星
    morning_star_cdl_config = IndicatorConfig(
        indicator_name="CDL_MORNING_STAR",
        adapter_function=PatternRecognitionIndicators.cdl_morning_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    morning_star_cdl_config.param_map = {"open": "open_data"}
    indicator_registry.register(morning_star_cdl_config)

    # CDL_EVENING_STAR - 宵の明星
    evening_star_cdl_config = IndicatorConfig(
        indicator_name="CDL_EVENING_STAR",
        adapter_function=PatternRecognitionIndicators.cdl_evening_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="pattern_recognition",
    )
    evening_star_cdl_config.param_map = {"open": "open_data"}
    indicator_registry.register(evening_star_cdl_config)


def setup_hilbert_transform_indicators():
    """Hilbert Transform系インジケーターの設定"""

    # HT_DCPERIOD
    ht_dcperiod_config = IndicatorConfig(
        indicator_name="HT_DCPERIOD",
        adapter_function=HilbertTransformIndicators.ht_dcperiod,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="hilbert_transform",
    )
    ht_dcperiod_config.param_map = {"close": "data"}
    indicator_registry.register(ht_dcperiod_config)

    # HT_DCPHASE
    ht_dcphase_config = IndicatorConfig(
        indicator_name="HT_DCPHASE",
        adapter_function=HilbertTransformIndicators.ht_dcphase,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="hilbert_transform",
    )
    ht_dcphase_config.param_map = {"close": "data"}
    indicator_registry.register(ht_dcphase_config)

    # HT_PHASOR
    ht_phasor_config = IndicatorConfig(
        indicator_name="HT_PHASOR",
        adapter_function=HilbertTransformIndicators.ht_phasor,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="hilbert_transform",
        output_names=["HT_PHASOR_0", "HT_PHASOR_1"],
        default_output="HT_PHASOR_0",
    )
    ht_phasor_config.param_map = {"close": "data"}
    indicator_registry.register(ht_phasor_config)

    # HT_SINE
    ht_sine_config = IndicatorConfig(
        indicator_name="HT_SINE",
        adapter_function=HilbertTransformIndicators.ht_sine,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="hilbert_transform",
        output_names=["HT_SINE_0", "HT_SINE_1"],
        default_output="HT_SINE_0",
    )
    ht_sine_config.param_map = {"close": "data"}
    indicator_registry.register(ht_sine_config)

    # HT_TRENDMODE
    ht_trendmode_config = IndicatorConfig(
        indicator_name="HT_TRENDMODE",
        adapter_function=HilbertTransformIndicators.ht_trendmode,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PATTERN_BINARY,
        category="hilbert_transform",
    )
    ht_trendmode_config.param_map = {"close": "data"}
    indicator_registry.register(ht_trendmode_config)


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()

    setup_statistics_indicators()
    setup_pattern_recognition_indicators()
    setup_hilbert_transform_indicators()


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
zlma_config.param_map = {"period": "length"}
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
indicator_registry.register(vwma_config)
vwma_config.param_map = {"close": "close", "volume": "volume", "length": "length"}

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

pvo_config = IndicatorConfig(
    indicator_name="PVO",
    adapter_function=MomentumIndicators.pvo,
    required_data=["volume"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    category="volume",
    output_names=["PVO_0", "PVO_1"],
    default_output="PVO_0",
)
pvo_config.add_parameter(
    ParameterConfig(name="fast", default_value=12, min_value=2, max_value=100)
)
pvo_config.add_parameter(
    ParameterConfig(name="slow", default_value=26, min_value=2, max_value=200)
)
pvo_config.add_parameter(
    ParameterConfig(name="signal", default_value=9, min_value=2, max_value=100)
)
pvo_config.param_map = {
    "volume": "volume",
    "fast": "fast",
    "slow": "slow",
    "signal": "signal",
}
indicator_registry.register(pvo_config)

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

# MAMA
mama_config = IndicatorConfig(
    indicator_name="MAMA",
    adapter_function=TrendIndicators.mama,
    required_data=["close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.PRICE_RATIO,
    category="trend",
    output_names=["MAMA_0", "MAMA_1"],
    default_output="MAMA_0",
)
mama_config.add_parameter(
    ParameterConfig(
        name="fastlimit",
        default_value=0.5,
        min_value=0.01,
        max_value=0.99,
        description="MAMA高速制限",
    )
)
mama_config.add_parameter(
    ParameterConfig(
        name="slowlimit",
        default_value=0.05,
        min_value=0.01,
        max_value=0.99,
        description="MAMA低速制限",
    )
)
mama_config.param_map = {
    "close": "data",
    "fastlimit": "fastlimit",
    "slowlimit": "slowlimit",
}
indicator_registry.register(mama_config)

# MAXINDEX
maxindex_config = IndicatorConfig(
    indicator_name="MAXINDEX",
    adapter_function=TrendIndicators.maxindex,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="statistics",
)
maxindex_config.add_parameter(
    ParameterConfig(
        name="period",
        default_value=14,
        min_value=2,
        max_value=100,
        description="最大値インデックス期間",
    )
)
maxindex_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(maxindex_config)

# MININDEX
minindex_config = IndicatorConfig(
    indicator_name="MININDEX",
    adapter_function=TrendIndicators.minindex,
    required_data=["close"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="statistics",
)
minindex_config.add_parameter(
    ParameterConfig(
        name="period",
        default_value=14,
        min_value=2,
        max_value=100,
        description="最小値インデックス期間",
    )
)
minindex_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(minindex_config)

# MINMAX
minmax_config = IndicatorConfig(
    indicator_name="MINMAX",
    adapter_function=TrendIndicators.minmax,
    required_data=["close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="statistics",
    output_names=["MINMAX_MIN", "MINMAX_MAX"],
    default_output="MINMAX_MIN",
)
minmax_config.add_parameter(
    ParameterConfig(
        name="period",
        default_value=14,
        min_value=2,
        max_value=100,
        description="最小最大期間",
    )
)
minmax_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(minmax_config)

# MINMAXINDEX
minmaxindex_config = IndicatorConfig(
    indicator_name="MINMAXINDEX",
    adapter_function=TrendIndicators.minmaxindex,
    required_data=["close"],
    result_type=IndicatorResultType.COMPLEX,
    scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
    category="statistics",
    output_names=["MINMAXINDEX_MIN", "MINMAXINDEX_MAX"],
    default_output="MINMAXINDEX_MIN",
)
minmaxindex_config.add_parameter(
    ParameterConfig(
        name="period",
        default_value=14,
        min_value=2,
        max_value=100,
        description="最小最大インデックス期間",
    )
)
minmaxindex_config.param_map = {"close": "data", "period": "length"}
indicator_registry.register(minmaxindex_config)

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
rsi_ema_cross_config.param_map = {"close": "data"}
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
indicator_registry.register(eom_cfg)

kvo_cfg = IndicatorConfig(
    indicator_name="KVO",
    adapter_function=VolumeIndicators.kvo,
    required_data=["high", "low", "close", "volume"],
    result_type=IndicatorResultType.SINGLE,
    scale_type=IndicatorScaleType.VOLUME,
    category="volume",
)
kvo_cfg.add_parameter(
    ParameterConfig(name="fast", default_value=34, min_value=2, max_value=100)
)
kvo_cfg.add_parameter(
    ParameterConfig(name="slow", default_value=55, min_value=2, max_value=200)
)
indicator_registry.register(kvo_cfg)

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
indicator_registry.register(cmf_cfg)
