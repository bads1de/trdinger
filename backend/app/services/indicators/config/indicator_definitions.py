"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

from app.services.indicators.technical_indicators.cycle import CycleIndicators
from app.services.indicators.technical_indicators.math_operators import (
    MathOperatorsIndicators,
)
from app.services.indicators.technical_indicators.math_transform import (
    MathTransformIndicators,
)
from app.services.indicators.technical_indicators.momentum import (
    MomentumIndicators,
)
from app.services.indicators.technical_indicators.pattern_recognition import (
    PatternRecognitionIndicators,
)
from app.services.indicators.technical_indicators.price_transform import (
    PriceTransformIndicators,
)
from app.services.indicators.technical_indicators.statistics import (
    StatisticsIndicators,
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

    # RSI
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="RSI計算期間",
        )
    )
    indicator_registry.register(rsi_config)

    # MACD
    macd_config = IndicatorConfig(
        indicator_name="MACD",
        adapter_function=MomentumIndicators.macd,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="macd_handler",
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=12,
            min_value=2,
            max_value=50,
            description="短期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=26,
            min_value=10,
            max_value=100,
            description="長期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="signal_period",
            default_value=9,
            min_value=2,
            max_value=50,
            description="シグナル期間",
        )
    )
    indicator_registry.register(macd_config)

    # STOCH
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        adapter_function=MomentumIndicators.stoch,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="stoch_handler",
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="momentum",
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
    indicator_registry.register(stoch_config)

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
    indicator_registry.register(cci_config)

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
            name="period",
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
        adapter_function=MomentumIndicators.williams_r,
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
            name="period",
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
            name="period",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROC計算期間",
        )
    )
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
            name="period",
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
            name="timeperiod1",
            default_value=7,
            min_value=1,
            max_value=100,
            description="期間1",
        )
    )
    ultosc_config.add_parameter(
        ParameterConfig(
            name="timeperiod2",
            default_value=14,
            min_value=1,
            max_value=100,
            description="期間2",
        )
    )
    ultosc_config.add_parameter(
        ParameterConfig(
            name="timeperiod3",
            default_value=28,
            min_value=1,
            max_value=100,
            description="期間3",
        )
    )
    indicator_registry.register(ultosc_config)

    # BOP
    bop_config = IndicatorConfig(
        indicator_name="BOP",
        adapter_function=MomentumIndicators.bop,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="momentum",
    )
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
            name="period",
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
            name="period",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCP計算期間",
        )
    )
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
            name="period",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCR計算期間",
        )
    )
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
            name="period",
            default_value=10,
            min_value=1,
            max_value=100,
            description="ROCR100計算期間",
        )
    )
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
            name="period",
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="Minus DM計算期間",
        )
    )
    indicator_registry.register(minus_dm_config)


def setup_trend_indicators():
    """トレンド系インジケーターの設定（オートストラテジー最適化版）"""

    # SMA
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="移動平均期間",
        )
    )
    indicator_registry.register(sma_config)

    # EMA
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="移動平均期間",
        )
    )
    indicator_registry.register(ema_config)

    # WMA
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="加重移動平均期間",
        )
    )
    indicator_registry.register(wma_config)

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
            name="period",
            default_value=14,
            min_value=2,
            max_value=200,
            description="三角移動平均期間",
        )
    )
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
            name="period",
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
    indicator_registry.register(atr_config)

    # Bollinger Bands
    bb_config = IndicatorConfig(
        indicator_name="BB",
        adapter_function=VolatilityIndicators.bollinger_bands,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="bb_handler",
        scale_type=IndicatorScaleType.PRICE_RATIO,
        category="volatility",
    )
    bb_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=20,
            min_value=2,
            max_value=100,
            description="移動平均期間",
        )
    )
    bb_config.add_parameter(
        ParameterConfig(
            name="std_dev",
            default_value=2.0,
            min_value=0.5,
            max_value=5.0,
            description="標準偏差の倍数",
        )
    )
    indicator_registry.register(bb_config)

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


def setup_price_transform_indicators():
    """価格変換系インジケーターの設定"""

    # AVGPRICE
    avgprice_config = IndicatorConfig(
        indicator_name="AVGPRICE",
        adapter_function=PriceTransformIndicators.avgprice,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="price_transform",
    )
    indicator_registry.register(avgprice_config)

    # MEDPRICE
    medprice_config = IndicatorConfig(
        indicator_name="MEDPRICE",
        adapter_function=PriceTransformIndicators.medprice,
        required_data=["high", "low"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="price_transform",
    )
    indicator_registry.register(medprice_config)

    # TYPPRICE
    typprice_config = IndicatorConfig(
        indicator_name="TYPPRICE",
        adapter_function=PriceTransformIndicators.typprice,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="price_transform",
    )
    indicator_registry.register(typprice_config)

    # WCLPRICE
    wclprice_config = IndicatorConfig(
        indicator_name="WCLPRICE",
        adapter_function=PriceTransformIndicators.wclprice,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="price_transform",
    )
    indicator_registry.register(wclprice_config)


def setup_cycle_indicators():
    """サイクル系インジケーターの設定"""

    # HT_DCPERIOD
    ht_dcperiod_config = IndicatorConfig(
        indicator_name="HT_DCPERIOD",
        adapter_function=CycleIndicators.ht_dcperiod,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="cycle",
    )
    indicator_registry.register(ht_dcperiod_config)

    # HT_DCPHASE
    ht_dcphase_config = IndicatorConfig(
        indicator_name="HT_DCPHASE",
        adapter_function=CycleIndicators.ht_dcphase,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="cycle",
    )
    indicator_registry.register(ht_dcphase_config)

    # HT_TRENDMODE
    ht_trendmode_config = IndicatorConfig(
        indicator_name="HT_TRENDMODE",
        adapter_function=CycleIndicators.ht_trendmode,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="cycle",
    )
    indicator_registry.register(ht_trendmode_config)

    # HT_PHASOR
    ht_phasor_config = IndicatorConfig(
        indicator_name="HT_PHASOR",
        adapter_function=CycleIndicators.ht_phasor,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="cycle",
    )
    indicator_registry.register(ht_phasor_config)

    # HT_SINE
    ht_sine_config = IndicatorConfig(
        indicator_name="HT_SINE",
        adapter_function=CycleIndicators.ht_sine,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="cycle",
    )
    indicator_registry.register(ht_sine_config)


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
            name="period",
            default_value=5,
            min_value=2,
            max_value=100,
            description="BETA計算期間",
        )
    )
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰期間",
        )
    )
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
            name="period",
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="時系列予測期間",
        )
    )
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
            name="period",
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰角度期間",
        )
    )
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
            name="period",
            default_value=14,
            min_value=2,
            max_value=100,
            description="線形回帰切片期間",
        )
    )
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
    indicator_registry.register(linearreg_slope_config)


def setup_math_transform_indicators():
    """数学変換系インジケーターの設定"""

    # 三角関数系
    for func_name in [
        "ACOS",
        "ASIN",
        "ATAN",
        "COS",
        "COSH",
        "SIN",
        "SINH",
        "TAN",
        "TANH",
    ]:
        config = IndicatorConfig(
            indicator_name=func_name,
            adapter_function=getattr(MathTransformIndicators, func_name.lower()),
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
            scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
            category="math_transform",
        )
        if func_name in ["ACOS", "ASIN"]:
            config.needs_normalization = True
        indicator_registry.register(config)

    # その他の数学関数
    for func_name in ["CEIL", "EXP", "FLOOR", "LN", "LOG10", "SQRT"]:
        config = IndicatorConfig(
            indicator_name=func_name,
            adapter_function=getattr(MathTransformIndicators, func_name.lower()),
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
            scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
            category="math_transform",
        )
        indicator_registry.register(config)


def setup_math_operators_indicators():
    """数学演算子系インジケーターの設定"""

    # 二項演算子（2つのデータが必要）
    for func_name in ["ADD", "DIV", "MULT", "SUB"]:
        config = IndicatorConfig(
            indicator_name=func_name,
            adapter_function=getattr(MathOperatorsIndicators, func_name.lower()),
            required_data=["data0", "data1"],  # 2つのデータ系列が必要
            result_type=IndicatorResultType.SINGLE,
            scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
            category="math_operators",
        )
        indicator_registry.register(config)

    # 期間ベース演算子
    for func_name, method_name, default_period in [
        ("MAX", "max_value", 30),
        ("MIN", "min_value", 30),
        ("SUM", "sum_values", 30),
    ]:
        config = IndicatorConfig(
            indicator_name=func_name,
            adapter_function=getattr(MathOperatorsIndicators, method_name),
            required_data=["close"],
            result_type=IndicatorResultType.SINGLE,
            scale_type=(
                IndicatorScaleType.PRICE_ABSOLUTE
                if func_name in ["MAX", "MIN", "SUM"]
                else IndicatorScaleType.OSCILLATOR_0_100
            ),
            category="math_operators",
        )
        config.add_parameter(
            ParameterConfig(
                name="period",
                default_value=default_period,
                min_value=2,
                max_value=200,
                description=f"{func_name}計算期間",
            )
        )
        indicator_registry.register(config)


def setup_pattern_recognition_indicators():
    """パターン認識系インジケーターの設定"""

    # 基本的なキャンドルスティックパターン（パラメータなし）
    basic_patterns = [
        "CDL_DOJI",
        "CDL_HAMMER",
        "CDL_HANGING_MAN",
        "CDL_SHOOTING_STAR",
        "CDL_ENGULFING",
        "CDL_HARAMI",
        "CDL_PIERCING",
        "CDL_THREE_BLACK_CROWS",
        "CDL_THREE_WHITE_SOLDIERS",
    ]

    for pattern_name in basic_patterns:
        config = IndicatorConfig(
            indicator_name=pattern_name,
            adapter_function=getattr(
                PatternRecognitionIndicators, pattern_name.lower()
            ),
            required_data=["open_data", "high", "low", "close"],
            result_type=IndicatorResultType.SINGLE,
            scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
            category="pattern_recognition",
        )
        indicator_registry.register(config)

    # パラメータ付きパターン
    # DARK_CLOUD_COVER
    dark_cloud_config = IndicatorConfig(
        indicator_name="CDL_DARK_CLOUD_COVER",
        adapter_function=PatternRecognitionIndicators.cdl_dark_cloud_cover,
        required_data=["open_data", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="pattern_recognition",
    )
    dark_cloud_config.add_parameter(
        ParameterConfig(
            name="penetration",
            default_value=0.5,
            min_value=0.1,
            max_value=1.0,
            description="浸透率",
        )
    )
    indicator_registry.register(dark_cloud_config)

    # MORNING_STAR
    morning_star_config = IndicatorConfig(
        indicator_name="CDL_MORNING_STAR",
        adapter_function=PatternRecognitionIndicators.cdl_morning_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="pattern_recognition",
    )
    morning_star_config.add_parameter(
        ParameterConfig(
            name="penetration",
            default_value=0.3,
            min_value=0.1,
            max_value=1.0,
            description="浸透率",
        )
    )
    indicator_registry.register(morning_star_config)

    # EVENING_STAR
    evening_star_config = IndicatorConfig(
        indicator_name="CDL_EVENING_STAR",
        adapter_function=PatternRecognitionIndicators.cdl_evening_star,
        required_data=["open", "high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
        category="pattern_recognition",
    )
    evening_star_config.add_parameter(
        ParameterConfig(
            name="penetration",
            default_value=0.3,
            min_value=0.1,
            max_value=1.0,
            description="浸透率",
        )
    )
    indicator_registry.register(evening_star_config)

    # 追加のパターン認識インジケーター
    additional_patterns = [
        "CDL_ABANDONED_BABY",
        "CDL_ADVANCE_BLOCK",
        "CDL_BELT_HOLD",
        "CDL_BREAKAWAY",
        "CDL_CLOSING_MARUBOZU",
        "CDL_CONCEALING_BABY_SWALLOW",
        "CDL_COUNTERATTACK",
        "CDL_DRAGONFLY_DOJI",
        "CDL_GAPSIDE_SIDE_WHITE",
        "CDL_GRAVESTONE_DOJI",
        "CDL_HOMINGPIGEON",
        "CDL_IDENTICAL_THREE_CROWS",
    ]

    for pattern_name in additional_patterns:
        # パターン名をメソッド名に変換（例：CDL_ABANDONED_BABY -> cdl_abandoned_baby）
        method_name = pattern_name.lower()

        # 対応するメソッドが存在するかチェック（存在しない場合はスキップ）
        if hasattr(PatternRecognitionIndicators, method_name):
            config = IndicatorConfig(
                indicator_name=pattern_name,
                adapter_function=getattr(PatternRecognitionIndicators, method_name),
                required_data=["open", "high", "low", "close"],
                result_type=IndicatorResultType.SINGLE,
                scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
                category="pattern_recognition",
            )
            indicator_registry.register(config)


def setup_ml_indicators():
    """ML予測確率指標の設定"""

    # ML_UP_PROB
    ml_up_prob_config = IndicatorConfig(
        indicator_name="ML_UP_PROB",
        adapter_function=None,  # IndicatorCalculatorで直接処理
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_1,
        category="ml_prediction",
    )
    indicator_registry.register(ml_up_prob_config)

    # ML_DOWN_PROB
    ml_down_prob_config = IndicatorConfig(
        indicator_name="ML_DOWN_PROB",
        adapter_function=None,  # IndicatorCalculatorで直接処理
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_1,
        category="ml_prediction",
    )
    indicator_registry.register(ml_down_prob_config)

    # ML_RANGE_PROB
    ml_range_prob_config = IndicatorConfig(
        indicator_name="ML_RANGE_PROB",
        adapter_function=None,  # IndicatorCalculatorで直接処理
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_1,
        category="ml_prediction",
    )
    indicator_registry.register(ml_range_prob_config)


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()
    setup_price_transform_indicators()
    setup_cycle_indicators()
    setup_statistics_indicators()
    setup_math_transform_indicators()
    setup_math_operators_indicators()
    setup_pattern_recognition_indicators()
    setup_ml_indicators()


# モジュール読み込み時に初期化
initialize_all_indicators()
