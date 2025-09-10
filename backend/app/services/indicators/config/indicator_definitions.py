"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

import logging

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

logger = logging.getLogger(__name__)


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
    cmf_cfg.param_map = {"length": "length"}
    indicator_registry.register(cmf_cfg)


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()


# モジュール読み込み時に初期化
# python-ta動的処理設定のグローバル設定
PANDAS_TA_CONFIG = {
    "RSI": {
        "function": "rsi",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "SMA": {
        "function": "sma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "EMA": {
        "function": "ema",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "WMA": {
        "function": "wma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "MACD": {
        "function": "macd",
        "params": {"fast": ["fast"], "slow": ["slow"], "signal": ["signal"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["MACD", "Signal", "Histogram"],
        "default_values": {"fast": 12, "slow": 26, "signal": 9},
    },
    "SUPERTREND": {
        "function": "supertrend",
        "params": {"length": ["length"], "multiplier": ["multiplier", "factor"]},
        "data_column": "open_high_low_close",
        "returns": "complex",
        "return_cols": ["ST", "D"],
        "default_values": {"length": 10, "multiplier": 3.0},
    },
    "UI": {
        "function": "ui",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "PPO": {
        "function": "ppo",
        "params": {"fast": ["fast"], "slow": ["slow"], "signal": ["signal"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["PPO_0", "PPO_1", "PPO_2"],
        "default_values": {"fast": 12, "slow": 26, "signal": 9},
    },
    "TEMA": {
        "function": "tema",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "BBANDS": {
        "function": "bbands",
        "params": {"length": ["length", "period"], "std": ["std", "multiplier"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["BBL", "BBM", "BBU"],
        "default_values": {"length": 20, "std": 2.0},
    },
    "AO": {
        "function": "ao",
        "params": {},
        "multi_column": True,
        "data_columns": ["High", "Low"],
        "returns": "single",
        "default_values": {},
    },
    "T3": {
        "function": "t3",
        "params": {"length": ["length", "period"], "a": ["a", "vfactor"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 5, "a": 0.7},
    },
    "EFI": {
        "function": "efi",
        "params": {
            "length": ["length", "period"],
            "mamode": ["mamode"],
            "drift": ["drift"],
        },
        "multi_column": True,
        "data_columns": ["Close", "Volume"],
        "returns": "single",
        "default_values": {"length": 13, "mamode": "ema", "drift": 1},
    },
    "PVR": {
        "function": "pvr",
        "params": {},
        "multi_column": True,
        "data_columns": ["Close", "Volume"],
        "returns": "single",
        "default_values": {},
    },
    "CTI": {
        "function": "cti",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
}

POSITIONAL_DATA_FUNCTIONS = {
    "rsi",
    "wma",
    "sar",
    "roc",
    "stoch",
    "bbands",
    "macd",
    "dpo",
    "rmi",
    "kama",
    "trima",
    "wma",
    "ma",
    "midpoint",
    "midprice",
    "ht_trendline",
    "adosc",
    "correl",
    "linearreg",
    "stddev",
    "tsf",
    "var",
    "linearreg_angle",
    "linearreg_intercept",
    "linearreg_slope",
    "hma",
    "zlma",
    "swma",
    "alma",
    "rma",
    "tsi",
    "pvo",
    "cfo",
    "cti",
    "sma_slope",
    "price_ema_ratio",
    "beta",
    "belta",
    "qqe",
    "smi",
    "trix",
    "apo",
    "WMA",
    "TRIMA",
    "MA",
    "chop",
    "vortex",
    "BBANDS",
    "hilo",
    "ad",
    "eom",
    "kvo",
    "cmf",
}

# ---- Append new pandas-ta indicators and custom ones ----
initialize_all_indicators()


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


# EFI (Elder's Force Index)
efi_config = IndicatorConfig(
    indicator_name="EFI",
    # Use pandas-ta directly instead of adapter function
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


def setup_pandas_ta_indicators():
    """
    pandas-ta設定からインジケーターを登録
    """
    # PANDAS_TA_CONFIGを使用してインジケーターを登録（外から参照可能）
    pass


# 初期化時にpandas-taインジケーターを設定
def initialize_pandas_ta_indicators():
    """pandas-taインジケーターの初期化"""
    setup_pandas_ta_indicators()


# モジュール読み込み時に初期化
initialize_pandas_ta_indicators()
