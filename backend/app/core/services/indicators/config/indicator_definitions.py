"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

from .indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
    IndicatorScaleType,
    indicator_registry,
)

from app.core.services.indicators.trend import TrendIndicators
from app.core.services.indicators.momentum import MomentumIndicators
from app.core.services.indicators.volatility import VolatilityIndicators


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

    # ADX
    adx_config = IndicatorConfig(
        indicator_name="ADX",
        adapter_function=VolatilityIndicators.adx,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="volatility",
    )
    adx_config.add_parameter(
        ParameterConfig(
            name="period",
            default_value=14,
            min_value=5,
            max_value=50,
            description="ADX計算期間",
        )
    )
    indicator_registry.register(adx_config)


def setup_volume_indicators():
    """出来高系インジケーターの設定"""
    # OBV
    obv_config = IndicatorConfig(
        indicator_name="OBV",
        # adapter_function=VolumeIndicators.obv, # 新しい実装が必要
        required_data=["close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.VOLUME,
        category="volume",
    )
    indicator_registry.register(obv_config)


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()


# モジュール読み込み時に初期化
initialize_all_indicators()
