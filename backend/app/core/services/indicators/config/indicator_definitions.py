"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

from .indicator_config import (
    IndicatorConfig,
    ParameterConfig,
    IndicatorResultType,
    indicator_registry,
)


def setup_momentum_indicators():
    """モメンタム系インジケーターの設定"""

    # RSI
    rsi_config = IndicatorConfig(
        indicator_name="RSI",
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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

    # APO (複数パラメータの例)
    apo_config = IndicatorConfig(
        indicator_name="APO",
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{fast_period}_{slow_period}",
    )
    apo_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=12,
            min_value=2,
            max_value=50,
            description="短期期間",
        )
    )
    apo_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=26,
            min_value=10,
            max_value=100,
            description="長期期間",
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

    # PPO
    ppo_config = IndicatorConfig(
        indicator_name="PPO",
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{fast_period}_{slow_period}",
    )
    ppo_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=12,
            min_value=2,
            max_value=50,
            description="短期期間",
        )
    )
    ppo_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=26,
            min_value=10,
            max_value=100,
            description="長期期間",
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

    # MACD (複数値結果の例)
    macd_config = IndicatorConfig(
        indicator_name="MACD",
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="macd_handler",
        legacy_name_format="{indicator}_{fast_period}",
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


def setup_trend_indicators():
    """トレンド系インジケーターの設定"""

    # SMA
    sma_config = IndicatorConfig(
        indicator_name="SMA",
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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

    # Bollinger Bands (複数値結果の例)
    bb_config = IndicatorConfig(
        indicator_name="BB",
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="bb_handler",
        legacy_name_format="BB_MIDDLE_{period}",
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


def setup_volume_indicators():
    """出来高系インジケーターの設定"""

    # OBV (パラメータなしの例)
    obv_config = IndicatorConfig(
        indicator_name="OBV",
        required_data=["close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}",
    )
    indicator_registry.register(obv_config)

    # ADOSC (複数パラメータの例)
    adosc_config = IndicatorConfig(
        indicator_name="ADOSC",
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{fast_period}_{slow_period}",
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="fast_period",
            default_value=3,
            min_value=1,
            max_value=20,
            description="短期期間",
        )
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="slow_period",
            default_value=10,
            min_value=5,
            max_value=50,
            description="長期期間",
        )
    )
    indicator_registry.register(adosc_config)


def setup_additional_indicators():
    """オートストラテジー用の追加インジケーター設定"""

    # STOCH (Stochastic Oscillator)
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="stoch_handler",
        legacy_name_format="STOCH_{k_period}",
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="k_period",
            default_value=14,
            min_value=5,
            max_value=30,
            description="%K期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="d_period",
            default_value=3,
            min_value=1,
            max_value=10,
            description="%D期間",
        )
    )
    stoch_config.add_parameter(
        ParameterConfig(
            name="slowing",
            default_value=3,
            min_value=1,
            max_value=10,
            description="スローイング期間",
        )
    )
    indicator_registry.register(stoch_config)

    # CCI (Commodity Channel Index)
    cci_config = IndicatorConfig(
        indicator_name="CCI",
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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

    # ADX (Average Directional Index)
    adx_config = IndicatorConfig(
        indicator_name="ADX",
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
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


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()
    setup_additional_indicators()


# モジュール読み込み時に初期化
initialize_all_indicators()
