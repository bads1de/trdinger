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

# 新しいnumpy配列ベース指標クラス（オートストラテジー最適化版）
from app.core.services.indicators.trend import TrendIndicators
from app.core.services.indicators.momentum import MomentumIndicators
from app.core.services.indicators.volatility import VolatilityIndicators

# 古いアダプター関数（段階的移行のため一時的に保持）
try:
    from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
    from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
    from app.core.services.indicators.adapters.volatility_adapter import (
        VolatilityAdapter,
    )
    from app.core.services.indicators.adapters.volume_adapter import VolumeAdapter
except ImportError:
    # 古いアダプターが削除されている場合はスキップ
    TrendAdapter = None
    MomentumAdapter = None
    VolatilityAdapter = None
    VolumeAdapter = None


def setup_momentum_indicators():
    """モメンタム系インジケーターの設定（オートストラテジー最適化版）"""

    # RSI - 新しいnumpy配列ベース指標クラス使用
    rsi_config = IndicatorConfig(
        indicator_name="RSI",
        adapter_function=MomentumIndicators.rsi,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
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

    # MACD - 新しいnumpy配列ベース指標クラス使用（複数値結果）
    macd_config = IndicatorConfig(
        indicator_name="MACD",
        adapter_function=MomentumIndicators.macd,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="macd_handler",
        legacy_name_format="{indicator}_{fast_period}",
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=12,
            min_value=2,
            max_value=50,
            description="短期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=26,
            min_value=10,
            max_value=100,
            description="長期期間",
        )
    )
    macd_config.add_parameter(
        ParameterConfig(
            name="signal",
            default_value=9,
            min_value=2,
            max_value=50,
            description="シグナル期間",
        )
    )
    indicator_registry.register(macd_config)


def setup_trend_indicators():
    """トレンド系インジケーターの設定（オートストラテジー最適化版）"""

    # SMA - 新しいnumpy配列ベース指標クラス使用
    sma_config = IndicatorConfig(
        indicator_name="SMA",
        adapter_function=TrendIndicators.sma,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.PRICE_RATIO,
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

    # EMA - 新しいnumpy配列ベース指標クラス使用
    ema_config = IndicatorConfig(
        indicator_name="EMA",
        adapter_function=TrendIndicators.ema,
        required_data=["close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.PRICE_RATIO,
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
    """ボラティリティ系インジケーターの設定（オートストラテジー最適化版）"""

    # ATR - 新しいnumpy配列ベース指標クラス使用
    atr_config = IndicatorConfig(
        indicator_name="ATR",
        adapter_function=VolatilityIndicators.atr,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
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

    # Bollinger Bands - 新しいnumpy配列ベース指標クラス使用
    bb_config = IndicatorConfig(
        indicator_name="BB",
        adapter_function=VolatilityIndicators.bollinger_bands,
        required_data=["close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="bb_handler",
        legacy_name_format="BB_MIDDLE_{period}",
        scale_type=IndicatorScaleType.PRICE_RATIO,
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
    """出来高系インジケーターの設定（一時的に無効化）"""

    # 出来高系指標は新しいアーキテクチャでまだ実装されていないため、
    # 一時的に無効化します
    pass

    # TODO: 出来高系指標の新しい実装を追加
    # # OBV (パラメータなしの例)
    # obv_config = IndicatorConfig(
    #     indicator_name="OBV",
    #     adapter_function=VolumeIndicators.obv,  # 新しい実装が必要
    #     required_data=["close", "volume"],
    #     result_type=IndicatorResultType.SINGLE,
    #     legacy_name_format="{indicator}",
    #     scale_type=IndicatorScaleType.VOLUME,
    # )
    # indicator_registry.register(obv_config)

    # TODO: ADOSC等の出来高系指標も新しい実装が必要
    # # ADOSC (複数パラメータの例)
    # adosc_config = IndicatorConfig(
    #     indicator_name="ADOSC",
    #     required_data=["high", "low", "close", "volume"],
    #     result_type=IndicatorResultType.SINGLE,
    #     legacy_name_format="{indicator}_{fast_period}_{slow_period}",
    # )
    # adosc_config.add_parameter(
    #     ParameterConfig(
    #         name="fast_period",
    #         default_value=3,
    #         min_value=1,
    #         max_value=20,
    #         description="短期期間",
    #     )
    # )
    # adosc_config.add_parameter(
    #     ParameterConfig(
    #         name="slow_period",
    #         default_value=10,
    #         min_value=5,
    #         max_value=50,
    #         description="長期期間",
    #     )
    # )
    # indicator_registry.register(adosc_config)


def setup_additional_indicators():
    """オートストラテジー用の追加インジケーター設定（オートストラテジー最適化版）"""

    # STOCH (Stochastic Oscillator) - 新しいnumpy配列ベース指標クラス使用
    stoch_config = IndicatorConfig(
        indicator_name="STOCH",
        adapter_function=MomentumIndicators.stoch,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.COMPLEX,
        result_handler="stoch_handler",
        legacy_name_format="STOCH_{fastk_period}",
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
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

    # CCI (Commodity Channel Index) - 新しいnumpy配列ベース指標クラス使用
    cci_config = IndicatorConfig(
        indicator_name="CCI",
        adapter_function=MomentumIndicators.cci,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.OSCILLATOR_PLUS_MINUS_100,
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

    # ADX (Average Directional Index) - 新しいnumpy配列ベース指標クラス使用
    adx_config = IndicatorConfig(
        indicator_name="ADX",
        adapter_function=VolatilityIndicators.adx,
        required_data=["high", "low", "close"],
        result_type=IndicatorResultType.SINGLE,
        legacy_name_format="{indicator}_{period}",
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
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
