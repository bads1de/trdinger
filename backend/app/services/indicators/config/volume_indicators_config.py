"""
出来高系インジケーターの設定
"""

from app.services.indicators.technical_indicators.volume import VolumeIndicators

from .indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
    ParameterConfig,
    indicator_registry,
)


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
        # pandas-ta設定
        pandas_function="obv",
        multi_column=True,
        data_columns=["Close", "Volume"],
        returns="single",
        default_values={},
    )
    obv_config.param_map = {"close": "close", "volume": "volume"}
    indicator_registry.register(obv_config)

    # AD
    ad_config = IndicatorConfig(
        indicator_name="AD",
        adapter_function=VolumeIndicators.ad,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.VOLUME,
        category="volume",
        # pandas-ta設定
        pandas_function="ad",
        multi_column=True,
        data_columns=["High", "Low", "Close", "Volume"],
        returns="single",
        default_values={},
    )
    ad_config.param_map = {"high": "high", "low": "low", "close": "close", "volume": "volume"}
    indicator_registry.register(ad_config)

    # ADOSC
    adosc_config = IndicatorConfig(
        indicator_name="ADOSC",
        adapter_function=VolumeIndicators.adosc,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.MOMENTUM_ZERO_CENTERED,
        category="volume",
        # pandas-ta設定
        pandas_function="adosc",
        multi_column=True,
        data_columns=["High", "Low", "Close", "Volume"],
        returns="single",
        default_values={"fast": 3, "slow": 10},
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="fast",
            default_value=3,
            min_value=2,
            max_value=50,
            description="高速期間",
        )
    )
    adosc_config.add_parameter(
        ParameterConfig(
            name="slow",
            default_value=10,
            min_value=2,
            max_value=100,
            description="低速期間",
        )
    )
    adosc_config.param_map = {
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "fast": "fast",
        "slow": "slow",
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
        # pandas-ta設定
        pandas_function="cmf",
        multi_column=True,
        data_columns=["High", "Low", "Close", "Volume"],
        returns="single",
        default_values={"length": 20},
    )
    cmf_cfg.add_parameter(
        ParameterConfig(
            name="length",
            default_value=20,
            min_value=2,
            max_value=200,
            description="Chaikin Money Flow期間",
        )
    )
    cmf_cfg.param_map = {"high": "high", "low": "low", "close": "close", "volume": "volume", "length": "length"}
    indicator_registry.register(cmf_cfg)

    # VWAP
    vwap_config = IndicatorConfig(
        indicator_name="VWAP",
        adapter_function=VolumeIndicators.vwap,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
        category="volume",
    )
    indicator_registry.register(vwap_config)

    # MFI
    mfi_config = IndicatorConfig(
        indicator_name="MFI",
        adapter_function=VolumeIndicators.mfi,
        required_data=["high", "low", "close", "volume"],
        result_type=IndicatorResultType.SINGLE,
        scale_type=IndicatorScaleType.OSCILLATOR_0_100,
        category="volume",
        output_names=["MFI"],
        default_output="MFI",
        aliases=["MFI", "MONEY_FLOW_INDEX"],
        # pandas-ta設定
        pandas_function="mfi",
        multi_column=True,
        data_columns=["High", "Low", "Close", "Volume"],
        returns="single",
        default_values={"length": 14, "drift": 1},
    )
    mfi_config.add_parameter(
        ParameterConfig(
            name="length",
            default_value=14,
            min_value=2,
            max_value=100,
            description="MFI calculation period",
        )
    )
    mfi_config.add_parameter(
        ParameterConfig(
            name="drift",
            default_value=1,
            min_value=1,
            max_value=10,
            description="Difference period",
        )
    )
    mfi_config.param_map = {"high": "high", "low": "low", "close": "close", "volume": "volume", "length": "length", "drift": "drift"}
    indicator_registry.register(mfi_config)