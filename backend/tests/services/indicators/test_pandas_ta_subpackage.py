"""pandas_ta サブパッケージの構成テスト。"""

from app.services.indicators.technical_indicators.pandas_ta import (
    MomentumIndicators,
    OverlapIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)


def test_pandas_ta_subpackage_exports_indicator_classes():
    """移動後のサブパッケージが各クラスを公開すること。"""
    classes = [
        MomentumIndicators,
        OverlapIndicators,
        TrendIndicators,
        VolatilityIndicators,
        VolumeIndicators,
    ]

    for cls in classes:
        assert cls.__module__.startswith(
            "app.services.indicators.technical_indicators.pandas_ta."
        )
