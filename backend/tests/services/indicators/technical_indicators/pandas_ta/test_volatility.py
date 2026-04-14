"""
VolatilityIndicators のテスト
"""

import warnings

import numpy as np
import pandas as pd

from app.services.indicators.technical_indicators.pandas_ta.volatility import (
    VolatilityIndicators,
)


class TestVolatilityIndicators:
    """VolatilityIndicators のテスト"""

    def test_hwc_suppresses_runtime_warnings(self):
        """HWC は RuntimeWarning を外に漏らさない"""
        close = pd.Series(
            np.linspace(100.0, 200.0, 500)
            + np.random.default_rng(42).normal(0.0, 1.0, 500)
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = VolatilityIndicators.hwc(close=close, na=2, nb=3, nc=4)

        assert captured == []
        assert isinstance(result, tuple)
        assert len(result) == 3
