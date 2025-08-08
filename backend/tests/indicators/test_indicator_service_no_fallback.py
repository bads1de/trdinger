import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.services.indicators import TechnicalIndicatorService, TALibError


def make_df(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    return pd.DataFrame({
        "Open": np.linspace(100, 120, n),
        "High": np.linspace(101, 121, n),
        "Low": np.linspace(99, 119, n),
        "Close": np.linspace(100, 120, n),
        "Volume": np.random.randint(1000, 2000, size=n),
    }, index=idx)


def test_indicator_service_raises_when_talib_unavailable():
    df = make_df()
    svc = TechnicalIndicatorService()

    # MomentumIndicators.rsi 内の talib.RSI 呼び出しをエラーにする
    with patch("app.services.indicators.technical_indicators.momentum.talib.RSI", side_effect=Exception("talib unavailable")):
        with pytest.raises(TALibError):
            svc.calculate_indicator(df, "RSI", {"period": 14})

