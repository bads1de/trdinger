import numpy as np
import pandas as pd
from app.services.ml.feature_engineering.enhanced_crypto_features import (
    EnhancedCryptoFeatures,
)


def make_df(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n) + np.sin(np.arange(n)) * 0.5,
            "Volume": np.random.randint(1000, 2000, size=n),
            "open_interest": np.random.randint(10000, 20000, size=n),
            "funding_rate": np.random.randn(n) * 0.0001,
            "fear_greed_value": np.random.randint(0, 100, size=n),
        },
        index=idx,
    )
    return df


def test_enhanced_features_with_talib_available():
    df = make_df()
    ecf = EnhancedCryptoFeatures()

    # TA-Libが利用可能な環境で正常に動作し、主要なテクニカル列が生成されることを確認
    result = ecf.create_comprehensive_features(df)

    # RSI列
    assert any(col.startswith("rsi_") for col in result.columns)
    # BB Position列
    assert any(col.startswith("bb_position_") for col in result.columns)
    # MACD関連列
    for c in ["macd", "macd_signal", "macd_histogram"]:
        assert c in result.columns
        assert result[c].notna().any()
