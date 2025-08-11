import pytest
import numpy as np
import pandas as pd
import sys
import os

# Ensure 'backend' directory is on sys.path so that 'import app.*' works
_backend_dir = os.path.dirname(os.path.dirname(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """OHLCVに近い構造のサンプル価格データ（Closeのみ必須）"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="1H")
    base_price = 50000.0
    returns = np.random.normal(0, 0.002, 1000)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({"timestamp": dates, "Close": prices}).set_index("timestamp")
    return df


@pytest.fixture
def known_price_changes() -> pd.DataFrame:
    """明確なパターンを含む価格データのフィクスチャ"""
    prices = [
        100,
        102,
        104,
        106,
        108,  # 上昇 (+2%)
        108,
        106,
        104,
        102,
        100,  # 下降 (-2%)
        100,
        100.5,
        99.5,
        100,
        100.2,  # 横ばい (±0.5%)
        100,
        105,
        110,
        115,
        120,  # 強い上昇 (+5%)
        120,
        114,
        108,
        102,
        96,  # 強い下降 (-5%)
    ]
    dates = pd.date_range("2023-01-01", periods=len(prices), freq="1H")
    return pd.DataFrame({"timestamp": dates, "Close": prices}).set_index("timestamp")
