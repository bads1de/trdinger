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


@pytest.fixture
def market_condition_data_factory():
    """
    特定の市場状況（急騰、急落、レンジ）の合成OHLCVデータを生成する
    データサービスオブジェクトのファクトリを返すフィクスチャ。
    """
    def _factory(kind: str, bars: int = 400):
        idx = pd.date_range("2024-01-01", periods=bars, freq="1H")
        if kind == "spike_up":
            base = np.linspace(100, 105, bars)
            base[bars // 2] += 30
        elif kind == "spike_down":
            base = np.linspace(105, 100, bars)
            base[bars // 2] -= 30
        else:  # range
            base = 150 + 2 * np.sin(np.linspace(0, 10 * np.pi, bars))
        
        open_ = np.concatenate([[base[0]], base[:-1]])
        high = np.maximum(open_, base) * 1.003
        low = np.minimum(open_, base) * 0.997
        vol = np.full(bars, 1000)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": base, "Volume": vol},
            index=idx,
        )

        class _DS:
            def get_data_for_backtest(
                self, symbol=None, timeframe=None, start_date=None, end_date=None
            ):
                return df
        
        return _DS()

    return _factory
