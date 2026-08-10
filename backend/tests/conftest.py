"""
グローバルテスト設定
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tests.helpers.indicators import calculate_rsi

# Ensure backend path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_backtest_result():
    """モックバックテスト結果のフィクスチャ"""
    return {
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.5,
            "total_trades": 100,
        },
        "equity_curve": [{"drawdown": 0.05} for _ in range(100)],
        "trade_history": [
            {"size": 1, "pnl": 100} if i % 2 == 0 else {"size": -1, "pnl": 50}
            for i in range(100)
        ],
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }


@pytest.fixture
def mock_ohlcv_data():
    """モックOHLCVデータのフィクスチャ"""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    close = 50000 + np.cumsum(rng.standard_normal(1000) * 100)
    return pd.DataFrame(
        {
            "Open": close + rng.standard_normal(1000) * 50,
            "High": close + np.abs(rng.standard_normal(1000) * 100),
            "Low": close - np.abs(rng.standard_normal(1000) * 100),
            "Close": close,
            "Volume": rng.integers(100, 10000, 1000),
        },
        index=dates,
    )


@pytest.fixture
def mock_strategy(mock_ohlcv_data):
    """モック戦略インスタンスのフィクスチャ"""
    strategy = MagicMock()
    strategy.data = mock_ohlcv_data
    strategy.indicators = {
        "sma_20": mock_ohlcv_data["Close"].rolling(20).mean(),
        "ema_50": mock_ohlcv_data["Close"].ewm(span=50).mean(),
        "rsi_14": calculate_rsi(mock_ohlcv_data["Close"], 14),
    }
    return strategy
