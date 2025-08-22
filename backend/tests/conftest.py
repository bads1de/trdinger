"""
テスト共通設定
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any


@pytest.fixture
def sample_ohlcv_data():
    """テスト用のサンプルOHLCVデータ"""
    np.random.seed(42)  # 再現性のため
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    # 基本的な価格データ生成
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2%のボラティリティ

    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] * (1 + change)
        close_prices.append(new_price)

    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices]
    open_prices = close_prices[:-1] + [close_prices[-1] * (1 + np.random.normal(0, 0.005))]
    volumes = np.random.uniform(1000000, 10000000, 100)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    return df


@pytest.fixture
def technical_indicator_service():
    """テクニカルインジケーターサービスのフィクスチャ"""
    from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
    return TechnicalIndicatorService()