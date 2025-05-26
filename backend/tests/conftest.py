"""
pytest設定ファイル

テスト全体で共有するフィクスチャとセットアップを定義します。
"""

import pytest
import asyncio
from typing import Generator
from unittest.mock import Mock

from app.config.settings import settings
from app.core.services.market_data_service import BybitMarketDataService


@pytest.fixture(scope="session")
def event_loop():
    """セッション全体で使用するイベントループ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_market_data_service():
    """モックされた市場データサービス"""
    service = Mock(spec=BybitMarketDataService)
    service.fetch_ohlcv_data.return_value = [
        [1640995200000, 47000.0, 48000.0, 46500.0, 47500.0, 1000.0],
        [1641081600000, 47500.0, 48500.0, 47000.0, 48000.0, 1200.0],
    ]
    service.normalize_symbol.return_value = "BTC/USD:BTC"
    return service


@pytest.fixture
def sample_ohlcv_data():
    """サンプルOHLCVデータ"""
    return [
        [1640995200000, 47000.0, 48000.0, 46500.0, 47500.0, 1000.0],
        [1641081600000, 47500.0, 48500.0, 47000.0, 48000.0, 1200.0],
        [1641168000000, 48000.0, 49000.0, 47500.0, 48500.0, 1100.0],
        [1641254400000, 48500.0, 49500.0, 48000.0, 49000.0, 1300.0],
        [1641340800000, 49000.0, 50000.0, 48500.0, 49500.0, 1400.0],
    ]


@pytest.fixture
def test_settings():
    """テスト用設定"""
    test_settings = settings.copy()
    test_settings.debug = True
    test_settings.database_url = "sqlite:///:memory:"
    return test_settings
