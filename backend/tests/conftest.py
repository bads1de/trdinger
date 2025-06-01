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


# === 実際のDBデータを使用したバックテストテスト用の追加設定 ===


@pytest.fixture(scope="session")
def test_database_connection():
    """データベース接続テスト"""
    try:
        from database.connection import SessionLocal

        db = SessionLocal()

        # 簡単な接続テスト
        result = db.execute("SELECT 1")
        assert result.fetchone()[0] == 1

        db.close()
        return True
    except Exception as e:
        pytest.skip(f"データベース接続に失敗: {e}")


@pytest.fixture(scope="session")
def verify_real_data_availability():
    """実際のデータの存在確認"""
    try:
        from database.connection import SessionLocal
        from database.repositories.ohlcv_repository import OHLCVRepository

        db = SessionLocal()
        repo = OHLCVRepository(db)

        # BTC/USDTの1dデータが存在するかチェック
        count = repo.count_records("BTC/USDT", "1d")

        db.close()

        if count < 30:  # 最低30日分のデータが必要
            pytest.skip(f"テストに十分なデータがありません: {count}件")

        return count
    except Exception as e:
        pytest.skip(f"データ確認に失敗: {e}")


@pytest.fixture
def performance_thresholds():
    """パフォーマンステストの閾値設定"""
    return {
        "small_dataset_time": 5.0,  # 小規模データセット: 5秒以内
        "medium_dataset_time": 10.0,  # 中規模データセット: 10秒以内
        "large_dataset_time": 30.0,  # 大規模データセット: 30秒以内
        "optimization_time": 60.0,  # 最適化: 60秒以内
        "small_dataset_memory": 100,  # 小規模データセット: 100MB以内
        "medium_dataset_memory": 200,  # 中規模データセット: 200MB以内
        "large_dataset_memory": 500,  # 大規模データセット: 500MB以内
        "optimization_memory": 300,  # 最適化: 300MB以内
        "memory_leak_threshold": 50,  # メモリリーク閾値: 50MB
    }


@pytest.fixture
def common_backtest_config():
    """共通のバックテスト設定"""
    return {
        "strategy_name": "SMA_CROSS",
        "timeframe": "1d",
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {"n1": 20, "n2": 50},
        },
    }
