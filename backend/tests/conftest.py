"""
pytest設定ファイル

テスト全体で共有するフィクスチャとセットアップを定義します。
"""

import sys
import os

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "core"))
)

import pytest
import asyncio
from typing import Generator
from unittest.mock import Mock

from app.config.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """セッション全体で使用するイベントループ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_market_data_service():
    """モックされた市場データサービス"""
    service = Mock()  # spec=BybitMarketDataService
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
        "strategy_name": "GENERATED_TEST",
        "timeframe": "1d",
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "GENERATED_TEST",
            "parameters": {"strategy_gene": {}},
        },
    }


# === pytest マーカー設定 ===


def pytest_configure(config):
    """pytest設定"""
    # カスタムマーカーの登録
    config.addinivalue_line("markers", "unit: 単体テスト")
    config.addinivalue_line("markers", "integration: 統合テスト")
    config.addinivalue_line("markers", "e2e: エンドツーエンドテスト")
    config.addinivalue_line("markers", "slow: 実行時間の長いテスト")
    config.addinivalue_line("markers", "market_validation: 市場検証テスト")
    config.addinivalue_line("markers", "performance: パフォーマンステスト")
    config.addinivalue_line("markers", "security: セキュリティテスト")


# === テストカテゴリ別実行用のフィクスチャ ===


@pytest.fixture(scope="session")
def test_categories():
    """テストカテゴリ情報"""
    return {
        "unit": "単体テスト - 個別コンポーネントの動作確認",
        "integration": "統合テスト - コンポーネント間の連携確認",
        "e2e": "エンドツーエンドテスト - 完全なワークフロー確認",
        "slow": "時間のかかるテスト - パフォーマンス・負荷テスト",
        "market_validation": "市場検証テスト - 実際の市場データでの検証",
        "performance": "パフォーマンステスト - 実行時間・メモリ使用量の確認",
        "security": "セキュリティテスト - セキュリティ要件の確認",
    }


@pytest.fixture
def test_execution_summary():
    """テスト実行サマリー用フィクスチャ"""
    return {
        "start_time": None,
        "end_time": None,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "categories": {},
    }
