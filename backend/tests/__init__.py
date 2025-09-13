"""
テストスイート統合インターフェース

整理されたテスト構造を提供し、各テストモジュールを統一的に利用可能にします。
"""

# テスト設定
import pytest

# 共通フィクスチャ
@pytest.fixture(scope="session")
def test_config():
    """テスト設定"""
    return {
        "test_data_path": "backend/tests/test_data",
        "mock_services": True,
        "performance_threshold": 10.0  # seconds
    }

# ユーティリティ関数
def get_test_data(filename):
    """テストデータの取得"""
    import os
    return os.path.join(os.path.dirname(__file__), "test_data", filename)

def assert_performance_under_threshold(duration, threshold=10.0):
    """パフォーマンスアサーション"""
    assert duration < threshold, ".2f"

def setup_test_logger():
    """テスト用ロガー設定"""
    import logging
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    return logger

__all__ = [
    "test_config",
    "get_test_data",
    "assert_performance_under_threshold",
    "setup_test_logger"
]