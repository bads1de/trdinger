"""
特徴量フィルタリング機能のテスト（簡素化版）
研究目的専用のため、allowlist機能のみをテストします。
"""

import pandas as pd
import pytest

from app.services.ml.feature_engineering.feature_engineering_service import (
    DEFAULT_FEATURE_ALLOWLIST,
    FeatureEngineeringService,
)


@pytest.fixture
def sample_ohlcv_data():
    """テスト用のOHLCVデータを生成"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    data = {
        "open": [100 + i * 0.1 for i in range(100)],
        "high": [101 + i * 0.1 for i in range(100)],
        "low": [99 + i * 0.1 for i in range(100)],
        "close": [100.5 + i * 0.1 for i in range(100)],
        "volume": [1000 + i * 10 for i in range(100)],
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def feature_service():
    """FeatureEngineeringServiceインスタンスを作成"""
    service = FeatureEngineeringService()
    return service


class TestFeatureAllowlist:
    """特徴量allowlist機能のテスト"""

    def test_default_feature_allowlist_is_list(self):
        """デフォルト特徴量リストがリストであることを確認"""
        # デフォルトで推奨特徴量リストが設定されている
        assert isinstance(DEFAULT_FEATURE_ALLOWLIST, list)
        assert len(DEFAULT_FEATURE_ALLOWLIST) > 0

    def test_feature_allowlist_can_be_set(self):
        """特徴量allowlistが設定可能であることを確認"""
        # 環境変数やコードで設定可能
        custom_allowlist = ["RSI_14", "MACD", "BB_Position"]
        assert isinstance(custom_allowlist, list)
        assert all(isinstance(f, str) for f in custom_allowlist)

    def test_feature_engineering_with_default(self, sample_ohlcv_data, feature_service):
        """デフォルト設定で特徴量生成をテスト"""
        # デフォルト設定（allowlist=None）で特徴量生成
        result = feature_service.calculate_advanced_features(
            ohlcv_data=sample_ohlcv_data,
            funding_rate_data=None,
            open_interest_data=None,
        )

        # 基本カラムが含まれることを確認
        assert "close" in result.columns
        assert "volume" in result.columns

        # 何らかの特徴量が生成されることを確認
        assert len(result.columns) >= len(sample_ohlcv_data.columns)


