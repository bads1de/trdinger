"""
FeatureEngineeringService のテスト

特徴量計算のパフォーマンスと正確性を検証します。
"""

import pytest
import pandas as pd
import numpy as np
import time


class TestFeatureEngineeringServicePerformance:
    """FeatureEngineeringService のパフォーマンステスト"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用OHLCVデータ"""
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="1h")

        close = 10000 + np.cumsum(np.random.randn(n_samples) * 10)
        high = close + np.abs(np.random.randn(n_samples) * 5)
        low = close - np.abs(np.random.randn(n_samples) * 5)
        open_ = close + np.random.randn(n_samples) * 3
        volume = np.abs(np.random.randn(n_samples) * 1000) + 100

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).set_index("timestamp")

        return df

    def test_calculate_advanced_features_returns_dataframe(self, sample_ohlcv_data):
        """calculate_advanced_features がDataFrameを返すことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        # 元のカラム + 追加の特徴量カラム
        assert len(result.columns) > len(sample_ohlcv_data.columns)

    def test_calculate_advanced_features_no_duplicate_columns(self, sample_ohlcv_data):
        """calculate_advanced_features が重複カラムを持たないことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        # 重複カラムがないか確認
        assert len(result.columns) == len(set(result.columns))

    def test_calculate_advanced_features_no_inf_values(self, sample_ohlcv_data):
        """calculate_advanced_features が無限大値を含まないことを確認"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()
        result = service.calculate_advanced_features(sample_ohlcv_data)

        # 数値列に無限大がないか確認
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.any(
                np.isinf(result[col].values)
            ), f"Column {col} contains inf values"

    def test_calculate_advanced_features_performance(self, sample_ohlcv_data):
        """calculate_advanced_features のパフォーマンステスト"""
        from app.services.ml.feature_engineering.feature_engineering_service import (
            FeatureEngineeringService,
        )

        service = FeatureEngineeringService()

        # ウォームアップ（JITコンパイル等）
        _ = service.calculate_advanced_features(sample_ohlcv_data.iloc[:50])

        # キャッシュをクリア
        service.feature_cache.clear()

        start_time = time.time()
        result = service.calculate_advanced_features(sample_ohlcv_data)
        duration = time.time() - start_time

        print(f"\nProcessing {len(sample_ohlcv_data)} rows took {duration:.4f} seconds")
        print(f"Generated {len(result.columns)} features")

        # 500行で30秒以内（余裕を持たせた値）
        assert duration < 30.0, f"Processing took too long: {duration:.2f}s"
