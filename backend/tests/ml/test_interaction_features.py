"""
相互作用特徴量計算のテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering.interaction_features import (
    InteractionFeatureCalculator,
)


class TestInteractionFeatureCalculator:
    """相互作用特徴量計算クラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 105,
                "low": np.random.randn(n).cumsum() + 95,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, n),
                # 基本特徴量
                "Momentum": np.random.randn(n),
                "Price_Change_5": np.random.randn(n),
                "RSI": np.random.uniform(0, 100, n),
                "ATR": np.random.uniform(0.5, 2.0, n),
                "ATR_20": np.random.uniform(0.5, 2.0, n),
            }
        )

        df.index = pd.date_range("2024-01-01", periods=n, freq="1H")
        return df

    def test_calculate_interaction_features_basic(self, sample_data):
        """基本的な相互作用特徴量計算のテスト（現在は何もしない）"""
        calculator = InteractionFeatureCalculator()
        result_df = calculator.calculate_interaction_features(sample_data)

        # カラム数が増えていないことを確認（現在は無効化されているため）
        assert len(result_df.columns) == len(sample_data.columns)

        # 元のデータが保持されているか
        pd.testing.assert_frame_equal(result_df, sample_data)

    def test_empty_data(self):
        """空のデータのテスト"""
        calculator = InteractionFeatureCalculator()
        empty_df = pd.DataFrame()
        result_df = calculator.calculate_interaction_features(empty_df)

        assert result_df.empty

    def test_get_feature_names(self):
        """特徴量名リスト取得のテスト"""
        calculator = InteractionFeatureCalculator()
        feature_names = calculator.get_feature_names()

        assert isinstance(feature_names, list)
        # 現在は空リストを返す
        assert len(feature_names) == 0
