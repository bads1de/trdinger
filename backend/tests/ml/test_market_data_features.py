import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.market_data_features import (
    MarketDataFeatureCalculator,
)


class TestMarketDataFeatures:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.random.rand(200) * 100,
                "high": np.random.rand(200) * 100,
                "low": np.random.rand(200) * 100,
                "close": np.random.rand(200) * 100,
                "volume": np.random.rand(200) * 1000,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def sample_oi_data(self, sample_data):
        # 増加傾向のあるOIデータを生成してRSIなどが計算できるようにする
        oi_values = np.linspace(1000, 2000, 200) + np.random.randn(200) * 50
        oi_df = pd.DataFrame({"open_interest": oi_values}, index=sample_data.index)
        return oi_df

    @pytest.fixture
    def sample_fr_data(self, sample_data):
        fr_df = pd.DataFrame(
            {"funding_rate": np.random.uniform(-0.01, 0.01, 200)},
            index=sample_data.index,
        )
        return fr_df

    def test_calculate_open_interest_features(self, sample_data, sample_oi_data):
        calculator = MarketDataFeatureCalculator()
        lookback_periods = {"short": 10}

        result = calculator.calculate_open_interest_features(
            sample_data, sample_oi_data, lookback_periods
        )

        expected_cols = [
            "OI_RSI",
            "Volume_OI_Ratio",
        ]

        for col in expected_cols:
            assert col in result.columns
            # 最初の数行はNaNになる可能性があるため、全体がNaNでないことを確認
            assert not result[col].isnull().all()
            # 欠損値処理が行われているか確認（fillnaされているはず）
            assert result[col].isnull().sum() == 0

    def test_calculate_pseudo_open_interest_features(self, sample_data):
        calculator = MarketDataFeatureCalculator()
        lookback_periods = {"short": 10}

        result = calculator.calculate_pseudo_open_interest_features(
            sample_data, lookback_periods
        )

        expected_cols = [
            "OI_RSI",
            "Volume_OI_Ratio",
        ]

        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().all()

    def test_calculate_funding_rate_features(self, sample_data, sample_fr_data):
        calculator = MarketDataFeatureCalculator()
        lookback_periods = {"short": 10}

        result = calculator.calculate_funding_rate_features(
            sample_data, sample_fr_data, lookback_periods
        )

        expected_cols = [
            "FR_Extremity_Zscore",
            "FR_Momentum",
        ]

        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().all()

    def test_calculate_composite_features(
        self, sample_data, sample_oi_data, sample_fr_data
    ):
        """複合特徴量のテスト"""
        calculator = MarketDataFeatureCalculator()
        lookback_periods = {"short": 10}

        # 先にFR特徴量を計算しておく必要がある（Market_Stress計算のため）
        # 実際のフローではcalculate_featuresメソッド内で順序制御されるが、
        # ユニットテストでは個別に呼ぶため、FR_Extremity_Zscoreが存在しない場合の挙動もテストされる

        result = calculator.calculate_composite_features(
            sample_data, sample_fr_data, sample_oi_data, lookback_periods
        )

        expected_cols = [
            "FR_Cumulative_Trend",
            "Market_Stress",
        ]

        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().all()
            assert result[col].isnull().sum() == 0

    def test_get_feature_names(self):
        calculator = MarketDataFeatureCalculator()
        names = calculator.get_feature_names()

        expected_names = [
            "OI_RSI",
            "Volume_OI_Ratio",
            "FR_Cumulative_Trend",
            "FR_Extremity_Zscore",
            "FR_Momentum",
            "Market_Stress",
        ]

        for name in expected_names:
            assert name in names


