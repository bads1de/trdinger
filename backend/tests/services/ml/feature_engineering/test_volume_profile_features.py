import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.volume_profile_features import VolumeProfileFeatureCalculator

class TestVolumeProfileFeatures:
    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用OHLCVデータを生成"""
        np.random.seed(42)
        n = 300
        dates = pd.date_range(start="2023-01-01", periods=n, freq="h")
        
        prices = np.random.normal(105, 2, n)
        highs = prices + 0.5
        lows = prices - 0.5
        volumes = np.random.rand(n) * 100
        volumes[100:200] *= 10
        
        df = pd.DataFrame({
            "open": prices, "high": highs, "low": lows, "close": prices, "volume": volumes
        }, index=dates)
        return df

    def test_calculate_features_smoke(self, sample_ohlcv):
        """基本的な特徴量計算の動作確認"""
        calc = VolumeProfileFeatureCalculator(lookback_period=50, num_bins=10)
        res = calc.calculate_features(sample_ohlcv)
        
        assert "POC_Distance_50" in res.columns
        assert "VAH_Distance_50" in res.columns
        assert "In_Value_Area_50" in res.columns
        assert "HVN_Distance" in res.columns
        assert "VP_Skewness" in res.columns
        assert "VP_Kurtosis" in res.columns
        assert not res.isnull().any().any()

    def test_values_correctness_basic(self, sample_ohlcv):
        """計算値の妥当性"""
        calc = VolumeProfileFeatureCalculator(lookback_period=100, num_bins=20)
        res = calc.calculate_features(sample_ohlcv, lookback_periods=[100])
        assert (res["POC_Distance_100"].abs() < 1.0).all()

    def test_zero_volume_handling(self):
        """出来高0の場合のハンドリング"""
        n = 100
        df = pd.DataFrame({
            "high": [110]*n, "low": [90]*n, "close": [100]*n, "volume": [0.0]*n
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        
        calc = VolumeProfileFeatureCalculator()
        res = calc.calculate_features(df)
        assert not res.isnull().any().any()

    def test_zero_price_range_handling(self):
        """価格が動かない場合のハンドリング"""
        n = 100
        df = pd.DataFrame({
            "high": [100]*n, "low": [100]*n, "close": [100]*n, "volume": [100]*n
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        
        calc = VolumeProfileFeatureCalculator()
        res = calc.calculate_features(df)
        assert not res.isnull().any().any()

    def test_vp_shape_features(self, sample_ohlcv):
        """歪度と尖度の計算テスト"""
        calc = VolumeProfileFeatureCalculator(lookback_period=50)
        res = calc.calculate_features(sample_ohlcv)
        assert (res["VP_Skewness"] != 0).any()
        assert (res["VP_Kurtosis"] != 0).any()

    def test_extreme_price_spike(self):
        """極端な価格スパイク時のハンドリング"""
        n = 100
        prices = [100.0] * n
        prices[50] = 1000.0
        df = pd.DataFrame({
            "high": prices, "low": prices, "close": prices, "volume": [100.0]*n
        }, index=pd.date_range("2023-01-01", periods=n, freq="h"))
        
        calc = VolumeProfileFeatureCalculator(lookback_period=20)
        res = calc.calculate_features(df, lookback_periods=[20])
        assert not res.isnull().any().any()

    def test_various_lookback_periods(self, sample_ohlcv):
        """複数期間での一括計算"""
        calc = VolumeProfileFeatureCalculator()
        res = calc.calculate_features(sample_ohlcv)
        assert "POC_Distance_50" in res.columns
        assert "POC_Distance_100" in res.columns
        assert "POC_Distance_200" in res.columns

    def test_empty_input_handling(self):
        """空のDataFrameに対する挙動"""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        calc = VolumeProfileFeatureCalculator()
        res = calc.calculate_features(df)
        assert res.empty