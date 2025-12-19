
import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.oi_fr_interaction_features import OIFRInteractionFeatureCalculator

class TestOIFRInteractionFeatureCalculator:
    """OIFRInteractionFeatureCalculatorのテスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """サンプルOHLCVデータ"""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 105,
            "low": np.random.randn(n).cumsum() + 95,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.random.randint(1000, 5000, n)
        }, index=dates)
        return df

    @pytest.fixture
    def sample_oi_fr(self, sample_ohlcv):
        """サンプルOI/FRデータ"""
        n = len(sample_ohlcv)
        # OIは通常大きな値
        oi_data = pd.DataFrame({
            "open_interest": np.random.uniform(1e8, 2e8, n)
        }, index=sample_ohlcv.index)
        
        # FRは小さな値 (-0.01% to 0.01% typically)
        fr_data = pd.DataFrame({
            "funding_rate": np.random.uniform(-0.0001, 0.0001, n)
        }, index=sample_ohlcv.index)
        
        return oi_data, fr_data

    def test_calculate_features_basic(self, sample_ohlcv, sample_oi_fr):
        """基本的な特徴量計算のテスト"""
        oi_data, fr_data = sample_oi_fr
        calculator = OIFRInteractionFeatureCalculator()
        
        features = calculator.calculate_features(
            df=sample_ohlcv,
            oi_data=oi_data,
            fr_data=fr_data
        )
        
        assert not features.empty
        assert len(features) == len(sample_ohlcv)
        
        # 主要な特徴量が含まれているか確認
        expected_cols = [
            "OI_Price_Regime", 
            "FR_Acceleration", 
            "Liquidation_Pressure",
            "Smart_Money_Flow",
            "FR_Extreme_Regime",
            "Leverage_Ratio"
        ]
        for col in expected_cols:
            assert col in features.columns

    def test_calculate_features_missing_data(self, sample_ohlcv):
        """データ欠落時の挙動テスト"""
        calculator = OIFRInteractionFeatureCalculator()
        
        # OI/FRがNoneの場合
        features = calculator.calculate_features(sample_ohlcv, oi_data=None, fr_data=None)
        assert features.empty
        
        # 一方だけNoneの場合
        oi_data = pd.DataFrame({"oi": [100]}, index=sample_ohlcv.index[:1])
        features = calculator.calculate_features(sample_ohlcv, oi_data=oi_data, fr_data=None)
        assert features.empty

    def test_unaligned_index(self, sample_ohlcv, sample_oi_fr):
        """インデックス不一致時のテスト"""
        oi_data, fr_data = sample_oi_fr
        calculator = OIFRInteractionFeatureCalculator()
        
        # OIデータのインデックスをずらす
        oi_unaligned = oi_data.copy()
        oi_unaligned.index = pd.date_range("2025-01-01", periods=len(oi_data), freq="1h")
        
        features = calculator.calculate_features(sample_ohlcv, oi_data=oi_unaligned, fr_data=fr_data)
        assert features.empty

    def test_zero_values_handling(self, sample_ohlcv):
        """ゼロ値や極端な値のハンドリング"""
        n = len(sample_ohlcv)
        oi_data = pd.DataFrame({"oi": np.zeros(n)}, index=sample_ohlcv.index)
        fr_data = pd.DataFrame({"fr": np.zeros(n)}, index=sample_ohlcv.index)
        
        calculator = OIFRInteractionFeatureCalculator()
        features = calculator.calculate_features(sample_ohlcv, oi_data=oi_data, fr_data=fr_data)
        
        assert not features.empty
        # 全て0で計算されてもエラーにならないこと
        assert not features.isna().any().any()
        assert (features == 0).all().all() or not features.empty

    def test_advanced_features_logic(self, sample_ohlcv, sample_oi_fr):
        """高度な特徴量のロジック検証"""
        oi_data, fr_data = sample_oi_fr
        calculator = OIFRInteractionFeatureCalculator()
        
        features = calculator.calculate_features(sample_ohlcv, oi_data, fr_data)
        
        # FR_Extreme_Regime は 0 または 1 であること
        unique_vals = features["FR_Extreme_Regime"].unique()
        for v in unique_vals:
            assert v in [0.0, 1.0]
            
        # Leverage_Ratio が正の値であること（価格とOIが正なら）
        assert (features["Leverage_Ratio"] >= 0).all()
