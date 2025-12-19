import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.market_data_features import MarketDataFeatureCalculator

class TestMarketDataFeatureCalculator:
    @pytest.fixture
    def sample_ohlcv(self):
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "open": np.random.randn(200) + 100,
            "high": np.random.randn(200) + 101,
            "low": np.random.randn(200) + 99,
            "close": np.random.randn(200) + 100,
            "volume": np.random.rand(200) * 1000
        }, index=dates)
        return df

    @pytest.fixture
    def sample_fr(self):
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        return pd.DataFrame({
            "timestamp": dates,
            "funding_rate": [0.0001] * 100 + [0.0005] * 100
        })

    @pytest.fixture
    def sample_oi(self):
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        return pd.DataFrame({
            "timestamp": dates,
            "open_interest": np.linspace(1000000, 2000000, 200)
        })

    def test_calculate_features_all(self, sample_ohlcv, sample_fr, sample_oi):
        """FRとOIの両方がある場合の計算"""
        calc = MarketDataFeatureCalculator()
        config = {
            "funding_rate_data": sample_fr,
            "open_interest_data": sample_oi
        }
        res = calc.calculate_features(sample_ohlcv, config)
        
        # 主要なカラムが含まれているか
        assert "FR_Extremity_Zscore" in res.columns
        assert "OI_RSI" in res.columns
        assert "Market_Stress" in res.columns
        assert not res.isnull().all().all()

    def test_funding_rate_features_logic(self, sample_ohlcv, sample_fr):
        """FR特徴量のロジック検証"""
        calc = MarketDataFeatureCalculator()
        res = calc.calculate_funding_rate_features(sample_ohlcv, sample_fr, {})
        
        assert "FR_MA_24" in res.columns
        assert "FR_MACD" in res.columns
        # Z-scoreの計算が妥当か (最初の100件は一定なので0になるはず)
        assert res.iloc[50]["FR_Extremity_Zscore"] == 0.0

    def test_open_interest_features_logic(self, sample_ohlcv, sample_oi):
        """OI特徴量のロジック検証"""
        calc = MarketDataFeatureCalculator()
        res = calc.calculate_open_interest_features(sample_ohlcv, sample_oi, {})
        
        assert "OI_RSI" in res.columns
        assert "OI_Trend_Strength" in res.columns
        assert "Volume_OI_Ratio" in res.columns

    def test_pseudo_oi_features(self, sample_ohlcv):
        """疑似OI特徴量のテスト"""
        calc = MarketDataFeatureCalculator()
        res = calc.calculate_pseudo_open_interest_features(sample_ohlcv, {})
        
        assert "OI_RSI" in res.columns
        assert "OI_MACD" in res.columns

    def test_timezone_alignment(self, sample_ohlcv, sample_fr):
        """タイムゾーンが異なる場合のアライメント"""
        calc = MarketDataFeatureCalculator()
        
        # OHLCVをUTCに、FRをタイムゾーンなしにする
        sample_ohlcv.index = sample_ohlcv.index.tz_localize("UTC")
        # _process_market_data が内部で適切に処理することを確認
        res, col = calc._process_market_data(sample_ohlcv, sample_fr, ["funding_rate"], "_fr")
        
        assert res.index.tz is not None
        assert col == "funding_rate_fr"
        assert not res["funding_rate_fr"].isnull().all()

    def test_missing_columns_fallback(self, sample_ohlcv):
        """期待されるカラムがない場合のフォールバック"""
        calc = MarketDataFeatureCalculator()
        bad_fr = pd.DataFrame({"timestamp": sample_ohlcv.index, "wrong_col": [0.1]*200})
        
        res = calc.calculate_funding_rate_features(sample_ohlcv, bad_fr, {})
        # カラムが見つからない場合は元のデータをそのまま返すはず
        assert "FR_MA_24" not in res.columns
        pd.testing.assert_frame_equal(res, sample_ohlcv)

    def test_composite_features_logic(self, sample_ohlcv, sample_fr, sample_oi):
        """複合特徴量のロジック検証"""
        calc = MarketDataFeatureCalculator()
        res = calc.calculate_composite_features(sample_ohlcv, sample_fr, sample_oi, {})
        
        assert "Market_Stress" in res.columns
        assert "FR_OI_Sentiment" in res.columns
        assert "OI_Weighted_Price_Dev" in res.columns

    def test_get_feature_names(self):
        """特徴量名リストの取得"""
        calc = MarketDataFeatureCalculator()
        names = calc.get_feature_names()
        assert "OI_RSI" in names
        assert "Market_Stress" in names
