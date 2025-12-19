import pytest
import pandas as pd
import numpy as np
from app.services.ml.feature_engineering.funding_rate_features import FundingRateFeatureCalculator, validate_funding_rate_data

class TestFundingRateFeatures:
    @pytest.fixture
    def ohlcv_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "open": 100 + np.random.randn(100),
            "high": 101 + np.random.randn(100),
            "low": 99 + np.random.randn(100),
            "close": 100 + np.random.randn(100),
            "volume": 1000 * np.random.rand(100)
        }, index=dates)
        df.index.name = "timestamp"
        return df

    @pytest.fixture
    def funding_rate_data(self):
        # 8時間ごとに1レコード
        dates = pd.date_range(start="2023-01-01", periods=20, freq="8h")
        df = pd.DataFrame({
            "timestamp": dates,
            "funding_rate": [0.0001] * 10 + [0.0002] * 5 + [-0.0001] * 5
        })
        return df

    def test_calculate_features_basic(self, ohlcv_data, funding_rate_data):
        """基本的な特徴量計算のテスト"""
        calc = FundingRateFeatureCalculator()
        res = calc.calculate_features(ohlcv_data, funding_rate_data)
        
        assert "fr_bps" in res.columns
        assert "fr_dev" in res.columns
        assert "fr_cycle_sin" in res.columns
        assert "fr_ema_3p" in res.columns
        
        # bps単位への変換確認 (0.0001 -> 1.0 bps)
        assert res.iloc[0]["fr_bps"] == 1.0
        # 乖離 (1.0 - 1.0 = 0.0)
        assert res.iloc[0]["fr_dev"] == 0.0

    def test_calculate_features_empty_fr(self, ohlcv_data):
        """FRデータが空の場合のテスト"""
        calc = FundingRateFeatureCalculator()
        res = calc.calculate_features(ohlcv_data, pd.DataFrame())
        # 元のデータと同じカラム構成のはず
        assert len(res.columns) == len(ohlcv_data.columns)

    def test_validate_funding_rate_data(self, funding_rate_data):
        """バリデーションのテスト"""
        assert validate_funding_rate_data(funding_rate_data) is True
        
        # カラム欠損
        with pytest.raises(ValueError, match="必須カラムが見つかりません"):
            validate_funding_rate_data(pd.DataFrame({"timestamp": []}))
            
        # ソート不備
        unsorted = funding_rate_data.copy()
        unsorted.iloc[0, 0], unsorted.iloc[1, 0] = unsorted.iloc[1, 0], unsorted.iloc[0, 0]
        with pytest.raises(ValueError, match="タイムスタンプがソートされていません"):
            validate_funding_rate_data(unsorted)

    def test_fr_cycle_periodicity(self, ohlcv_data, funding_rate_data):
        """周期的な特徴量の検証"""
        calc = FundingRateFeatureCalculator(config={"settlement_interval": 8})
        res = calc.calculate_features(ohlcv_data, funding_rate_data)
        
        # 8時間周期で循環することを確認
        # 0時, 8時, 16時 は sin=0, cos=1 になるはず
        h0 = res.index[0].hour % 8
        if h0 == 0:
            assert pytest.approx(res.iloc[0]["fr_cycle_sin"]) == 0.0
            assert pytest.approx(res.iloc[0]["fr_cycle_cos"]) == 1.0
