import pytest
import pandas as pd
import numpy as np
from app.services.ml.common.volatility_utils import (
    calculate_volatility_std,
    calculate_volatility_atr,
    calculate_historical_volatility,
    calculate_realized_volatility
)

class TestVolatilityUtils:
    @pytest.fixture
    def sample_returns(self):
        return pd.Series(np.random.normal(0, 0.01, 100))

    @pytest.fixture
    def sample_ohlc(self):
        n = 50
        return pd.DataFrame({
            "high": pd.Series([102]*n),
            "low": pd.Series([98]*n),
            "close": pd.Series([100]*n)
        })

    def test_calculate_volatility_std(self, sample_returns):
        """標準偏差ベースのボラティリティ"""
        res = calculate_volatility_std(sample_returns, window=10)
        assert len(res) == 100
        assert np.isnan(res.iloc[8])
        assert not np.isnan(res.iloc[9])
        
        # 空
        assert calculate_volatility_std(pd.Series([])).empty

    def test_calculate_volatility_atr(self, sample_ohlc):
        """ATRベースのボラティリティ"""
        res = calculate_volatility_atr(sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"], window=10)
        assert len(res) == 50
        # h=102, l=98 なら TR=4
        assert pytest.approx(res.iloc[10]) == 4.0
        
        # %表示
        res_pct = calculate_volatility_atr(sample_ohlc["high"], sample_ohlc["low"], sample_ohlc["close"], as_percentage=True)
        assert pytest.approx(res_pct.iloc[15]) == 0.04 # 4/100

    def test_calculate_historical_volatility(self, sample_returns):
        """年率換算ヒストリカルボラティリティ"""
        res = calculate_historical_volatility(sample_returns, window=20, annualize=True, periods_per_year=252)
        # std * sqrt(252)
        expected = sample_returns.rolling(20).std() * np.sqrt(252)
        assert pytest.approx(res.iloc[20]) == expected.iloc[20]

    def test_calculate_realized_volatility(self, sample_returns):
        """実現ボラティリティ（日次換算）"""
        res = calculate_realized_volatility(sample_returns, window=24, periods_per_day=24)
        expected = sample_returns.rolling(24).std() * np.sqrt(24)
        assert pytest.approx(res.iloc[24]) == expected.iloc[24]
