import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.trend import TrendIndicators

class TestTrendUnitExtended:
    @pytest.fixture
    def sample_df(self):
        rows = 100
        return pd.DataFrame({
            "open": np.random.normal(100, 5, rows),
            "high": np.random.normal(105, 5, rows),
            "low": np.random.normal(95, 5, rows),
            "close": np.random.normal(100, 5, rows),
            "volume": np.random.normal(1000, 100, rows)
        }, index=pd.date_range("2023-01-01", periods=rows))

    def test_basic_ma_comprehensive(self, sample_df):
        c = sample_df["close"]
        # SMA
        assert isinstance(TrendIndicators.sma(c, 20), pd.Series)
        # EMA
        assert isinstance(TrendIndicators.ema(c, 20), pd.Series)
        # WMA
        assert isinstance(TrendIndicators.wma(data=None, close=c, length=20), pd.Series)
        with pytest.raises(ValueError): TrendIndicators.wma(None, None)

    def test_complex_ma_comprehensive(self, sample_df):
        c, v = sample_df["close"], sample_df["volume"]
        # ZLMA
        assert isinstance(TrendIndicators.zlma(c, 10), pd.Series)
        with patch("pandas_ta.zlma", return_value=None):
            assert np.isnan(TrendIndicators.zlma(c, 10)).all()
        # ALMA
        assert isinstance(TrendIndicators.alma(c), pd.Series)
        with pytest.raises(ValueError): TrendIndicators.alma(c, sigma=0)
        # VWMA
        assert isinstance(TrendIndicators.vwma(c, v), pd.Series)

    def test_linreg_sar_amat(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # LINREG
        assert isinstance(TrendIndicators.linreg(c), pd.Series)
        assert isinstance(TrendIndicators.linregslope(c), pd.Series)
        # SAR
        assert isinstance(TrendIndicators.sar(h, l), pd.Series)
        # AMAT
        assert isinstance(TrendIndicators.amat(c), pd.Series)

    def test_vortex_bias_dpo(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # Vortex
        assert len(TrendIndicators.vortex(h, l, c)) == 2
        # Bias
        assert isinstance(TrendIndicators.bias(c), pd.Series)
        # DPO
        assert isinstance(TrendIndicators.dpo(c), pd.Series)