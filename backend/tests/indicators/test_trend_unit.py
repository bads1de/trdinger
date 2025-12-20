import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.overlap import OverlapIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestTrendUnitExtended:
    @pytest.fixture
    def sample_df(self):
        rows = 100
        return pd.DataFrame(
            {
                "open": np.random.normal(100, 5, rows),
                "high": np.random.normal(105, 5, rows),
                "low": np.random.normal(95, 5, rows),
                "close": np.random.normal(100, 5, rows),
                "volume": np.random.normal(1000, 100, rows),
            },
            index=pd.date_range("2023-01-01", periods=rows),
        )

    def test_basic_ma_comprehensive(self, sample_df):
        """移動平均系のテスト - OverlapIndicatorsに移動済み"""
        c = sample_df["close"]
        # SMA - OverlapIndicatorsから
        assert isinstance(OverlapIndicators.sma(c, 20), pd.Series)
        # EMA - OverlapIndicatorsから
        assert isinstance(OverlapIndicators.ema(c, 20), pd.Series)
        # WMA - OverlapIndicatorsから
        assert isinstance(
            OverlapIndicators.wma(data=None, close=c, length=20), pd.Series
        )
        with pytest.raises(ValueError):
            OverlapIndicators.wma(None, None)

    def test_complex_ma_comprehensive(self, sample_df):
        """複雑な移動平均系のテスト - OverlapIndicatorsに移動済み"""
        c, v = sample_df["close"], sample_df["volume"]
        # ZLMA - OverlapIndicatorsから
        assert isinstance(OverlapIndicators.zlma(c, 10), pd.Series)
        with patch("pandas_ta.zlma", return_value=None):
            assert np.isnan(OverlapIndicators.zlma(c, 10)).all()
        # ALMA - OverlapIndicatorsから
        assert isinstance(OverlapIndicators.alma(c), pd.Series)
        with pytest.raises(ValueError):
            OverlapIndicators.alma(c, sigma=0)
        # VWMA - OverlapIndicatorsから
        assert isinstance(OverlapIndicators.vwma(c, v), pd.Series)

    def test_linreg_sar_amat(self, sample_df):
        """線形回帰/SAR/AMATのテスト"""
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # LINREG - OverlapIndicatorsに移動済み
        assert isinstance(OverlapIndicators.linreg(c), pd.Series)
        assert isinstance(OverlapIndicators.linregslope(c), pd.Series)
        # SAR - TrendIndicatorsに残留
        assert isinstance(TrendIndicators.sar(h, l), pd.Series)
        # AMAT - TrendIndicatorsに残留
        assert isinstance(TrendIndicators.amat(c), pd.Series)

    def test_vortex_bias_dpo(self, sample_df):
        """Vortex/Bias/DPOのテスト"""
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # Vortex - TrendIndicatorsに残留
        assert len(TrendIndicators.vortex(h, l, c)) == 2
        # Bias - MomentumIndicatorsに移動済み
        assert isinstance(MomentumIndicators.bias(c), pd.Series)
        # DPO - TrendIndicatorsに残留
        assert isinstance(TrendIndicators.dpo(c), pd.Series)
