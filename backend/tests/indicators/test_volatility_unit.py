import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

class TestVolatilityUnitExtended:
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

    def test_atr_natr_comprehensive(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # ATR 失敗
        with patch("pandas_ta.atr", return_value=None):
            res = VolatilityIndicators.atr(h, l, c)
            assert np.isnan(res).all()
        # NATR 正常
        assert isinstance(VolatilityIndicators.natr(h, l, c), pd.Series)
        # NATR 異常
        with patch("pandas_ta.natr", return_value=None):
            assert np.isnan(VolatilityIndicators.natr(h, l, c)).all()

    def test_keltner_error_handling(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # kc 失敗
        with patch("pandas_ta.kc", return_value=None):
            res = VolatilityIndicators.keltner(h, l, c)
            assert all(np.isnan(s).all() for s in res)
        # データ不足
        res = VolatilityIndicators.keltner(h[:10], l[:10], c[:10])
        assert all(np.isnan(s).all() for s in res)

    def test_donchian_comprehensive(self, sample_df):
        h, l = sample_df["high"], sample_df["low"]
        assert len(VolatilityIndicators.donchian(h, l)) == 3
        with patch("pandas_ta.donchian", return_value=None):
            res = VolatilityIndicators.donchian(h, l)
            assert all(np.isnan(s).all() for s in res)

    def test_ui_rvi_vhf_gri(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # UI
        assert isinstance(VolatilityIndicators.ui(c), pd.Series)
        # RVI
        assert isinstance(VolatilityIndicators.rvi(c, h, l), pd.Series)
        # VHF
        assert isinstance(VolatilityIndicators.vhf(c), pd.Series)
        # GRI
        assert isinstance(VolatilityIndicators.gri(h, l, c), pd.Series)
        # GRI 失敗フォールバック
        with patch("pandas_ta.kvo", side_effect=Exception()):
            assert isinstance(VolatilityIndicators.gri(h, l, c), pd.Series)


    def test_supertrend_extended(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # 1. 正常
        res = VolatilityIndicators.supertrend(h, l, c)
        assert len(res) == 3
        # 2. 失敗 (df is None)
        with patch("pandas_ta.supertrend", return_value=None):
            res = VolatilityIndicators.supertrend(h, l, c)
            assert len(res) == 3

    def test_yang_zhang_parkinson_garman(self, sample_df):
        o, h, l, c = sample_df["open"], sample_df["high"], sample_df["low"], sample_df["close"]
        assert isinstance(VolatilityIndicators.yang_zhang(o, h, l, c), pd.Series)
        assert isinstance(VolatilityIndicators.parkinson(h, l), pd.Series)
        assert isinstance(VolatilityIndicators.garman_klass(o, h, l, c), pd.Series)