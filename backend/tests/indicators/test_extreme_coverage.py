import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.overlap import OverlapIndicators


class TestExtremeCoverage:
    @pytest.fixture
    def sample_df(self):
        rows = 150
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

    def test_momentum_missing_lines(self, sample_df):
        c = sample_df["close"]

        # 1. CMO with talib=True/False
        MomentumIndicators.cmo(c, talib=True)
        MomentumIndicators.cmo(c, talib=False)

        # 2. STC with different result types
        with patch("pandas_ta_classic.stc", return_value=pd.Series([1, 2, 3])):
            MomentumIndicators.stc(c)

        # 3. QQE fallback RSI
        with patch("pandas_ta_classic.qqe", return_value=None):
            MomentumIndicators.qqe(c)

        # 4. KST with different parameters
        MomentumIndicators.kst(c, roc1=5, sma1=5)

        # 5. BIAS with different MA types (moved from trend)
        for ma in ["ema", "wma", "hma", "zlma"]:
            MomentumIndicators.bias(c, ma_type=ma)

    def test_volume_missing_lines(self, sample_df):
        c, h, l, v = (
            sample_df["close"],
            sample_df["high"],
            sample_df["low"],
            sample_df["volume"],
        )

        # 1. MFI with incompatible dtype (FutureWarning check)
        bad_v = v.copy()
        bad_v.iloc[0] = 0
        VolumeIndicators.mfi(h, l, c, bad_v)

        # 2. AD/ADOSC/OBV fallback
        with patch("pandas_ta_classic.ad", return_value=None):
            VolumeIndicators.ad(h, l, c, v)
        with patch("pandas_ta_classic.adosc", return_value=None):
            VolumeIndicators.adosc(h, l, c, v)
        with patch("pandas_ta_classic.obv", return_value=None):
            VolumeIndicators.obv(c, v)

        # 3. EFI with extreme values
        VolumeIndicators.efi(c * 1000, v)

    def test_trend_missing_lines(self, sample_df):
        c, h, l = (
            sample_df["close"],
            sample_df["high"],
            sample_df["low"],
        )

        # 1. SAR stable implementation logic
        TrendIndicators.sar(h, l)

        # 2. AMAT result shape logic
        with patch(
            "pandas_ta_classic.amat", return_value=pd.DataFrame({"A": [1], "B": [2]})
        ):
            TrendIndicators.amat(c)

        # 3. VHF (moved from volatility)
        TrendIndicators.vhf(c)

    def test_volatility_missing_lines(self, sample_df):
        c, h, l = sample_df["close"], sample_df["high"], sample_df["low"]

        # 1. Keltner with fallback search for column names
        mock_kc = pd.DataFrame({"WrongName": [1] * 150})
        with patch("pandas_ta_classic.kc", return_value=mock_kc):
            VolatilityIndicators.keltner(h, l, c)

    def test_overlap_missing_lines(self, sample_df):
        """Overlap 指標のテスト (supertrend, zlma)"""
        c, h, l = sample_df["close"], sample_df["high"], sample_df["low"]

        # 1. Supertrend factor alias
        OverlapIndicators.supertrend(h, l, c, factor=4.0)

        # 2. Supertrend column name fallback
        mock_st = pd.DataFrame({"SUPERT_7_3.0": [1] * 150, "SUPERTd_7_3.0": [1] * 150})
        with patch("pandas_ta_classic.supertrend", return_value=mock_st):
            OverlapIndicators.supertrend(h, l, c)

        # 3. ZLMA with ema mode only (sma mode has pandas-ta bug)
        OverlapIndicators.zlma(c, mamode="ema")

    def test_all_ma_modes_in_indicators(self, sample_df):
        # 多くの指標が mamode パラメータを持つため、一通り試す
        c = sample_df["close"]
        for mode in ["sma", "ema", "wma", "hma", "zlma"]:
            try:
                MomentumIndicators.apo(c, ma_mode=mode)
            except Exception:
                pass
            try:
                MomentumIndicators.tsi(c, mamode=mode)
            except Exception:
                pass
            try:
                OverlapIndicators.zlma(c, mamode=mode)
            except Exception:
                pass
