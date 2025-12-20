import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.volume import VolumeIndicators


class TestVolumeUnitExtended:
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

    def test_ad_adosc_obv(self, sample_df):
        h, l, c, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["volume"],
        )
        # 1. 正常
        assert isinstance(VolumeIndicators.ad(h, l, c, v), pd.Series)
        assert isinstance(VolumeIndicators.adosc(h, l, c, v), pd.Series)
        assert isinstance(VolumeIndicators.obv(c, v), pd.Series)
        # 2. 失敗フォールバック
        with patch("pandas_ta.ad", return_value=None):
            assert (
                len(VolumeIndicators.ad(h, l, c, v)) == 0
            )  # or NaN Series depending on implementation
        with patch("pandas_ta.adosc", return_value=None):
            assert len(VolumeIndicators.adosc(h, l, c, v)) == 0

    def test_eom_comprehensive(self, sample_df):
        h, l, c, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["volume"],
        )
        # 1. 正常
        assert isinstance(VolumeIndicators.eom(h, l, c, v), pd.Series)
        # 2. 失敗時 (None or Empty)
        with patch("pandas_ta.eom", return_value=None):
            res = VolumeIndicators.eom(h, l, c, v)
            assert np.isnan(res).all()

    def test_vwap_comprehensive(self, sample_df):
        h, l, c, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["volume"],
        )
        # 1. 正常 (pandas-ta)
        assert isinstance(VolumeIndicators.vwap(h, l, c, v), pd.Series)
        # 2. フォールバック (DatetimeIndexなし)
        c_no_idx = c.reset_index(drop=True)
        h_no_idx = h.reset_index(drop=True)
        l_no_idx = l.reset_index(drop=True)
        v_no_idx = v.reset_index(drop=True)
        res = VolumeIndicators.vwap(h_no_idx, l_no_idx, c_no_idx, v_no_idx)
        assert isinstance(res, pd.Series)
        assert not res.isna().all()

    def test_cmf_comprehensive(self, sample_df):
        h, l, c, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["volume"],
        )
        # 1. 正常
        assert isinstance(VolumeIndicators.cmf(h, l, c, v), pd.Series)
        # 2. 非数値データの混入
        bad_c = c.astype(object)
        bad_c.iloc[0] = "error"
        with pytest.raises(ValueError):
            VolumeIndicators.cmf(h, l, bad_c, v)

    def test_efi_comprehensive(self, sample_df):
        c, v = sample_df["close"], sample_df["volume"]
        # 1. 正常
        assert isinstance(VolumeIndicators.efi(c, v), pd.Series)
        # 2. 極端な値のクリッピング
        bad_c = c.copy()
        bad_c.iloc[10] = 1e12  # 異常値
        res = VolumeIndicators.efi(bad_c, v)
        assert not np.isinf(res).any()

    def test_pvo_comprehensive(self, sample_df):
        v = sample_df["volume"]
        # 1. 正常
        res = VolumeIndicators.pvo(v)
        assert len(res) == 3
        # 2. 失敗
        with patch("pandas_ta.pvo", return_value=None):
            res = VolumeIndicators.pvo(v)
            assert all(np.isnan(s).all() for s in res)

    def test_pvt_nvi_comprehensive(self, sample_df):
        from app.services.indicators.data_validation import PandasTAError

        c, v = sample_df["close"], sample_df["volume"]
        # PVT 失敗 (全NaNはエラーになる)
        with patch("pandas_ta.pvt", return_value=None):
            with pytest.raises(PandasTAError):
                VolumeIndicators.pvt(c, v)
        # NVI 失敗
        with patch("pandas_ta.nvi", return_value=None):
            with pytest.raises(PandasTAError):
                VolumeIndicators.nvi(c, v)

    def test_rvol_extended(self, sample_df):
        v = sample_df["volume"]
        # 1. DatetimeIndexあり
        assert isinstance(VolumeIndicators.rvol(v), pd.Series)
        # 2. DatetimeIndexなし
        assert isinstance(VolumeIndicators.rvol(v.reset_index(drop=True)), pd.Series)
        # 3. 例外発生時のフォールバック
        with patch.object(pd.Series, "groupby", side_effect=Exception()):
            assert isinstance(VolumeIndicators.rvol(v), pd.Series)
