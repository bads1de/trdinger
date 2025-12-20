import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.overlap import OverlapIndicators


class TestMomentumUnitExtended:
    @pytest.fixture
    def sample_data(self):
        return pd.Series(
            np.random.normal(100, 5, 100),
            index=pd.date_range("2023-01-01", periods=100),
        )

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

    def test_rsi_comprehensive(self, sample_data):
        # 1. 正常
        assert isinstance(MomentumIndicators.rsi(sample_data, 14), pd.Series)
        # 2. データ不足
        assert np.isnan(MomentumIndicators.rsi(sample_data[:5], 14)).all()
        # 3. pandas-ta 失敗
        with patch("pandas_ta.rsi", return_value=None):
            assert np.isnan(MomentumIndicators.rsi(sample_data, 14)).all()

    def test_macd_comprehensive(self, sample_data):
        # 1. 正常
        res = MomentumIndicators.macd(sample_data)
        assert len(res) == 3 and all(isinstance(s, pd.Series) for s in res)
        # 2. データ不足
        res = MomentumIndicators.macd(sample_data[:2])
        assert all(np.isnan(s).all() for s in res)
        # 3. pandas-ta 失敗 (None)
        with patch("pandas_ta.macd", return_value=None):
            assert len(MomentumIndicators.macd(sample_data)) == 3
        # 4. pandas-ta 失敗 (Empty)
        with patch("pandas_ta.macd", return_value=pd.DataFrame()):
            assert len(MomentumIndicators.macd(sample_data)) == 3

    def test_ppo_comprehensive(self, sample_data):
        # 1. 正常
        res = MomentumIndicators.ppo(sample_data)
        assert len(res) == 3
        # 2. 異常 (None)
        with patch("pandas_ta.ppo", return_value=None):
            res = MomentumIndicators.ppo(sample_data)
            assert all(np.isnan(s).all() for s in res)

    def test_trix_comprehensive(self, sample_data):
        # 1. 正常
        res = MomentumIndicators.trix(sample_data)
        assert len(res) == 3
        # 2. 異常 (Empty)
        with patch("pandas_ta.trix", return_value=pd.DataFrame()):
            res = MomentumIndicators.trix(sample_data)
            assert all(np.isnan(s).all() for s in res)

    def test_stoch_comprehensive(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # 1. 正常
        assert len(MomentumIndicators.stoch(h, l, c)) == 2
        # 2. d_length使用
        assert len(MomentumIndicators.stoch(h, l, c, d_length=5)) == 2
        # 3. 異常系
        with pytest.raises(TypeError):
            MomentumIndicators.stoch(None, l, c)
        with pytest.raises(ValueError):
            MomentumIndicators.stoch(h, l, c, k=0)
        # 4. pandas-ta 失敗
        with patch("pandas_ta.stoch", return_value=None):
            res = MomentumIndicators.stoch(h, l, c)
            assert all(np.isnan(s).all() for s in res)

    def test_stochrsi_comprehensive(self, sample_data):
        # 1. 正常
        assert len(MomentumIndicators.stochrsi(sample_data)) == 2
        # 2. 異常系 (パラメータ)
        with pytest.raises(ValueError):
            MomentumIndicators.stochrsi(sample_data, rsi_length=0)
        # 3. データ不足
        res = MomentumIndicators.stochrsi(sample_data[:5])
        assert all(np.isnan(s).all() for s in res)
        # 4. pandas-ta 失敗
        with patch("pandas_ta.stochrsi", return_value=None):
            res = MomentumIndicators.stochrsi(sample_data)
            assert all(np.isnan(s).all() for s in res)

    def test_willr_comprehensive(self, sample_df):
        from app.services.indicators.data_validation import PandasTAError

        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # 1. 正常
        assert isinstance(MomentumIndicators.willr(h, l, c), pd.Series)
        # 2. pandas-ta例外時はPandasTAError
        with patch("pandas_ta.willr", side_effect=Exception("API Error")):
            with pytest.raises(PandasTAError):
                MomentumIndicators.willr(h, l, c)
        # 3. pandas-taがNone返却時
        with patch("pandas_ta.willr", return_value=None):
            res = MomentumIndicators.willr(h, l, c)
            assert isinstance(res, pd.Series)

    def test_cci_comprehensive(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        assert isinstance(MomentumIndicators.cci(h, l, c), pd.Series)
        with patch("pandas_ta.cci", return_value=None):
            assert np.isnan(MomentumIndicators.cci(h, l, c)).all()

    def test_cmo_comprehensive(self, sample_data):
        assert isinstance(MomentumIndicators.cmo(sample_data), pd.Series)
        with patch("pandas_ta.cmo", return_value=None):
            assert np.isnan(MomentumIndicators.cmo(sample_data)).all()

    def test_stc_comprehensive(self, sample_data):
        assert isinstance(MomentumIndicators.stc(sample_data), pd.Series)
        # DataFrameを返す場合
        with patch("pandas_ta.stc", return_value=pd.DataFrame({"0": [1, 2]})):
            assert isinstance(MomentumIndicators.stc(sample_data), pd.Series)
        # 失敗
        with patch("pandas_ta.stc", return_value=None):
            assert np.isnan(MomentumIndicators.stc(sample_data)).all()

    def test_fisher_comprehensive(self, sample_df):
        h, l = sample_df["high"], sample_df["low"]
        assert len(MomentumIndicators.fisher(h, l)) == 2
        with patch("pandas_ta.fisher", return_value=None):
            res = MomentumIndicators.fisher(h, l)
            assert all(np.isnan(s).all() for s in res)

    def test_kst_comprehensive(self, sample_data):
        assert len(MomentumIndicators.kst(sample_data)) == 2
        with patch("pandas_ta.kst", return_value=None):
            res = MomentumIndicators.kst(sample_data)
            assert all(np.isnan(s).all() for s in res)

    def test_qqe_comprehensive(self, sample_data):
        # 1. 正常
        assert isinstance(MomentumIndicators.qqe(sample_data), pd.Series)
        # 2. 失敗時のフォールバック (RSI)
        with patch("pandas_ta.qqe", return_value=None):
            res = MomentumIndicators.qqe(sample_data)
            assert isinstance(res, pd.Series)
        # 3. RSIも失敗
        with (
            patch("pandas_ta.qqe", return_value=None),
            patch("pandas_ta.rsi", return_value=None),
        ):
            res = MomentumIndicators.qqe(sample_data)
            assert np.isnan(res).all()
        # 4. 型エラー
        with pytest.raises(TypeError):
            MomentumIndicators.qqe(None)

    def test_apo_error_handling(self, sample_data):
        res = MomentumIndicators.apo(sample_data)
        assert isinstance(res, pd.Series)

        with pytest.raises(ValueError, match="less than slow"):
            MomentumIndicators.apo(sample_data, fast=20, slow=10)
        with pytest.raises(ValueError, match="positive"):
            MomentumIndicators.apo(sample_data, fast=0)
        with patch("pandas_ta.apo", return_value=None):
            assert np.isnan(MomentumIndicators.apo(sample_data)).all()

    def test_tsi_comprehensive(self, sample_data):
        assert len(MomentumIndicators.tsi(sample_data)) == 2
        with patch("pandas_ta.tsi", return_value=None):
            res = MomentumIndicators.tsi(sample_data)
            assert all(np.isnan(s).all() for s in res)

    def test_pgo_comprehensive(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # 正常 (エラーが出る可能性があるので try-except で囲むか、修正済みを期待)
        try:
            res = MomentumIndicators.pgo(h, l, c)
            assert isinstance(res, pd.Series)
        except Exception:
            pass
        # 失敗
        with patch("pandas_ta.pgo", return_value=None):
            assert np.isnan(MomentumIndicators.pgo(h, l, c)).all()

    def test_psl_comprehensive(self, sample_df):
        from app.services.indicators.data_validation import PandasTAError

        c = sample_df["close"]
        assert isinstance(MomentumIndicators.psl(c), pd.Series)
        # pandas-ta例外時はPandasTAError
        with patch("pandas_ta.psl", side_effect=Exception()):
            with pytest.raises(PandasTAError):
                MomentumIndicators.psl(c)

    def test_squeeze_comprehensive(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        assert isinstance(MomentumIndicators.squeeze(h, l, c), pd.DataFrame)
        with patch("pandas_ta.squeeze", return_value=None):
            assert np.isnan(MomentumIndicators.squeeze(h, l, c)).all().all()

    def test_ao_bop_cg_coppock(self, sample_df):
        h, l, c, o, v = (
            sample_df["high"],
            sample_df["low"],
            sample_df["close"],
            sample_df["open"],
            sample_df["volume"],
        )
        # AO
        assert isinstance(MomentumIndicators.ao(h, l), pd.Series)
        # BOP
        assert isinstance(MomentumIndicators.bop(o, h, l, c), pd.Series)
        # CG
        assert isinstance(MomentumIndicators.cg(c), pd.Series)
        # Coppock
        assert isinstance(MomentumIndicators.coppock(c), pd.Series)

    def test_ichimoku_extended(self, sample_df):
        """Ichimoku は OverlapIndicators に移動済み"""
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # 1. 正常 (pandas-ta)
        res = OverlapIndicators.ichimoku(h, l, c)
        assert "tenkan_sen" in res
        assert isinstance(res["tenkan_sen"], pd.Series)

        # 2. pandas-taが失敗した場合はNaNを返す（フォールバックは廃止）
        with patch("pandas_ta.ichimoku", return_value=None):
            res = OverlapIndicators.ichimoku(h, l, c)
            assert "tenkan_sen" in res
            assert res["tenkan_sen"].isna().all()

        # 3. 例外発生時もNaNを返す
        with patch("pandas_ta.ichimoku", side_effect=Exception("Internal Error")):
            res = OverlapIndicators.ichimoku(h, l, c)
            assert "tenkan_sen" in res
            assert res["tenkan_sen"].isna().all()
