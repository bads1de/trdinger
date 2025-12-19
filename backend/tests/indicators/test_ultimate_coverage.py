import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators

class TestUltimateCoverage:
    @pytest.fixture
    def sample_df(self):
        rows = 100
        return pd.DataFrame({
            "open": np.random.normal(100, 1, rows),
            "high": np.random.normal(101, 1, rows),
            "low": np.random.normal(99, 1, rows),
            "close": np.random.normal(100, 1, rows),
            "volume": np.random.normal(1000, 10, rows)
        }, index=pd.date_range("2023-01-01", periods=rows))

    def test_volume_all_fallbacks(self, sample_df):
        h, l, c, v = sample_df["high"], sample_df["low"], sample_df["close"], sample_df["volume"]
        
        # すべての計算がNoneを返す状況をMockで作り、フォールバック（NaN返却）を網羅
        with patch("pandas_ta.ad", return_value=None), \
             patch("pandas_ta.adosc", return_value=None), \
             patch("pandas_ta.obv", return_value=None), \
             patch("pandas_ta.eom", return_value=None), \
             patch("pandas_ta.cmf", return_value=None), \
             patch("pandas_ta.pvo", return_value=None):
            
            assert np.isnan(VolumeIndicators.ad(h, l, c, v)).all()
            assert np.isnan(VolumeIndicators.adosc(h, l, c, v)).all()
            assert np.isnan(VolumeIndicators.obv(c, v)).all()
            assert np.isnan(VolumeIndicators.eom(h, l, c, v)).all()
            assert np.isnan(VolumeIndicators.cmf(h, l, c, v)).all()
            res_pvo = VolumeIndicators.pvo(v)
            assert all(np.isnan(s).all() for s in res_pvo)

    def test_momentum_complex_params(self, sample_df):
        c, h, l = sample_df["close"], sample_df["high"], sample_df["low"]
        
        # 1. TRIX の全分岐 (scalar, drift, offset)
        res = MomentumIndicators.trix(c, length=10, signal=5, scalar=1.0, drift=2, offset=1)
        assert len(res) == 3
        
        # 2. STC の極端な平滑化パラメータ
        assert isinstance(MomentumIndicators.stc(c, d1=1, d2=1), pd.Series)
        
        # 3. FISHER / KST / TSI の全パラメータ
        assert len(MomentumIndicators.fisher(h, l, length=5, signal=2)) == 2
        assert len(MomentumIndicators.kst(c, signal=5)) == 2
        assert len(MomentumIndicators.tsi(c, drift=2)) == 2
        
        # 4. APO / CTI のパラメータ
        assert isinstance(MomentumIndicators.apo(c, ma_mode="sma"), pd.Series)
        assert isinstance(MomentumIndicators.cti(c, length=5), pd.Series)

    def test_trend_volatility_ma_modes(self, sample_df):
        c, h, l = sample_df["close"], sample_df["high"], sample_df["low"]
        
        # 1. BIAS の全 MA タイプを再確認
        for ma in ["sma", "ema", "wma", "hma", "zlma"]:
            assert isinstance(TrendIndicators.bias(c, ma_type=ma), pd.Series)
            
        # 2. ACCBANDS / KELTNER の計算失敗
        with patch("pandas_ta.accbands", return_value=None):
            res = VolatilityIndicators.accbands(h, l, c)
            assert all(np.isnan(s).all() for s in res)
            
        # 3. RVI の詳細パラメータ
        assert isinstance(VolatilityIndicators.rvi(c, h, l, refined=True, thirds=True), pd.Series)

    def test_handle_pandas_ta_errors_deep(self, sample_df):
        # ValueError はデコレーターでそのまま再発生することを確認
        with patch("pandas_ta.rsi", side_effect=ValueError("Test")):
            with pytest.raises(ValueError):
                MomentumIndicators.rsi(sample_df["close"], 14)

    def test_indicator_validation_paths(self, sample_df):
        # データが短すぎる場合のバリデーション分岐を確認
        short_data = pd.Series([1.0], index=sample_df.index[:1])
        # rsi等のメソッドは内部で validation をチェックし、失敗時は NaN シリーズを返す
        res = MomentumIndicators.rsi(short_data, 14)
        assert np.isnan(res).all()
        
        # atr も同様に NaN を返す
        res_atr = VolatilityIndicators.atr(short_data, short_data, short_data)
        assert np.isnan(res_atr).all()




    def test_stoch_parameter_logic(self, sample_df):
        h, l, c = sample_df["high"], sample_df["low"], sample_df["close"]
        # d_length が d を上書きするロジック
        res = MomentumIndicators.stoch(h, l, c, d=3, d_length=10)
        assert len(res) == 2
        # すべて NaN の場合の戻り値形式
        with patch("pandas_ta.stoch", return_value=pd.DataFrame()):
            res = MomentumIndicators.stoch(h, l, c)
            assert len(res) == 2
