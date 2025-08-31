"""
TrendインジケータのPandasオンリー移行テスト
"""
import pytest
import pandas as pd
import numpy as np
from backend.app.services.indicators.technical_indicators.trend import TrendIndicators as TI
from backend.app.services.indicators.utils import PandasTAError
class TestTrendMigration:
    """TrendインジケータのPandasオンリー移行テスト"""

    def test_sma_pandas_series(self):
        """SMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='close')
        result = TI.sma(data, length=3)

        assert isinstance(result, pd.Series)
        # SMA(3) の計算結果を確認 - 最初の2つはNaN、残りは正しい値
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    # def test_sma_type_error deleted - now only accepts pd.Series

    def test_ema_pandas_series(self):
        """EMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.ema(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本的なEMA計算の確認（最初の値はNaN）
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # length-1 以降は値が存在
        assert not np.isnan(result.iloc[2])

    # def test_ema_type_error deleted - now only accepts pd.Series

    def test_ppo_pandas_series(self):
        """PPO: データとしてpd.Seriesを受け取りTuple[pd.Series, pd.Series, pd.Series]を返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.ppo(data, fast=3, slow=5, signal=2)

        assert isinstance(result, tuple)
        assert len(result) == 3
        for series in result:
            assert isinstance(series, pd.Series)

    # def test_ppo_type_error deleted - now only accepts pd.Series

    def test_dema_pandas_series(self):
        """DEMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.dema(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本的な検証 (最初の値はNaN)
        assert np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[5])  # DEFAはEMAを多段なので

    # def test_dema_type_error deleted - now only accepts pd.Series

    def test_tema_pandas_series(self):
        """TEMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.tema(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本的な検証 (最初の値はNaN)
        assert np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[8])  # TEMAはEMAを多段なので後ろの方は値がある

    # def test_tema_type_error deleted - now only accepts pd.Series

    def test_stc_pandas_series(self):
        """STC: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 6, dtype=float)  # 大きいデータでテスト
        result = TI.stc(data, length=10, fast_length=23, slow_length=50)

        assert isinstance(result, pd.Series)
        # STCは複雑なので、NaNが含まれていることを確認する程度
        assert len(result) == len(data)

    # def test_stc_type_error deleted - now only accepts pd.Series

    def test_wma_pandas_series(self):
        """WMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.wma(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本的な検証 (最初の値はNaN)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])

    # def test_wma_type_error deleted - now only accepts pd.Series

    def test_trima_pandas_series(self):
        """TRIMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.trima(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本的な検証 (最初の値はNaN)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])

    # def test_trima_type_error deleted - now only accepts pd.Series

    def test_kama_pandas_series(self):
        """KAMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.kama(data, length=5)

        assert isinstance(result, pd.Series)
        # 基本的な検証 (最初の値はNaN)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[5])

    # def test_kama_type_error deleted - now only accepts pd.Series

    def test_t3_pandas_series(self):
        """T3: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.t3(data, length=5, a=0.7)

        assert isinstance(result, pd.Series)
        # T3は複雑なので、NaNじゃない値があることを確認
        assert not np.isnan(result.iloc[-1])

    # def test_t3_type_error deleted - now only accepts pd.Series

    def test_sar_pandas_series(self):
        """SAR: high, low, closeを受け取りpd.Seriesを返すことを確認"""
        high = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
        low = pd.Series([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        close = pd.Series([10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5], dtype=float)
        result = TI.sar(high, low, af=0.02, max_af=0.2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)

    # def test_sar_type_error deleted - now only accepts pd.Series

    def test_midprice_pandas_series(self):
        """MIDPRICE: high, lowを受け取りpd.Seriesを返すことを確認"""
        high = pd.Series([10, 11, 12, 13, 14, 15], dtype=float)
        low = pd.Series([9, 8, 7, 6, 5, 4], dtype=float)
        result = TI.midprice(high, low, length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    # def test_midprice_type_error deleted - now only accepts pd.Series

    def test_hma_pandas_series(self):
        """HMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3, dtype=float)  # 大きいデータでテスト
        result = TI.hma(data, length=5)

        assert isinstance(result, pd.Series)
        # HMAは複雑なので、NaNじゃない値があることを確認
        assert not np.isnan(result.iloc[-1])

    # def test_hma_type_error deleted - now only accepts pd.Series

    def test_midpoint_pandas_series(self):
        """MIDPOINT: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.midpoint(data, length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    # def test_midpoint_type_error deleted - now only accepts pd.Series

    def test_zlma_pandas_series(self):
        """ZLMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.zlma(data, length=5)

        assert isinstance(result, pd.Series)
        # ZLMAはEM Aベースなので、適当な検証
        assert len(result) == len(data)

    # def test_zlma_type_error deleted - now only accepts pd.Series

    def test_vwma_pandas_series(self):
        """VWMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        volume = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        result = TI.vwma(data, volume=volume, length=3)

        assert isinstance(result, pd.Series)
        # VWMAの基本検証
        assert len(result) == len(data)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    # def test_vwma_type_error deleted - now only accepts pd.Series

    def test_swma_pandas_series(self):
        """SWMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.swma(data, length=5)

        assert isinstance(result, pd.Series)
        # SWMAの基本検証
        assert len(result) == len(data)

    # def test_swma_type_error deleted - now only accepts pd.Series

    def test_alma_pandas_series(self):
        """ALMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3, dtype=float)
        result = TI.alma(data, length=9, sigma=6.0, offset=0.85)

        assert isinstance(result, pd.Series)
        # ALMAの基本検証
        assert len(result) == len(data)

    # def test_alma_type_error deleted - now only accepts pd.Series

    def test_rma_pandas_series(self):
        """RMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.rma(data, length=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # RMA(alpha=1/length)でEMAと同様の検証
        assert not np.isnan(result.iloc[2])

    # def test_rma_type_error deleted - now only accepts pd.Series

    def test_ichimoku_cloud_pandas_series(self):
        """Ichimoku Cloud: high, low, closeを受け取り五つのSeriesを返すことを確認"""
        high = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
        low = pd.Series([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        close = pd.Series([10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5], dtype=float)

        result = TI.ichimoku(high, low, close, tenkan=2, kijun=3, senkou=5)

        assert isinstance(result, tuple)
        assert len(result) == 5
        for series in result:
            assert isinstance(series, pd.Series)
            assert len(series) == len(high)

    # def test_ichimoku_cloud_type_error deleted - now only accepts pd.Series

    def test_stochf_pandas_series(self):
        """STOCF: high, low, closeを受け取りTuple[pd.Series, pd.Series]を返すことを確認"""
        high = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
        low = pd.Series([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        close = pd.Series([10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5], dtype=float)

        result = TI.stochf(high, low, close, length=3, fast_length=2)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for series in result:
            assert isinstance(series, pd.Series)
            assert len(series) == len(high)

    # def test_stochf_type_error deleted - now only accepts pd.Series

    def test_ma_pandas_series(self):
        """MA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.ma(data, period=3)

        assert isinstance(result, pd.Series)
        # MA はSMAなので、SMAと同じ結果
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    # def test_ma_type_error deleted - now only accepts pd.Series

    def test_ma_with_close(self):
        """MA: closeパラメータが指定された場合の処理を確認"""
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = TI.ma(data=None, close=data, period=3)
        assert isinstance(result, pd.Series)

    def test_mavp_pandas_series(self):
        """MAVP: データとperiodsとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        periods = pd.Series([2, 2, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
        result = TI.mavp(data, periods, minperiod=2, maxperiod=5, matype=0)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)
        assert not np.isnan(result.iloc[1])  # minperiod=2なので

    # def test_mavp_type_error deleted - now only accepts pd.Series

    def test_mavp_length_error(self):
        """MAVP: dataとperiodsの長さが一致しない場合PandasTAErrorが発生することを確認"""
        data = pd.Series([1, 2, 3], dtype=float)
        periods = pd.Series([2, 2], dtype=float)
        with pytest.raises(PandasTAError):
            TI.mavp(data, periods, minperiod=2, maxperiod=5, matype=0)

    def test_price_ema_ratio_pandas_series(self):
        """PRICE_EMA_RATIO: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.price_ema_ratio(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)
        assert not np.isnan(result.iloc[-1])

    # test_mama_pandas_series removed - mama function removed due to pandas-ta compatibility issues

    def test_maxindex_pandas_series(self):
        """MAXINDEX: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 10, 3, 4, 5, 6, 7, 8, 9, 2], dtype=float)
        result = TI.maxindex(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_minindex_pandas_series(self):
        """MININDEX: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 10, 3, 4, 5, 6, 7, 8, 9, 2], dtype=float)
        result = TI.minindex(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_minmax_pandas_series(self):
        """MINMAX: データとしてpd.Seriesを受け取りTuple[pd.Series, pd.Series]を返すことを確認"""
        data = pd.Series([1, 10, 3, 4, 5], dtype=float)
        result = TI.minmax(data, length=3)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for series in result:
            assert isinstance(series, pd.Series)
            assert len(series) == len(data)

    def test_minmaxindex_pandas_series(self):
        """MINMAXINDEX: データとしてpd.Seriesを受け取りTuple[pd.Series, pd.Series]を返すことを確認"""
        data = pd.Series([1, 10, 3, 4, 5], dtype=float)
        result = TI.minmaxindex(data, length=3)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for series in result:
            assert isinstance(series, pd.Series)
            assert len(series) == len(data)

    def test_hwma_pandas_series(self):
        """HWMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.hwma(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_jma_pandas_series(self):
        """JMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.jma(data, length=7, phase=0.0, power=2.0)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_mcgd_pandas_series(self):
        """MCGD: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.mcgd(data, length=10)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_ohlc4_pandas_series(self):
        """OHLC4: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        open_ = pd.Series([10, 11, 12, 13], dtype=float)
        high = pd.Series([12, 13, 14, 15], dtype=float)
        low = pd.Series([9, 10, 11, 12], dtype=float)
        close = pd.Series([11, 12, 13, 14], dtype=float)
        result = TI.ohlc4(open_, high, low, close)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(open_)

    def test_pwma_pandas_series(self):
        """PWMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.pwma(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_sinwma_pandas_series(self):
        """SINWMA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.sinwma(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_ssf_pandas_series(self):
        """SSF: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.ssf(data, length=3)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_vidya_pandas_series(self):
        """VIDYA: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float) * 3  # 大きいデータ
        result = TI.vidya(data, length=10, adjust=True)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_wcp_pandas_series(self):
        """WCP: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = TI.wcp(data)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)
        pd.testing.assert_series_equal(result, data)

    def test_linreg_pandas_series(self):
        """LINREG: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.linreg(data, length=5)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_linreg_slope_pandas_series(self):
        """LINREG_SLOPE: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.linreg_slope(data, length=5)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_linreg_intercept_pandas_series(self):
        """LINREG_INTERCEPT: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.linreg_intercept(data, length=5)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_linreg_angle_pandas_series(self):
        """LINREG_ANGLE: データとしてpd.Seriesを受け取りpd.Seriesを返すことを確認"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = TI.linreg_angle(data, length=5, degrees=False)

        assert isinstance(result, pd.Series)
        # 基本検証
        assert len(result) == len(data)

    def test_hl2_comprehensive(self):
        """HL2: comprehensive test"""
        high = pd.Series([10, 11, 12], dtype=float)
        low = pd.Series([9, 8, 7], dtype=float)
        result = TI.hl2(high, low)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)
        expected = (high + low) / 2
        pd.testing.assert_series_equal(result, expected)

    def test_hlc3_comprehensive(self):
        """HLC3: comprehensive test"""
        high = pd.Series([10, 11, 12], dtype=float)
        low = pd.Series([9, 8, 7], dtype=float)
        close = pd.Series([9.5, 10.5, 11.5], dtype=float)
        result = TI.hlc3(high, low, close)

        assert isinstance(result, pd.Series)
        assert len(result) == len(high)
        expected = (high + low + close) / 3
        pd.testing.assert_series_equal(result, expected)