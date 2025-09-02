"""
PVRパラメータ修正のTDDテスト
"""
import pandas as pd
import numpy as np
import pytest
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.utils import PandasTAError


class TestPVRParameterFix:
    """PVRメソッドのパラメータ修正テスト"""

    def setup_method(self):
        """テストデータのセットアップ"""
        # サンプルデータ作成
        np.random.seed(42)
        n = 50
        self.close = pd.Series(np.random.uniform(100, 200, n), name='close')
        self.volume = pd.Series(np.random.uniform(1000, 10000, n).astype(int), name='volume')

    def test_pvr_with_length_parameter(self):
        """PVRメソッドがlengthパラメータを受け付けて正しい結果を返すことを確認"""
        # lengthパラメータで呼び出し
        result = VolumeIndicators.pvr(close=self.close, volume=self.volume, length=10)

        # 結果の検証
        assert isinstance(result, pd.Series), "結果はpd.Seriesであるべき"
        assert len(result) > 0, "結果は空でないべき"
        assert not result.isna().all(), "結果が全てNaNであってはならない"

    def test_pvr_with_period_parameter(self):
        """PVRメソッドがperiodパラメータを受け付けて正しい結果を返すことを確認"""
        # periodパラメータで呼び出し
        result = VolumeIndicators.pvr(close=self.close, volume=self.volume, period=10)

        # 結果の検証
        assert isinstance(result, pd.Series), "結果はpd.Seriesであるべき"
        assert len(result) > 0, "結果は空でないべき"
        assert not result.isna().all(), "結果が全てNaNであってはならない"

    def test_pvr_length_and_period_same_result(self):
        """lengthとperiodパラメータで同じ結果を返すことを確認"""
        result_length = VolumeIndicators.pvr(close=self.close, volume=self.volume, length=10)
        result_period = VolumeIndicators.pvr(close=self.close, volume=self.volume, period=10)

        # 結果が同じであることを確認
        pd.testing.assert_series_equal(result_length, result_period, check_names=False)

    def test_pvr_invalid_parameters(self):
        """無効なパラメータで適切にエラーが発生することを確認"""
        with pytest.raises(PandasTAError):
            # 無効なタイプのパラメータ
            VolumeIndicators.pvr(close="invalid", volume=self.volume)

        with pytest.raises(PandasTAError):
            # 無効なタイプのパラメータ
            VolumeIndicators.pvr(close=self.close, volume="invalid")