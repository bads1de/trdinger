"""
VWAPパラメータ修正のTDDテスト
"""
import pandas as pd
import numpy as np
import pytest
from app.services.indicators.technical_indicators.volume import VolumeIndicators
from app.services.indicators.utils import PandasTAError


class TestVWAPParameterFix:
    """VWAPメソッドのパラメータ修正テスト"""

    def setup_method(self):
        """テストデータのセットアップ"""
        # サンプルデータ作成
        np.random.seed(42)
        n = 50
        self.high = pd.Series(np.random.uniform(200, 250, n), name='high')
        self.low = pd.Series(np.random.uniform(150, 200, n), name='low')
        self.close = pd.Series(np.random.uniform(175, 225, n), name='close')
        self.volume = pd.Series(np.random.uniform(1000, 10000, n).astype(int), name='volume')

    def test_vwap_with_period_parameter(self):
        """VWAPメソッドがperiodパラメータを受け付けて正しい結果を返すことを確認"""
        # periodパラメータで呼び出し
        result = VolumeIndicators.vwap(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            period=10
        )

        # 結果の検証
        assert isinstance(result, pd.Series), "結果はpd.Seriesであるべき"
        assert len(result) > 0, "結果は空でないべき"
        assert not result.isna().all(), "結果が全てNaNであってはならない"

    def test_vwap_without_period_parameter(self):
        """VWAPメソッドがperiodパラメータなしで動作することを確認"""
        # periodパラメータなしで呼び出し（デフォルト値を使用）
        result = VolumeIndicators.vwap(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume
        )

        # 結果の検証
        assert isinstance(result, pd.Series), "結果はpd.Seriesであるべき"
        assert len(result) > 0, "結果は空でないべき"
        assert not result.isna().all(), "結果が全てNaNであってはならない"

    def test_vwap_with_anchor_parameter(self):
        """VWAPメソッドがanchorパラメータを受け付けて正しい結果を返すことを確認"""
        # anchorパラメータで呼び出し
        result = VolumeIndicators.vwap(
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            anchor="M"
        )

        # 結果の検証
        assert isinstance(result, pd.Series), "結果はpd.Seriesであるべき"
        assert len(result) > 0, "結果は空でないべき"
        assert not result.isna().all(), "結果が全てNaNであってはならない"

    def test_vwap_invalid_parameters(self):
        """無効なパラメータで適切にエラーが発生することを確認"""
        with pytest.raises(PandasTAError):
            # 無効なタイプのパラメータ
            VolumeIndicators.vwap(
                high="invalid",
                low=self.low,
                close=self.close,
                volume=self.volume
            )

        with pytest.raises(PandasTAError):
            # 無効なタイプのパラメータ
            VolumeIndicators.vwap(
                high=self.high,
                low="invalid",
                close=self.close,
                volume=self.volume
            )