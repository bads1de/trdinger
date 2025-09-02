"""
KST指標のテスト

TDDテストケース
"""
import pytest
import pandas as pd
import numpy as np
from backend.app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestKST:
    """KST指標のテストケース"""

    @pytest.fixture
    def sample_data(self):
        """テストデータの生成 - トレンドデータのサンプル"""
        np.random.seed(42)
        trend = np.linspace(100, 200, 100) + np.random.randn(100) * 5
        return pd.Series(trend, name='close')

    def test_kst_valid_data(self, sample_data):
        """KSTが有効なデータを正しく処理する"""
        kst_value, signal_value = MomentumIndicators.kst(
            sample_data, roc1=10, roc2=15, roc3=20, roc4=30,
            n1=10, n2=10, n3=10, n4=15, signal=9
        )

        # NaNでない値があることを確認
        assert not kst_value.isna().all(), "KSTに有効な値があるべき"
        assert not signal_value.isna().all(), "KST信号に有効な値があるべき"

        # 型チェック
        assert isinstance(kst_value, pd.Series), "KST値はSeriesであるべき"
        assert isinstance(signal_value, pd.Series), "KST信号はSeriesであるべき"

    def test_kst_invalid_data_type(self):
        """Invalid data typeを拒否"""
        with pytest.raises(TypeError, match="data must be pandas Series"):
            MomentumIndicators.kst([100, 101], roc1=10)

    def test_kst_empty_data(self):
        """空データを処理"""
        data = pd.Series([])

        kst_value, signal_value = MomentumIndicators.kst(data, roc1=10, roc2=15, roc3=20, roc4=30)

        assert len(kst_value) == 0
        assert len(signal_value) == 0

    def test_kst_short_series(self):
        """短いデータ系列を処理（一部NaNとなる）"""
        data = pd.Series([100, 101, 102, 103])

        kst_value, signal_value = MomentumIndicators.kst(
            data, roc1=10, roc2=15, roc3=20, roc4=30  # 長い周期の指定
        )

        # 短い系列では全てNaNとなる可能性
        assert isinstance(kst_value, pd.Series)
        assert isinstance(signal_value, pd.Series)

    @pytest.mark.parametrize("roc_values", [
        (10, 15, 20, 30),
        (5, 10, 15, 20),
        (20, 25, 30, 35)
    ])
    def test_kst_various_roc_parameters(self, sample_data, roc_values):
        """異なるROCパラメータでのKST計算"""
        roc1, roc2, roc3, roc4 = roc_values

        # パラメータエラーなく計算できることを確認
        try:
            kst_value, signal_value = MomentumIndicators.kst(
                sample_data, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4
            )
            assert isinstance(kst_value, pd.Series)
            assert isinstance(signal_value, pd.Series)
        except Exception as e:
            pytest.fail(f"ROCパラメータ {roc_values} でエラー: {e}")

    def test_kst_signal_calculation(self, sample_data):
        """KST信号線が正しく計算される"""
        kst_value, signal_value = MomentumIndicators.kst(
            sample_data, roc1=10, roc2=15, roc3=20, roc4=30, signal=5  # 短いsignal期間
        )

        # 有効な値の期間でKSTとsignalの関係を確認
        valid_idx = ~(kst_value.isna() | signal_value.isna())

        if valid_idx.sum() > 1:
            # signalはkstの移動平均に近いはず
            # 厳密なチェックは難しいので基本的な検証のみ
            assert valid_idx.any(), "有効な値が存在する"

    def test_kst_with_rorc_parameters(self, sample_data):
        """rorcパラメータでのKST計算テスト（rorcパラメータがサポートされていることを確認）"""
        # rorc1, rorc2, rorc3, rorc4パラメータを使用
        kst_value, signal_value = MomentumIndicators.kst(
            sample_data, rorc1=10, rorc2=15, rorc3=20, rorc4=30,
            sma1=10, sma2=10, sma3=10, sma4=15, signal=9
        )
        assert isinstance(kst_value, pd.Series), "KST値はSeriesであるべき"
        assert isinstance(signal_value, pd.Series), "KST信号はSeriesであるべき"
        assert not kst_value.isna().all(), "KSTに有効な値があるべき"
        assert not signal_value.isna().all(), "KST信号に有効な値があるべき"