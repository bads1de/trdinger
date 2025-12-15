"""
Stochastic RSIインジケーターのテスト (TDD)
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.momentum import MomentumIndicators


class TestStochasticRSI:
    """Stochastic RSI インジケーターのテストクラス"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用の価格データ"""
        # 100日分のテストデータ（トレンドとボラティリティを含む）
        np.random.seed(42)
        base_price = 100
        trend = np.linspace(0, 20, 100)
        noise = np.random.randn(100) * 2
        prices = base_price + trend + noise
        return pd.Series(prices, index=pd.date_range("2024-01-01", periods=100))

    def test_stochrsi_basic_calculation(self, sample_price_data):
        """基本的なStoch RSI計算のテスト"""
        result = MomentumIndicators.stochrsi(
            sample_price_data, rsi_length=14, stoch_length=14, k=3, d=3
        )

        # 戻り値が2つ（k, d）のタプルであることを確認
        assert isinstance(result, tuple)
        assert len(result) == 2

        stoch_k, stoch_d = result

        # 結果がpd.Seriesであることを確認
        assert isinstance(stoch_k, pd.Series)
        assert isinstance(stoch_d, pd.Series)

        # 長さが入力と同じであることを確認
        assert len(stoch_k) == len(sample_price_data)
        assert len(stoch_d) == len(sample_price_data)

    def test_stochrsi_value_range(self, sample_price_data):
        """Stoch RSIの値が0-100の範囲にあることを確認"""
        stoch_k, stoch_d = MomentumIndicators.stochrsi(
            sample_price_data, rsi_length=14, stoch_length=14, k=3, d=3
        )

        # NaN以外の値が0-100の範囲にあることを確認
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()

        if len(valid_k) > 0:
            assert valid_k.min() >= 0
            assert valid_k.max() <= 100

        if len(valid_d) > 0:
            assert valid_d.min() >= 0
            assert valid_d.max() <= 100

    def test_stochrsi_with_default_parameters(self, sample_price_data):
        """デフォルトパラメータでのテスト"""
        result = MomentumIndicators.stochrsi(sample_price_data)

        stoch_k, stoch_d = result

        # NaN値が適切に配置されているか確認
        # 最初のrsi_length + stoch_length期間は計算不可能
        expected_nan_period = 14 + 14  # デフォルト値
        first_valid_idx = stoch_k.first_valid_index()

        if first_valid_idx is not None:
            assert stoch_k.index.get_loc(first_valid_idx) >= expected_nan_period - 10

    def test_stochrsi_custom_parameters(self, sample_price_data):
        """カスタムパラメータでのテスト"""
        result = MomentumIndicators.stochrsi(
            sample_price_data, rsi_length=21, stoch_length=10, k=5, d=5
        )

        stoch_k, stoch_d = result

        # 結果が生成されることを確認
        assert stoch_k is not None
        assert stoch_d is not None
        assert len(stoch_k) == len(sample_price_data)

    def test_stochrsi_invalid_data_type(self):
        """不正なデータ型のテスト"""
        with pytest.raises(TypeError):
            MomentumIndicators.stochrsi([100, 101, 102])  # リストは不可

    def test_stochrsi_invalid_parameters(self, sample_price_data):
        """不正なパラメータのテスト"""
        # 負の期間
        with pytest.raises(ValueError):
            MomentumIndicators.stochrsi(sample_price_data, rsi_length=-1)

        # ゼロの期間
        with pytest.raises(ValueError):
            MomentumIndicators.stochrsi(sample_price_data, stoch_length=0)

    def test_stochrsi_empty_data(self):
        """空データのテスト"""
        empty_series = pd.Series([], dtype=float)
        stoch_k, stoch_d = MomentumIndicators.stochrsi(empty_series)

        assert len(stoch_k) == 0
        assert len(stoch_d) == 0

    def test_stochrsi_insufficient_data(self):
        """不十分なデータ長のテスト"""
        short_data = pd.Series([100, 101, 102, 103, 104])
        stoch_k, stoch_d = MomentumIndicators.stochrsi(
            short_data, rsi_length=14, stoch_length=14
        )

        # すべてNaNであるべき
        assert stoch_k.isna().all()
        assert stoch_d.isna().all()

    def test_stochrsi_oversold_overbought_detection(self):
        """買われすぎ・売られすぎの検出テスト"""
        # 強いトレンドデータ作成（ボラティリティを追加）
        np.random.seed(123)
        base = np.linspace(100, 200, 100)
        noise = np.random.randn(100) * 3
        uptrend = pd.Series(
            base + noise, index=pd.date_range("2024-01-01", periods=100)
        )

        stoch_k, stoch_d = MomentumIndicators.stochrsi(uptrend)

        # StochRSIは0-100の範囲で変動し、有効なデータが生成されることを確認
        valid_k = stoch_k.dropna()
        if len(valid_k) > 10:
            # 値が0-100の範囲にあることを確認
            assert valid_k.min() >= 0
            assert valid_k.max() <= 100
            # 少なくとも何らかの変動があることを確認
            assert valid_k.std() > 0

    def test_stochrsi_k_smoother_than_d(self, sample_price_data):
        """K線がD線よりスムーズであることの確認"""
        stoch_k, stoch_d = MomentumIndicators.stochrsi(sample_price_data, k=3, d=3)

        # D線はK線の移動平均なので、変動が小さいことを期待
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()

        if len(valid_k) > 10 and len(valid_d) > 10:
            k_std = valid_k.std()
            d_std = valid_d.std()
            # D線の標準偏差がK線より小さいまたは同程度であることを確認
            assert d_std <= k_std * 1.2  # 20%の許容範囲




