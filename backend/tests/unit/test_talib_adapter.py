"""
TA-Libアダプタークラスの包括的テスト

TDDアプローチでTA-Libアダプターの機能をテストします。
現在の実装に合わせて、機能別アダプタークラスをテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト対象のインポート
from app.core.services.indicators.adapters import (
    BaseAdapter,
    TALibCalculationError,
    TrendAdapter,
    MomentumAdapter,
    VolatilityAdapter,
    VolumeAdapter,
)
from app.core.utils.data_utils import ensure_series, DataConversionError


class TestBaseAdapter:
    """BaseAdapterクラスのテストクラス"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用の価格データ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)  # 再現性のため

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.Series(prices, index=dates, name="close")

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "open": close_prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_ensure_series_with_series(self, sample_price_data):
        """pandas.Seriesの変換テスト"""
        result = ensure_series(sample_price_data, raise_on_error=True)
        assert isinstance(result, pd.Series)
        assert result.equals(sample_price_data)

    def test_ensure_series_with_list(self):
        """リストからpandas.Seriesへの変換テスト"""
        test_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ensure_series(test_list, raise_on_error=True)
        assert isinstance(result, pd.Series)
        assert list(result.values) == test_list

    def test_ensure_series_with_numpy_array(self):
        """numpy配列からpandas.Seriesへの変換テスト"""
        test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ensure_series(test_array, raise_on_error=True)
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, test_array)

    def test_ensure_series_with_invalid_type(self):
        """サポートされていないデータ型のテスト"""
        # 辞書型など、明らかにサポートされていない型を使用
        invalid_data = {"key": "value"}
        with pytest.raises(DataConversionError):
            ensure_series(invalid_data, raise_on_error=True)

    def test_create_series_result(self):
        """計算結果のSeries変換テスト"""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_index = pd.date_range("2024-01-01", periods=5, freq="D")
        test_name = "TEST_INDICATOR"

        result = BaseAdapter._create_series_result(test_data, test_index, test_name)

        assert isinstance(result, pd.Series)
        assert result.name == test_name
        assert result.index.equals(test_index)
        np.testing.assert_array_equal(result.values, test_data)


class TestTrendAdapter:
    """TrendAdapterクラスのテストクラス"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用の価格データ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)  # 再現性のため

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.Series(prices, index=dates, name="close")

    def test_sma_calculation(self, sample_price_data):
        """SMA計算のテスト"""
        result = TrendAdapter.sma(sample_price_data, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert not result.iloc[:19].notna().any()  # 最初の19個はNaN
        assert result.iloc[19:].notna().all()  # 20個目以降は値あり
        assert result.name == "SMA"

    def test_ema_calculation(self, sample_price_data):
        """EMA計算のテスト"""
        result = TrendAdapter.ema(sample_price_data, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "EMA"

    def test_tema_calculation(self, sample_price_data):
        """TEMA計算のテスト"""
        result = TrendAdapter.tema(sample_price_data, period=30)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "TEMA"

    def test_dema_calculation(self, sample_price_data):
        """DEMA計算のテスト"""
        result = TrendAdapter.dema(sample_price_data, period=30)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "DEMA"

    def test_t3_calculation(self, sample_price_data):
        """T3計算のテスト"""
        result = TrendAdapter.t3(sample_price_data, period=5, vfactor=0.7)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "T3"

    def test_wma_calculation(self, sample_price_data):
        """WMA計算のテスト"""
        result = TrendAdapter.wma(sample_price_data, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "WMA"


class TestMomentumAdapter:
    """MomentumAdapterクラスのテストクラス"""

    @pytest.fixture
    def sample_price_data(self):
        """テスト用の価格データ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)  # 再現性のため

        # 現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.Series(prices, index=dates, name="close")

    def test_rsi_calculation(self, sample_price_data):
        """RSI計算のテスト"""
        result = MomentumAdapter.rsi(sample_price_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "RSI"

        # RSIは0-100の範囲
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_momentum_calculation(self, sample_price_data):
        """モメンタム計算のテスト"""
        result = MomentumAdapter.momentum(sample_price_data, period=10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.index.equals(sample_price_data.index)
        assert result.name == "MOM"


class TestVolatilityAdapter:
    """VolatilityAdapterクラスのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "open": close_prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_atr_calculation(self, sample_ohlcv_data):
        """ATR計算のテスト"""
        result = VolatilityAdapter.atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            period=14,
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "ATR"


class TestVolumeAdapter:
    """VolumeAdapterクラスのテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        # OHLCV データを生成
        data = pd.DataFrame(
            {
                "open": close_prices * (1 + np.random.normal(0, 0.001, 100)),
                "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        return data

    def test_ad_calculation(self, sample_ohlcv_data):
        """A/D Line計算のテスト"""
        result = VolumeAdapter.ad(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            sample_ohlcv_data["volume"],
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_invalid_input_handling_trend(self):
        """TrendAdapterの不正な入力のハンドリングテスト"""
        # 空のSeriesでエラーが発生することを確認
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(TALibCalculationError):
            TrendAdapter.sma(empty_series, period=20)

        # 期間が不正な場合
        valid_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(TALibCalculationError):
            TrendAdapter.sma(valid_series, period=0)

        with pytest.raises(TALibCalculationError):
            TrendAdapter.sma(valid_series, period=-1)

        # データ長が期間より短い場合
        with pytest.raises(TALibCalculationError):
            TrendAdapter.sma(valid_series, period=10)

    def test_invalid_input_handling_momentum(self):
        """MomentumAdapterの不正な入力のハンドリングテスト"""
        # 空のSeriesでエラーが発生することを確認
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(TALibCalculationError):
            MomentumAdapter.rsi(empty_series, period=14)

        # 期間が不正な場合
        valid_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(TALibCalculationError):
            MomentumAdapter.rsi(valid_series, period=0)


class TestPerformanceAndAccuracy:
    """パフォーマンスと精度のテスト"""

    @pytest.fixture
    def large_price_data(self):
        """大きなテスト用価格データ"""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 1000)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.Series(prices, index=dates, name="close")

    def test_performance_comparison(self, large_price_data):
        """パフォーマンス比較テスト"""
        import time

        # TA-Libでの計算時間
        start_time = time.time()
        talib_result = TrendAdapter.sma(large_price_data, period=20)
        talib_time = time.time() - start_time

        # pandasでの計算時間
        start_time = time.time()
        pandas_result = large_price_data.rolling(window=20).mean()
        pandas_time = time.time() - start_time

        print(f"TA-Lib時間: {talib_time:.4f}秒")
        print(f"pandas時間: {pandas_time:.4f}秒")

        # 結果の精度比較（小数点以下6桁まで）
        # NaN値を除外して比較
        talib_clean = talib_result.dropna()
        pandas_clean = pandas_result.dropna()

        # インデックスを合わせる
        common_index = talib_clean.index.intersection(pandas_clean.index)

        pd.testing.assert_series_equal(
            talib_clean.loc[common_index].round(6),
            pandas_clean.loc[common_index].round(6),
            check_names=False,
        )

    def test_data_type_conversion(self):
        """データ型変換のテスト"""
        # リストからの変換
        list_data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        result = TrendAdapter.sma(pd.Series(list_data), period=5)
        assert isinstance(result, pd.Series)

        # numpy配列からの変換
        array_data = np.array(list_data)
        result = TrendAdapter.sma(pd.Series(array_data), period=5)
        assert isinstance(result, pd.Series)

    def test_nan_handling(self):
        """NaN値のハンドリングテスト"""
        # データにNaNを含む場合の処理
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.Series(range(50), index=dates, dtype=float)
        data.iloc[20:25] = np.nan

        result = TrendAdapter.sma(data, period=10)

        # 結果がSeriesであることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)


class TestIntegration:
    """統合テスト"""

    def test_adapter_consistency(self):
        """アダプター間の一貫性テスト"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        data = pd.Series(prices, index=dates, name="close")

        # 各アダプターが同じ形式の結果を返すことを確認
        sma_result = TrendAdapter.sma(data, period=20)
        ema_result = TrendAdapter.ema(data, period=20)
        rsi_result = MomentumAdapter.rsi(data, period=14)

        # 全て同じ長さとインデックスを持つことを確認
        assert len(sma_result) == len(data)
        assert len(ema_result) == len(data)
        assert len(rsi_result) == len(data)

        assert sma_result.index.equals(data.index)
        assert ema_result.index.equals(data.index)
        assert rsi_result.index.equals(data.index)

    def test_error_propagation(self):
        """エラー伝播のテスト"""
        # 全てのアダプターが同じエラーハンドリングを行うことを確認
        empty_series = pd.Series([], dtype=float)

        with pytest.raises(TALibCalculationError):
            TrendAdapter.sma(empty_series, period=20)

        with pytest.raises(TALibCalculationError):
            MomentumAdapter.rsi(empty_series, period=14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
