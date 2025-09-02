"""
KVO (Klinger Volume Oscillator) 指標の修正テスト
NULL返却の問題をテストおよび修正
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


class TestKVOIndicator:
    """KVO指標のテストクラス"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='h')

        # OHLCVデータを生成
        close = 100 + np.cumsum(np.random.randn(length) * 2)
        high = close * (1 + np.random.rand(length) * 0.05)
        low = close * (1 - np.random.rand(length) * 0.05)
        open_price = close + np.random.randn(length) * 2
        volume = np.random.randint(1000, 10000, length)

        return pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    @pytest.fixture
    def sample_ohlcv_data(self):
        """pytest fixture for sample data"""
        return self.setup_sample_data()

    def test_kvo_normal_calculation(self, sample_ohlcv_data):
        """正規のKVO計算テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        kvo_result = VolumeIndicators.kvo(
            high=pd.Series(sample_ohlcv_data['high']),
            low=pd.Series(sample_ohlcv_data['low']),
            close=pd.Series(sample_ohlcv_data['close']),
            volume=pd.Series(sample_ohlcv_data['volume']),
            fast=34,
            slow=55
        )

        assert kvo_result is not None
        assert isinstance(kvo_result, tuple)
        assert len(kvo_result) == 2

        kvo_line, signal_line = kvo_result
        assert isinstance(kvo_line, pd.Series)
        assert isinstance(signal_line, pd.Series)

    def test_kvo_insufficient_data(self, sample_ohlcv_data):
        """データ不足時の処理テスト（NULL返却問題再現）"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # 最小必要データ（55 + 1）より少ないデータを準備
        short_df = sample_ohlcv_data.head(50)

        # データ不足で適切なエラーが発生するかテスト
        with pytest.raises((PandasTAError, ValueError)):
            VolumeIndicators.kvo(
                high=pd.Series(short_df['high']),
                low=pd.Series(short_df['low']),
                close=pd.Series(short_df['close']),
                volume=pd.Series(short_df['volume']),
                fast=34,
                slow=55
            )

    def test_kvo_data_length_mismatch(self, sample_ohlcv_data):
        """データ長不一致時の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # 異なる長さのデータを用意
        short_volume = pd.Series(sample_ohlcv_data['volume'][:80])

        # データ長不一致でPandasTAErrorが発生（デコレータ経由）
        with pytest.raises(PandasTAError, match="same length"):
            VolumeIndicators.kvo(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                volume=short_volume,  # 異なる長さ
                fast=34,
                slow=55
            )

    def test_kvo_nan_values_issue(self, sample_ohlcv_data):
        """NaN値処理テスト（NULL返却問題）"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # NaNを含むデータを生成
        nan_data = sample_ohlcv_data.copy()
        nan_data.loc[10:20, 'volume'] = np.nan

        # NaNが原因でNULLが返されないことを確認
        result = VolumeIndicators.kvo(
            high=pd.Series(nan_data['high']),
            low=pd.Series(nan_data['low']),
            close=pd.Series(nan_data['close']),
            volume=pd.Series(nan_data['volume']),
            fast=34,
            slow=55
        )

        assert result is not None
        assert isinstance(result, tuple)
        kvo_line, signal_line = result

        # NULL返却ではなく、空のSeriesが返されるかも
        if len(kvo_line) > 0:
            assert not kvo_line.isna().all()

    def test_kvo_negative_values(self, sample_ohlcv_data):
        """負の値を含むデータの処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 負の価格を含むデータを生成
        negative_data = sample_ohlcv_data.copy()
        negative_data.loc[10:15, 'high'] = -abs(negative_data.loc[10:15, 'high'])

        result = VolumeIndicators.kvo(
            high=pd.Series(negative_data['high']),
            low=pd.Series(negative_data['low']),
            close=pd.Series(negative_data['close']),
            volume=pd.Series(negative_data['volume']),
            fast=34,
            slow=55
        )

        # 結果が返され、適切に処理されていることを確認
        assert result is not None
        assert isinstance(result, tuple)

    def test_kvo_extreme_volume_values(self, sample_ohlcv_data):
        """極端な出来高値の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 極端に大きな出来高を含むデータを生成
        extreme_data = sample_ohlcv_data.copy()
        extreme_data.loc[10:15, 'volume'] = extreme_data.loc[10:15, 'volume'] * 1e10

        result = VolumeIndicators.kvo(
            high=pd.Series(extreme_data['high']),
            low=pd.Series(extreme_data['low']),
            close=pd.Series(extreme_data['close']),
            volume=pd.Series(extreme_data['volume']),
            fast=34,
            slow=55
        )

        assert result is not None
        assert isinstance(result, tuple)

    def test_kvo_zero_division_prevention(self, sample_ohlcv_data):
        """ゼロ除算防止テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # ゼロ値を含むデータを生成（リスクヘッジ）
        zero_volume_data = sample_ohlcv_data.copy()
        zero_volume_data.loc[10:15, 'volume'] = 0

        result = VolumeIndicators.kvo(
            high=pd.Series(zero_volume_data['high']),
            low=pd.Series(zero_volume_data['low']),
            close=pd.Series(zero_volume_data['close']),
            volume=pd.Series(zero_volume_data['volume']),
            fast=34,
            slow=55
        )

        # ゼロ除算が発生せず結果が返されることを確認
        assert result is not None
        assert isinstance(result, tuple)

    def test_kvo_pandas_ta_failure_simulation(self, sample_ohlcv_data):
        """pandas-taがNoneを返すケースのテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # pandas-ta.kvoがNoneを返す場合のシミュレーション
        with patch('pandas_ta.kvo', return_value=None):
            # NULL防止の実装によりPandasTAErrorが発生（デコレータの挙動）
            with pytest.raises(PandasTAError):
                VolumeIndicators.kvo(
                    high=pd.Series(sample_ohlcv_data['high']),
                    low=pd.Series(sample_ohlcv_data['low']),
                    close=pd.Series(sample_ohlcv_data['close']),
                    volume=pd.Series(sample_ohlcv_data['volume']),
                    fast=34,
                    slow=55
                )

    def test_kvo_pandas_ta_empty_dataframe_simulation(self, sample_ohlcv_data):
        """pandas-taが空のDataFrameを返すケースのテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # pandas-ta.kvoが空のDataFrameを返す場合のシミュレーション
        empty_df = pd.DataFrame()
        with patch('pandas_ta.kvo', return_value=empty_df):
            # NULL防止の実装によりPandasTAErrorが発生（デコレータの挙動）
            with pytest.raises(PandasTAError):
                VolumeIndicators.kvo(
                    high=pd.Series(sample_ohlcv_data['high']),
                    low=pd.Series(sample_ohlcv_data['low']),
                    close=pd.Series(sample_ohlcv_data['close']),
                    volume=pd.Series(sample_ohlcv_data['volume']),
                    fast=34,
                    slow=55
                )

    def test_kvo_integration_with_service(self, sample_ohlcv_data):
        """TechnicalIndicatorServiceとの統合テスト"""
        try:
            from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

            service = TechnicalIndicatorService()
            result = service.calculate_indicator_kvo(
                sample_ohlcv_data.copy()
            )

            if result is not None:
                assert isinstance(result, tuple)
                assert len(result) == 2

        except ImportError:
            pytest.skip("TechnicalIndicatorService 未実装")
        except AttributeError:
            # service.calculate_indicator_kvoが実装されていない場合
            pytest.skip("KVO service method not implemented")

    def test_kvo_calculates_with_valid_values(self, sample_ohlcv_data):
        """KVOが正常に計算され、一部に有効な値を含むことのテスト"""
        from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
        import numpy as np

        service = TechnicalIndicatorService()

        # KVOの計算結果を取得
        result = service.calculate_indicator(sample_ohlcv_data, "KVO", {})

        # 結果はタプルで、2つのSeriesであるはず
        assert isinstance(result, tuple), "KVO should return tuple"
        assert len(result) == 2, "KVO should return 2 elements"

        kvo_line, signal_line = result
        assert isinstance(kvo_line, (pd.Series, np.ndarray)), "KVO line should be Series or ndarray"
        assert isinstance(signal_line, (pd.Series, np.ndarray)), "KVO signal should be Series or ndarray"

        # pandas Seriesに変換して検証
        if isinstance(kvo_line, np.ndarray):
            kvo_line = pd.Series(kvo_line)
        if isinstance(signal_line, np.ndarray):
            signal_line = pd.Series(signal_line)

        # 初期部分はNaN、終端部分に有効な値があるはず（pandas-taの特性）
        assert kvo_line.isna().sum() > 0, "KVO line should have some NaN values (calculation lag)"
        assert not kvo_line.isna().all(), "KVO line should not be all NaN"
        assert signal_line.isna().sum() > 0, "KVO signal should have some NaN values (calculation lag)"
        assert not signal_line.isna().all(), "KVO signal should not be all NaN"

        # 有効な値が存在することを確認
        valid_kvo = kvo_line.dropna()
        valid_signal = signal_line.dropna()
        assert len(valid_kvo) > 0, "KVO line should have valid values"
        assert len(valid_signal) > 0, "KVO signal should have valid values"

        # 同じ長さであることを確認
        assert len(kvo_line) == len(signal_line) == len(sample_ohlcv_data), "All series should have same length"


if __name__ == "__main__":
    pytest.main([__file__])