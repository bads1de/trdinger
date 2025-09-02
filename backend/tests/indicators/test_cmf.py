"""
CMF (Chaikin Money Flow) 指標の修正テスト
データ不一致の問題をテストおよび修正
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestCMFIndicator:
    """CMF指標のテストクラス"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='H')

        # 基本的なOHLCVデータを生成
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

    def test_cmf_normal_calculation(self, sample_ohlcv_data):
        """正規のCMF計算テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # サービス経由で計算
        cmf_result = VolumeIndicators.cmf(
            high=pd.Series(sample_ohlcv_data['high']),
            low=pd.Series(sample_ohlcv_data['low']),
            close=pd.Series(sample_ohlcv_data['close']),
            volume=pd.Series(sample_ohlcv_data['volume']),
            length=20
        )

        assert cmf_result is not None
        assert isinstance(cmf_result, pd.Series)
        assert len(cmf_result) == len(sample_ohlcv_data)
        assert not cmf_result.isna().all()

        # CMFは通常-1から1の範囲（可能）
        assert cmf_result.dropna().apply(lambda x: -1 <= x <= 1).all()

    def test_cmf_data_length_mismatch_issue(self, sample_ohlcv_data):
        """データ長不一致のテスト（問題再現）"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # 異なる長さのデータを意図的に作成
        short_high = pd.Series(sample_ohlcv_data['high'][:50])
        short_low = pd.Series(sample_ohlcv_data['low'][:50])
        short_close = pd.Series(sample_ohlcv_data['close'][:50])
        short_volume = pd.Series(sample_ohlcv_data['volume'][:40])  # 故意に短くする

        # データ長不一致でPandasTAErrorが発生することを期待（ワッパー経由）
        with pytest.raises(PandasTAError, match="all input series to have the same length"):
            VolumeIndicators.cmf(
                high=short_high,
                low=short_low,
                close=short_close,
                volume=short_volume,
                length=20
            )

    def test_cmf_nan_handling_issue(self, sample_ohlcv_data):
        """NaN値処理のテスト（問題再現）"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # NaNを含むデータを生成
        nan_data = sample_ohlcv_data.copy()
        nan_data.loc[10:20, 'volume'] = np.nan  # ボリュームにNaNを入れる

        cmf_result = VolumeIndicators.cmf(
            high=pd.Series(nan_data['high']),
            low=pd.Series(nan_data['low']),
            close=pd.Series(nan_data['close']),
            volume=pd.Series(nan_data['volume']),
            length=20
        )

        # pandas-taのデフォルト挙動を確認（NaNがどのように処理されるか）
        assert cmf_result is not None

        # NaNの影響で結果にNaNが出現するか確認
        nan_positions = nan_data['volume'].isna()
        if nan_positions.any():
            # pandas-taがNaNをどのように処理するかテスト
            # 初期実装では NaN が結果に伝搬する可能性
            pass

    def test_cmf_pandas_ta_mock_test(self, sample_ohlcv_data):
        """pandas-taの挙動をモックでテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # pandas-taの帰り値をモック
        with patch('pandas_ta.cmf') as mock_cmf:
            # 正常なDataFrameを返すモック
            mock_result = pd.Series([0.5, 0.3, -0.2, np.nan], index=sample_ohlcv_data.index[:4])
            mock_cmf.return_value = mock_result

            result = VolumeIndicators.cmf(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=20
            )

            assert result is not None

    def test_cmf_empty_data_handling(self, sample_ohlcv_data):
        """空データ時の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # 空のSeriesを渡す - PandasTAErrorが発生することを期待
        with pytest.raises(PandasTAError, match="Insufficient data"):
            VolumeIndicators.cmf(
                high=pd.Series([], dtype=float),
                low=pd.Series([], dtype=float),
                close=pd.Series([], dtype=float),
                volume=pd.Series([], dtype=float),
                length=20
            )

    def test_cmf_invalid_input_types(self, sample_ohlcv_data):
        """不正な入力タイプのテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # テスト1: 文字列のSeriesを渡す - TypeErrorを期待
        with pytest.raises(PandasTAError, match="high must be pandas Series"):
            VolumeIndicators.cmf(
                high="not a series",
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=20
            )

    def test_cmf_invalid_numeric_data(self, sample_ohlcv_data):
        """数値に変換できないデータのテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators
        from app.services.indicators.utils import PandasTAError

        # 数値に変換できないデータを同じ長さでテスト
        # 必要最小長を満たすデータを生成（length=20なので最低21ポイント）
        invalid_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u']
        valid_series = pd.Series(sample_ohlcv_data['low'][:21])  # 同じ長さ

        # PandasTAError（ValueError経由）を期待
        with pytest.raises(PandasTAError):
            VolumeIndicators.cmf(
                high=pd.Series(invalid_data),
                low=valid_series,
                close=valid_series,
                volume=pd.Series(range(21)),  # 有効な数値データ
                length=20
            )

    def test_cmf_integration_with_service(self, sample_ohlcv_data):
        """TechnicalIndicatorServiceとの統合テスト"""
        try:
            from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

            service = TechnicalIndicatorService()
            result = service.calculate_indicator(
                sample_ohlcv_data.copy(),
                "CMF",
                {"length": 20}
            )

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == len(sample_ohlcv_data)

        except ImportError:
            pytest.skip("TechnicalIndicatorService 未実装")

    def test_cmf_pandas_ta_failure_simulation(self, sample_ohlcv_data):
        """pandas-ta失敗時のフォールバックテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # pandas-taがNoneを返すケースをモック
        with patch('pandas_ta.cmf', return_value=None):
            # 現在の実装では空のSeriesを返すはず
            empty_result = VolumeIndicators.cmf(
                high=pd.Series(sample_ohlcv_data['high']),
                low=pd.Series(sample_ohlcv_data['low']),
                close=pd.Series(sample_ohlcv_data['close']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=20
            )

            assert isinstance(empty_result, pd.Series)
            # pandas-taがNoneの場合は空のSeriesを返す実装なので
            assert len(empty_result) == 0 or empty_result.isna().all()


if __name__ == "__main__":
    pytest.main([__file__])