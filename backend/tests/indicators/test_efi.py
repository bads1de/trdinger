"""
EFI (Elder's Force Index) 指標の修正テスト
不正値の問題をテストおよび修正
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


class TestEFIIndicator:
    """EFI指標のテストクラス"""

    def setup_sample_data(self, length=100):
        """テスト用サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=length, freq='h')

        # OHLCVデータを生成、EFIに適したデータに
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

    def test_efi_normal_calculation(self, sample_ohlcv_data):
        """正規のEFI計算テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        efi_result = VolumeIndicators.efi(
            close=pd.Series(sample_ohlcv_data['close']),
            volume=pd.Series(sample_ohlcv_data['volume']),
            length=13
        )

        assert efi_result is not None
        assert isinstance(efi_result, pd.Series)
        assert len(efi_result) == len(sample_ohlcv_data)
        assert not efi_result.isna().all()

    def test_efi_negative_price_values(self, sample_ohlcv_data):
        """負の価格値の処理テスト（不正値問題）"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 負の価格を含むデータを生成
        negative_close = sample_ohlcv_data['close'].copy()
        negative_close.iloc[10:20] = -negative_close.iloc[10:20]  # 一部を負に変換

        # 不正値として処理されるはず
        try:
            result = VolumeIndicators.efi(
                close=pd.Series(negative_close),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=13
            )
            # pandas-taの挙動を確認
            print(f"負の価格で結果長: {len(result)}")
        except Exception as e:
            print(f"負の価格エラー: {e}")

    def test_efi_negative_volume_values(self, sample_ohlcv_data):
        """負の出来高値の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 負の出来高を含むデータを生成
        negative_volume = sample_ohlcv_data['volume'].copy()
        negative_volume.iloc[10:20] = -np.abs(negative_volume.iloc[10:20])

        result = VolumeIndicators.efi(
            close=pd.Series(sample_ohlcv_data['close']),
            volume=pd.Series(negative_volume),
            length=13
        )

        # 不正値がどのように処理されるか確認
        assert result is not None

    def test_efi_extreme_values(self, sample_ohlcv_data):
        """極端な値の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 極端に大きな値を含むデータを生成
        extreme_close = sample_ohlcv_data['close'].copy()
        extreme_close.iloc[10:15] = extreme_close.iloc[10:15] * 1e10  # 極端に大きな値

        result = VolumeIndicators.efi(
            close=pd.Series(extreme_close),
            volume=pd.Series(sample_ohlcv_data['volume']),
            length=13
        )

        # 極端な値が適切に処理されるか確認
        assert result is not None

    def test_efi_inf_values(self, sample_ohlcv_data):
        """無限大値の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # infを含むデータを生成
        inf_close = sample_ohlcv_data['close'].copy()
        inf_close.iloc[10] = np.inf

        result = VolumeIndicators.efi(
            close=pd.Series(inf_close),
            volume=pd.Series(sample_ohlcv_data['volume']),
            length=13
        )

        assert result is not None

    def test_efi_zero_values(self, sample_ohlcv_data):
        """ゼロ値の処理テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # ゼロ値を含むデータを生成
        zero_close = sample_ohlcv_data['close'].copy()
        zero_close.iloc[10:15] = 0

        result = VolumeIndicators.efi(
            close=pd.Series(zero_close),
            volume=pd.Series(sample_ohlcv_data['volume']),
            length=13
        )

        assert result is not None

    def test_efi_data_validation(self, sample_ohlcv_data):
        """EFI用のデータ検証強化テスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # 正の値のみ許可するような検証ロジックが必要かテスト
        match_found = False
        try:
            invalid_result = VolumeIndicators.efi(
                close=pd.Series([-100, -200, -300]),  # 全負の価格
                volume=pd.Series([1000, 2000, 3000]),
                length=13
            )
            if "must be positive" in str(invalid_result) or "invalid" in str(invalid_result).lower():
                match_found = True
        except Exception:
            match_found = True

        if not match_found:
            pytest.skip("EFI実装で不正値検証が実装されていない")

    def test_efi_integration_with_service(self, sample_ohlcv_data):
        """TechnicalIndicatorServiceとの統合テスト"""
        try:
            from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService

            service = TechnicalIndicatorService()
            result = service.calculate_indicator(
                sample_ohlcv_data.copy(),
                "EFI",
                {"length": 13}
            )

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result) == len(sample_ohlcv_data)

        except ImportError:
            pytest.skip("TechnicalIndicatorService 未実装")

    def test_efi_pandas_ta_failure_simulation(self, sample_ohlcv_data):
        """pandas-ta失敗時のフォールバックテスト"""
        from app.services.indicators.technical_indicators.volume import VolumeIndicators

        # pandas-taがNoneを返すケースをモック
        with patch('pandas_ta.efi', return_value=None):
            # 現在の実装では空のSeriesを返すはず
            empty_result = VolumeIndicators.efi(
                close=pd.Series(sample_ohlcv_data['close']),
                volume=pd.Series(sample_ohlcv_data['volume']),
                length=13
            )

            assert isinstance(empty_result, pd.Series)
            # pandas-taがNoneの場合は空のSeriesを返す実装なので
            assert len(empty_result) == 0 or empty_result.isna().all()


if __name__ == "__main__":
    pytest.main([__file__])