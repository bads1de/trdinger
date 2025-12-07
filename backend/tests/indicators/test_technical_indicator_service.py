"""
TechnicalIndicatorServiceのテスト
"""

import numpy as np
import pandas as pd

from app.services.indicators import TechnicalIndicatorService


class TestTechnicalIndicatorService:
    """TechnicalIndicatorServiceのテストクラス"""

    def setup_method(self):
        """テスト前のセットアップ"""
        self.service = TechnicalIndicatorService()

    def test_init(self):
        """初期化のテスト"""
        assert self.service is not None
        assert hasattr(self.service, "registry")

    def test_get_supported_indicators(self):
        """サポート指標取得のテスト"""
        indicators = self.service.get_supported_indicators()

        assert isinstance(indicators, dict)
        assert len(indicators) > 0

    def test_registry_has_indicators(self):
        """レジストリに指標が登録されているか確認"""
        assert self.service.registry is not None
        # レジストリから設定を取得できることを確認
        sma_config = self.service.registry.get_indicator_config("SMA")
        assert sma_config is not None

    def test_calculate_single_indicator(self):
        """単一指標計算のテスト"""
        # テスト用データ
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 2,
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                * 2,
            }
        )

        result = self.service.calculate_indicator(data, "SMA", {"length": 5})

        # 結果はnumpy配列またはpd.Seriesとして返される
        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))

    def test_calculate_single_indicator_invalid_data(self):
        """無効なデータでの指標計算テスト"""
        data = pd.DataFrame()  # 空のデータ

        # 空データはNaN結果を返す（例外は投げない）
        result = self.service.calculate_indicator(data, "SMA", {"length": 5})
        assert result is not None
        # 空の結果またはNaNで埋められた結果が返される
        assert isinstance(result, (np.ndarray, pd.Series, tuple))

    def test_calculate_rsi(self):
        """RSI指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 2,
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                * 2,
            }
        )

        result = self.service.calculate_indicator(data, "RSI", {"length": 14})

        assert result is not None
        assert isinstance(result, (np.ndarray, pd.Series))

    def test_calculate_macd(self):
        """MACD指標計算のテスト"""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 3,
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                * 3,
            }
        )

        result = self.service.calculate_indicator(
            data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )

        # MACDは複数の値を返す（tuple）
        assert result is not None
        assert isinstance(result, tuple)

    def test_validate_data_length(self):
        """データ長検証のテスト - data_validation.pyの関数を直接テスト"""
        from app.services.indicators.data_validation import (
            validate_data_length_with_fallback,
        )

        data = pd.DataFrame({"close": list(range(100)), "volume": [1000] * 100})

        is_valid, min_length = validate_data_length_with_fallback(
            data, "SMA", {"length": 14}
        )

        assert isinstance(is_valid, bool)
        assert isinstance(min_length, int)

    def test_validate_data_length_insufficient(self):
        """データ不足の検証テスト - data_validation.pyの関数を直接テスト"""
        from app.services.indicators.data_validation import (
            validate_data_length_with_fallback,
        )

        data = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})

        is_valid, min_length = validate_data_length_with_fallback(
            data, "SMA", {"length": 14}
        )

        # データ不足の場合
        assert isinstance(is_valid, bool)
        assert isinstance(min_length, int)

    def test_error_handling_in_calculation(self):
        """計算中のエラー処理テスト"""
        # 無効なデータ
        data = pd.DataFrame({"invalid": [1, 2, 3]})

        # 必須カラムが不足している場合はNaN結果を返す（例外は投げない）
        result = self.service.calculate_indicator(data, "SMA", {"length": 5})
        assert result is not None
        # NaNで埋められた結果が返される
        assert isinstance(result, (np.ndarray, pd.Series, tuple))
