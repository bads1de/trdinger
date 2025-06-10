"""
新しいモメンタム系指標のテスト

ADX、Aroon、MFI指標のテストを実行します。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from backend.app.core.services.indicators.talib_adapter import (
    TALibAdapter,
    TALibCalculationError,
)
from backend.app.core.services.indicators.momentum_indicators import (
    ADXIndicator,
    AroonIndicator,
    MFIIndicator,
)


class TestTALibAdapterNewIndicators:
    """TALibAdapterの新しい指標メソッドのテスト"""

    def setup_method(self):
        """テスト用のサンプルデータを準備"""
        np.random.seed(42)
        self.sample_size = 100

        # サンプルOHLCVデータ
        dates = pd.date_range("2023-01-01", periods=self.sample_size, freq="D")
        self.high = pd.Series(
            np.random.uniform(100, 110, self.sample_size), index=dates
        )
        self.low = pd.Series(np.random.uniform(90, 100, self.sample_size), index=dates)
        self.close = pd.Series(
            np.random.uniform(95, 105, self.sample_size), index=dates
        )
        self.volume = pd.Series(
            np.random.uniform(1000, 10000, self.sample_size), index=dates
        )

    def test_adx_calculation(self):
        """ADX計算のテスト"""
        period = 14
        result = TALibAdapter.adx(self.high, self.low, self.close, period)

        # 基本的な検証
        assert isinstance(result, pd.Series)
        assert result.name == f"ADX_{period}"
        assert len(result) == len(self.close)

        # ADXは0-100の範囲
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_aroon_calculation(self):
        """Aroon計算のテスト"""
        period = 14
        result = TALibAdapter.aroon(self.high, self.low, period)

        # 基本的な検証
        assert isinstance(result, dict)
        assert "aroon_down" in result
        assert "aroon_up" in result

        aroon_down = result["aroon_down"]
        aroon_up = result["aroon_up"]

        assert isinstance(aroon_down, pd.Series)
        assert isinstance(aroon_up, pd.Series)
        assert len(aroon_down) == len(self.high)
        assert len(aroon_up) == len(self.high)

        # Aroonは0-100の範囲
        valid_down = aroon_down.dropna()
        valid_up = aroon_up.dropna()
        assert all(0 <= val <= 100 for val in valid_down)
        assert all(0 <= val <= 100 for val in valid_up)

    def test_mfi_calculation(self):
        """MFI計算のテスト"""
        period = 14
        result = TALibAdapter.mfi(self.high, self.low, self.close, self.volume, period)

        # 基本的な検証
        assert isinstance(result, pd.Series)
        assert result.name == f"MFI_{period}"
        assert len(result) == len(self.close)

        # MFIは0-100の範囲
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_adx_input_validation(self):
        """ADX入力検証のテスト"""
        # データ長不一致
        short_high = self.high[:50]
        with pytest.raises(TALibCalculationError, match="データ長が一致しません"):
            TALibAdapter.adx(short_high, self.low, self.close, 14)

        # 期間が長すぎる
        with pytest.raises(
            TALibCalculationError, match="データ長.*が期間.*より短いです"
        ):
            TALibAdapter.adx(self.high[:10], self.low[:10], self.close[:10], 14)

    def test_mfi_input_validation(self):
        """MFI入力検証のテスト"""
        # データ長不一致
        short_volume = self.volume[:50]
        with pytest.raises(TALibCalculationError, match="データ長が一致しません"):
            TALibAdapter.mfi(self.high, self.low, self.close, short_volume, 14)


class TestADXIndicator:
    """ADXIndicatorクラスのテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.indicator = ADXIndicator()

        # サンプルデータフレーム
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """初期化のテスト"""
        assert self.indicator.indicator_type == "ADX"
        assert self.indicator.supported_periods == [14, 21]

    def test_calculate(self):
        """計算メソッドのテスト"""
        period = 14
        result = self.indicator.calculate(self.df, period)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.df)

        # ADXの値域確認
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_get_description(self):
        """説明取得のテスト"""
        description = self.indicator.get_description()
        assert "ADX" in description
        assert "トレンドの強さ" in description


class TestAroonIndicator:
    """AroonIndicatorクラスのテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.indicator = AroonIndicator()

        # サンプルデータフレーム
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """初期化のテスト"""
        assert self.indicator.indicator_type == "AROON"
        assert self.indicator.supported_periods == [14, 25]

    def test_calculate(self):
        """計算メソッドのテスト"""
        period = 14
        result = self.indicator.calculate(self.df, period)

        assert isinstance(result, pd.DataFrame)
        assert "aroon_down" in result.columns
        assert "aroon_up" in result.columns
        assert len(result) == len(self.df)

        # Aroonの値域確認
        for col in ["aroon_down", "aroon_up"]:
            valid_values = result[col].dropna()
            assert all(0 <= val <= 100 for val in valid_values)

    def test_get_description(self):
        """説明取得のテスト"""
        description = self.indicator.get_description()
        assert "Aroon" in description
        assert "トレンドの変化" in description


class TestMFIIndicator:
    """MFIIndicatorクラスのテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.indicator = MFIIndicator()

        # サンプルデータフレーム
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """初期化のテスト"""
        assert self.indicator.indicator_type == "MFI"
        assert self.indicator.supported_periods == [14, 21]

    def test_calculate(self):
        """計算メソッドのテスト"""
        period = 14
        result = self.indicator.calculate(self.df, period)

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.df)

        # MFIの値域確認
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_calculate_without_volume(self):
        """出来高なしでの計算エラーテスト"""
        df_no_volume = self.df.drop("volume", axis=1)

        with pytest.raises(ValueError, match="MFI計算には出来高データが必要です"):
            self.indicator.calculate(df_no_volume, 14)

    def test_get_description(self):
        """説明取得のテスト"""
        description = self.indicator.get_description()
        assert "MFI" in description
        assert "出来高を考慮した" in description


if __name__ == "__main__":
    pytest.main([__file__])
