"""
インジケーターJSON形式統合テスト

新しいJSON形式のインジケーター設定が既存のシステムと
正常に統合できることを確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backend.app.core.services.indicators.config import (
    indicator_registry,
)
from backend.app.core.services.indicators.technical_indicators.momentum import MomentumIndicators
from backend.app.core.services.indicators.technical_indicators.trend import TrendIndicators
from backend.app.core.services.indicators.technical_indicators.volatility import VolatilityIndicators
from backend.app.core.services.indicators.technical_indicators.volume import VolumeIndicators


class TestIndicatorJSONIntegration:
    """インジケーターJSON形式統合テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # 価格データを生成
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2
        open_prices = close_prices + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            },
            index=dates,
        )

    def test_indicator_registry_initialization(self):
        """インジケーターレジストリの初期化テスト"""
        # レジストリが正常に初期化されていることを確認
        indicators = indicator_registry.list_indicators()

        # 主要なインジケーターが登録されていることを確認
        expected_indicators = [
            "RSI",
            "APO",
            "PPO",
            "MACD",
            "SMA",
            "EMA",
            "ATR",
            "BB",
            "OBV",
            "ADOSC",
        ]
        for indicator in expected_indicators:
            assert indicator in indicators, f"{indicator} not found in registry"

    def test_json_name_generation(self):
        """JSON形式の名前生成テスト"""
        # RSI
        rsi_json_name = indicator_registry.generate_json_name("RSI")
        assert rsi_json_name == "RSI"

        # APO
        apo_json_name = indicator_registry.generate_json_name("APO")
        assert apo_json_name == "APO"

    def test_momentum_adapter_json_integration(self, sample_data):
        """MomentumAdapterのJSON形式統合テスト"""
        # RSI計算
        rsi_result = MomentumIndicators.rsi(sample_data["close"].to_numpy(), period=14)

        # 結果の検証
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(sample_data)

        # APO計算
        apo_result = MomentumIndicators.apo(
            sample_data["close"].to_numpy(), fast_period=12, slow_period=26, matype=0
        )

        assert isinstance(apo_result, np.ndarray)
        assert len(apo_result) == len(sample_data)

    def test_trend_adapter_json_integration(self, sample_data):
        """TrendAdapterのJSON形式統合テスト"""
        # SMA計算
        sma_result = TrendIndicators.sma(sample_data["close"].to_numpy(), period=20)

        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(sample_data)

    def test_ema_adapter_json_integration(self, sample_data):
        """EMA計算のJSON形式統合テスト"""
        ema_result = TrendIndicators.ema(sample_data["close"].to_numpy(), period=20)

        assert isinstance(ema_result, np.ndarray)
        assert len(ema_result) == len(sample_data)

    def test_volatility_adapter_json_integration(self, sample_data):
        """VolatilityAdapterのJSON形式統合テスト"""
        # ATR計算
        atr_result = VolatilityIndicators.atr(
            sample_data["high"].to_numpy(), sample_data["low"].to_numpy(), sample_data["close"].to_numpy(), period=14
        )

        assert isinstance(atr_result, np.ndarray)
        assert len(atr_result) == len(sample_data)

        # Bollinger Bands計算
        bb_result = VolatilityIndicators.bollinger_bands(
            sample_data["close"].to_numpy(), period=20, std_dev=2.0
        )

        assert isinstance(bb_result, tuple)
        assert len(bb_result) == 3

        for band in bb_result:
            assert isinstance(band, np.ndarray)
            assert len(band) == len(sample_data)

    def test_volume_adapter_json_integration(self, sample_data):
        """VolumeAdapterのJSON形式統合テスト"""
        # ADOSC計算
        adosc_result = VolumeIndicators.adosc(
            sample_data["high"].to_numpy(),
            sample_data["low"].to_numpy(),
            sample_data["close"].to_numpy(),
            sample_data["volume"].to_numpy(),
            fast_period=3,
            slow_period=10,
        )

        assert isinstance(adosc_result, np.ndarray)
        assert len(adosc_result) == len(sample_data)

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しないインジケーター
        unknown_config = indicator_registry.get_indicator_config("UNKNOWN")
        assert unknown_config is None


if __name__ == "__main__":
    pytest.main([__file__])
