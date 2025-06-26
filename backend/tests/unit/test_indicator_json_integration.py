"""
インジケーターJSON形式統合テスト

新しいJSON形式のインジケーター設定が既存のシステムと
正常に統合できることを確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.core.services.indicators.config import (
    indicator_registry,
    compatibility_manager,
    migrator
)
from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
from app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter
from app.core.services.indicators.adapters.volume_adapter import VolumeAdapter


class TestIndicatorJSONIntegration:
    """インジケーターJSON形式統合テスト"""
    
    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 価格データを生成
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2
        open_prices = close_prices + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def test_indicator_registry_initialization(self):
        """インジケーターレジストリの初期化テスト"""
        # レジストリが正常に初期化されていることを確認
        indicators = indicator_registry.list_indicators()
        
        # 主要なインジケーターが登録されていることを確認
        expected_indicators = ["RSI", "APO", "PPO", "MACD", "SMA", "EMA", "ATR", "BB", "OBV", "ADOSC"]
        for indicator in expected_indicators:
            assert indicator in indicators, f"{indicator} not found in registry"
    
    def test_json_name_generation(self):
        """JSON形式の名前生成テスト"""
        # RSI（単一パラメータ）
        rsi_json = indicator_registry.generate_json_name("RSI", {"period": 14})
        expected_rsi = {"indicator": "RSI", "parameters": {"period": 14}}
        assert rsi_json == expected_rsi
        
        # APO（複数パラメータ）
        apo_json = indicator_registry.generate_json_name("APO", {
            "fast_period": 12, "slow_period": 26, "matype": 0
        })
        expected_apo = {
            "indicator": "APO", 
            "parameters": {"fast_period": 12, "slow_period": 26, "matype": 0}
        }
        assert apo_json == expected_apo
    
    def test_legacy_name_generation(self):
        """レガシー形式の名前生成テスト"""
        # RSI
        rsi_legacy = indicator_registry.generate_legacy_name("RSI", {"period": 14})
        assert rsi_legacy == "RSI_14"
        
        # APO
        apo_legacy = indicator_registry.generate_legacy_name("APO", {
            "fast_period": 12, "slow_period": 26, "matype": 0
        })
        assert apo_legacy == "APO_12_26"
    
    def test_legacy_name_parsing(self):
        """レガシー形式の名前解析テスト"""
        # 単一パラメータ
        rsi_parsed = migrator.parse_legacy_name("RSI_14")
        expected = {"indicator": "RSI", "parameters": {"period": 14}}
        assert rsi_parsed == expected
        
        # 複数パラメータ
        apo_parsed = migrator.parse_legacy_name("APO_12_26")
        expected = {"indicator": "APO", "parameters": {"fast_period": 12, "slow_period": 26}}
        assert apo_parsed == expected
        
        # パラメータなし
        obv_parsed = migrator.parse_legacy_name("OBV")
        expected = {"indicator": "OBV", "parameters": {}}
        assert obv_parsed == expected
    
    def test_backward_compatibility_mode(self):
        """後方互換性モードのテスト"""
        # 互換性モード有効時
        compatibility_manager.enable_compatibility_mode()
        
        # レガシー形式の解決
        legacy_resolved = compatibility_manager.resolve_indicator_name("RSI_14")
        expected = {"indicator": "RSI", "parameters": {"period": 14}}
        assert legacy_resolved == expected
        
        # JSON形式の解決
        json_input = {"indicator": "RSI", "parameters": {"period": 14}}
        json_resolved = compatibility_manager.resolve_indicator_name(json_input)
        assert json_resolved == json_input
    
    def test_momentum_adapter_json_integration(self, sample_data):
        """MomentumAdapterのJSON形式統合テスト"""
        # RSI計算
        rsi_result = MomentumAdapter.rsi(sample_data['close'], period=14)
        
        # 結果の検証
        assert isinstance(rsi_result, pd.Series)
        assert len(rsi_result) == len(sample_data)
        assert rsi_result.name is not None  # 名前が設定されていることを確認
        
        # APO計算
        apo_result = MomentumAdapter.apo(
            sample_data['close'], fast_period=12, slow_period=26, matype=0
        )
        
        assert isinstance(apo_result, pd.Series)
        assert len(apo_result) == len(sample_data)
        assert apo_result.name is not None
    
    def test_trend_adapter_json_integration(self, sample_data):
        """TrendAdapterのJSON形式統合テスト"""
        # SMA計算
        sma_result = TrendAdapter.sma(sample_data['close'], period=20)
        
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(sample_data)
        assert sma_result.name is not None
        
        # EMA計算
        ema_result = TrendAdapter.ema(sample_data['close'], period=20)
        
        assert isinstance(ema_result, pd.Series)
        assert len(ema_result) == len(sample_data)
        assert ema_result.name is not None
    
    def test_volatility_adapter_json_integration(self, sample_data):
        """VolatilityAdapterのJSON形式統合テスト"""
        # ATR計算
        atr_result = VolatilityAdapter.atr(
            sample_data['high'], sample_data['low'], sample_data['close'], period=14
        )
        
        assert isinstance(atr_result, pd.Series)
        assert len(atr_result) == len(sample_data)
        assert atr_result.name is not None
        
        # Bollinger Bands計算
        bb_result = VolatilityAdapter.bollinger_bands(
            sample_data['close'], period=20, std_dev=2.0
        )
        
        assert isinstance(bb_result, dict)
        assert 'upper' in bb_result
        assert 'middle' in bb_result
        assert 'lower' in bb_result
        
        for band in bb_result.values():
            assert isinstance(band, pd.Series)
            assert len(band) == len(sample_data)
            assert band.name is not None
    
    def test_volume_adapter_json_integration(self, sample_data):
        """VolumeAdapterのJSON形式統合テスト"""
        # ADOSC計算
        adosc_result = VolumeAdapter.adosc(
            sample_data['high'], sample_data['low'], 
            sample_data['close'], sample_data['volume'],
            fast_period=3, slow_period=10
        )
        
        assert isinstance(adosc_result, pd.Series)
        assert len(adosc_result) == len(sample_data)
        assert adosc_result.name is not None
    
    def test_auto_strategy_compatibility(self):
        """オートストラテジー機能との互換性テスト"""
        # オートストラテジー用の設定が正常に動作することを確認
        
        # 互換性モードでの名前生成
        compatibility_manager.enable_compatibility_mode()
        
        # レガシー形式での名前生成（既存コードとの互換性）
        legacy_name = compatibility_manager.generate_name(
            "RSI", {"period": 14}, format_type="legacy"
        )
        assert legacy_name == "RSI_14"
        
        # JSON形式での名前生成
        json_name = compatibility_manager.generate_name(
            "RSI", {"period": 14}, format_type="json"
        )
        expected_json = {"indicator": "RSI", "parameters": {"period": 14}}
        assert json_name == expected_json
        
        # 自動選択（互換性モード有効時はレガシー形式を優先）
        auto_name = compatibility_manager.generate_name(
            "RSI", {"period": 14}, format_type="auto"
        )
        assert auto_name == "RSI_14"  # レガシー形式が返される
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しないインジケーター
        unknown_config = indicator_registry.get("UNKNOWN")
        assert unknown_config is None
        
        # 不正なレガシー形式
        invalid_parsed = migrator.parse_legacy_name("INVALID_FORMAT_123_456_789")
        assert invalid_parsed is None
        
        # 互換性モード無効時のレガシー形式
        compatibility_manager.disable_compatibility_mode()
        
        with pytest.raises(ValueError):
            compatibility_manager.resolve_indicator_name("RSI_14")
        
        # 互換性モードを再度有効化（他のテストに影響しないように）
        compatibility_manager.enable_compatibility_mode()


if __name__ == "__main__":
    pytest.main([__file__])
