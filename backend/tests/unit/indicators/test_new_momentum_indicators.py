#!/usr/bin/env python3
"""
新規追加モメンタム指標のテスト
BOP, APO, PPO, AROONOSC, DX指標のテスト
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from app.core.services.indicators.momentum_indicators import (
    BOPIndicator,
    APOIndicator,
    PPOIndicator,
    AROONOSCIndicator,
    DXIndicator,
    get_momentum_indicator
)
from app.core.services.indicators.exceptions import TALibCalculationError


class TestNewMomentumIndicators:
    """新規追加モメンタム指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': close_prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100),
        }, index=dates)

    def test_bop_indicator(self, sample_data):
        """BOP指標のテスト"""
        indicator = BOPIndicator()
        
        # 基本計算テスト
        result = indicator.calculate(sample_data, 1)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "BOP"
        
        # 値の範囲テスト（-1から1の範囲）
        valid_values = result.dropna()
        assert all(valid_values >= -1)
        assert all(valid_values <= 1)
        
        # 説明テスト
        assert "Balance Of Power" in indicator.get_description()

    def test_apo_indicator(self, sample_data):
        """APO指標のテスト"""
        indicator = APOIndicator()
        
        # 基本計算テスト
        result = indicator.calculate(sample_data, 12, slow_period=26)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert "APO_12_26" in result.name
        
        # 異なるパラメータでのテスト
        result2 = indicator.calculate(sample_data, 10, slow_period=20, matype=1)
        assert "APO_10_20" in result2.name
        
        # 説明テスト
        assert "Absolute Price Oscillator" in indicator.get_description()

    def test_ppo_indicator(self, sample_data):
        """PPO指標のテスト"""
        indicator = PPOIndicator()
        
        # 基本計算テスト
        result = indicator.calculate(sample_data, 12, slow_period=26)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert "PPO_12_26" in result.name
        
        # 異なるパラメータでのテスト
        result2 = indicator.calculate(sample_data, 10, slow_period=20, matype=1)
        assert "PPO_10_20" in result2.name
        
        # 説明テスト
        assert "Percentage Price Oscillator" in indicator.get_description()

    def test_aroonosc_indicator(self, sample_data):
        """AROONOSC指標のテスト"""
        indicator = AROONOSCIndicator()
        
        # 基本計算テスト
        result = indicator.calculate(sample_data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "AROONOSC_14"
        
        # 値の範囲テスト（-100から100の範囲）
        valid_values = result.dropna()
        assert all(valid_values >= -100)
        assert all(valid_values <= 100)
        
        # 異なる期間でのテスト
        result2 = indicator.calculate(sample_data, 25)
        assert result2.name == "AROONOSC_25"
        
        # 説明テスト
        assert "Aroon Oscillator" in indicator.get_description()

    def test_dx_indicator(self, sample_data):
        """DX指標のテスト"""
        indicator = DXIndicator()
        
        # 基本計算テスト
        result = indicator.calculate(sample_data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "DX_14"
        
        # 値の範囲テスト（0から100の範囲）
        valid_values = result.dropna()
        assert all(valid_values >= 0)
        assert all(valid_values <= 100)
        
        # 異なる期間でのテスト
        result2 = indicator.calculate(sample_data, 21)
        assert result2.name == "DX_21"
        
        # 説明テスト
        assert "Directional Movement Index" in indicator.get_description()

    def test_factory_function(self, sample_data):
        """ファクトリー関数のテスト"""
        # 新規指標のファクトリー関数テスト
        new_indicators = ["BOP", "APO", "PPO", "AROONOSC", "DX"]
        
        for indicator_type in new_indicators:
            indicator = get_momentum_indicator(indicator_type)
            assert indicator is not None
            assert indicator.indicator_type == indicator_type
            
            # 計算テスト
            if indicator_type == "BOP":
                result = indicator.calculate(sample_data, 1)
            elif indicator_type in ["APO", "PPO"]:
                result = indicator.calculate(sample_data, 12, slow_period=26)
            else:
                result = indicator.calculate(sample_data, 14)
            
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)

    def test_error_handling(self, sample_data):
        """エラーハンドリングのテスト"""
        # 不正なデータでのテスト
        invalid_data = sample_data.drop(columns=['high'])
        
        # AROONOSC（high, lowが必要）
        aroonosc = AROONOSCIndicator()
        with pytest.raises(ValueError, match="AROONOSC計算には"):
            aroonosc.calculate(invalid_data, 14)
        
        # DX（high, low, closeが必要）
        dx = DXIndicator()
        with pytest.raises(ValueError, match="DX計算には"):
            dx.calculate(invalid_data, 14)
        
        # BOP（open, high, low, closeが必要）
        bop = BOPIndicator()
        invalid_data2 = sample_data.drop(columns=['open'])
        with pytest.raises(ValueError, match="BOP計算には"):
            bop.calculate(invalid_data2, 1)

    def test_supported_periods(self):
        """サポート期間のテスト"""
        # 各指標のサポート期間確認
        bop = BOPIndicator()
        assert bop.supported_periods == [1]
        
        apo = APOIndicator()
        assert 12 in apo.supported_periods
        assert 26 in apo.supported_periods
        
        ppo = PPOIndicator()
        assert 12 in ppo.supported_periods
        assert 26 in ppo.supported_periods
        
        aroonosc = AROONOSCIndicator()
        assert 14 in aroonosc.supported_periods
        assert 25 in aroonosc.supported_periods
        
        dx = DXIndicator()
        assert 14 in dx.supported_periods
        assert 21 in dx.supported_periods

    def test_calculation_consistency(self, sample_data):
        """計算の一貫性テスト"""
        # 同じパラメータで複数回計算して結果が一致することを確認
        bop = BOPIndicator()
        result1 = bop.calculate(sample_data, 1)
        result2 = bop.calculate(sample_data, 1)
        
        pd.testing.assert_series_equal(result1, result2)
        
        # APOとPPOの関係性テスト（PPOはAPOのパーセンテージ版）
        apo = APOIndicator()
        ppo = PPOIndicator()
        
        apo_result = apo.calculate(sample_data, 12, slow_period=26)
        ppo_result = ppo.calculate(sample_data, 12, slow_period=26)
        
        # PPOとAPOは異なる値を持つべき（PPOはパーセンテージ）
        assert not apo_result.equals(ppo_result)

    def test_integration_with_auto_strategy(self, sample_data):
        """オートストラテジー統合テスト"""
        # 新規指標がオートストラテジーで使用可能かテスト
        from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
        
        generator = RandomGeneGenerator()
        
        # 新規指標が利用可能リストに含まれているかテスト
        new_indicators = ["BOP", "APO", "PPO", "AROONOSC", "DX"]
        for indicator in new_indicators:
            assert indicator in generator.available_indicators
        
        # パラメータ生成テスト
        for indicator in new_indicators:
            params = generator._generate_indicator_parameters(indicator)
            assert isinstance(params, dict)
            assert "period" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
