"""
Ultimate Oscillator 指標のテスト

TDD方式でUltimateOscillatorIndicatorクラスの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# テスト対象のインポート（まだ実装されていないのでImportErrorが発生する予定）
try:
    from app.core.services.indicators.momentum_indicators import UltimateOscillatorIndicator
    from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
except ImportError:
    # まだ実装されていない場合はNoneを設定
    UltimateOscillatorIndicator = None
    MomentumAdapter = None


class TestUltimateOscillatorIndicator:
    """UltimateOscillatorIndicatorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テストデータの作成
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格データを生成
        base_price = 100
        price_trend = np.linspace(0, 10, 100)
        price_noise = np.random.normal(0, 1, 100)
        close_prices = base_price + price_trend + price_noise
        
        # 高値・安値を終値から生成
        high_prices = close_prices + np.random.uniform(0.5, 1.5, 100)
        low_prices = close_prices - np.random.uniform(0.5, 1.5, 100)
        
        self.test_data = pd.DataFrame({
            'open': close_prices + np.random.uniform(-0.5, 0.5, 100),
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=self.dates)

    def test_ultimate_oscillator_indicator_import(self):
        """UltimateOscillatorIndicatorクラスがインポートできることをテスト"""
        # Red: まだ実装されていないのでNoneになっているはず
        assert UltimateOscillatorIndicator is not None, "UltimateOscillatorIndicatorクラスが実装されていません"

    def test_ultimate_oscillator_indicator_initialization(self):
        """UltimateOscillatorIndicatorの初期化テスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        
        # 基本属性の確認
        assert indicator.indicator_type == "ULTOSC"
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0
        
        # 期待される期間が含まれているか
        expected_periods = [7, 14, 28]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_ultimate_oscillator_calculation_basic(self):
        """Ultimate Oscillator計算の基本テスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        period = 7  # 短期期間
        
        # モックを使用してMomentumAdapter.ultimate_oscillatorをテスト
        with patch.object(MomentumAdapter, 'ultimate_oscillator') as mock_ultosc:
            # モックの戻り値を設定
            expected_result = pd.Series(
                np.random.uniform(0, 100, 100),
                index=self.test_data.index,
                name='ULTOSC_7_14_28'
            )
            mock_ultosc.return_value = expected_result
            
            # Ultimate Oscillator計算を実行
            result = indicator.calculate(self.test_data, period)
            
            # 結果の検証
            assert isinstance(result, pd.Series)
            assert len(result) == len(self.test_data)
            
            # MomentumAdapter.ultimate_oscillatorが正しい引数で呼ばれたか確認
            mock_ultosc.assert_called_once_with(
                self.test_data["high"], 
                self.test_data["low"], 
                self.test_data["close"], 
                period,
                period * 2,
                period * 4
            )

    def test_ultimate_oscillator_calculation_different_periods(self):
        """異なる期間でのUltimate Oscillator計算テスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        
        for period in [7, 14, 28]:
            with patch.object(MomentumAdapter, 'ultimate_oscillator') as mock_ultosc:
                expected_result = pd.Series(
                    np.random.uniform(0, 100, 100),
                    index=self.test_data.index,
                    name=f'ULTOSC_{period}_{period*2}_{period*4}'
                )
                mock_ultosc.return_value = expected_result
                
                result = indicator.calculate(self.test_data, period)
                
                assert isinstance(result, pd.Series)
                mock_ultosc.assert_called_once_with(
                    self.test_data["high"], 
                    self.test_data["low"], 
                    self.test_data["close"], 
                    period,
                    period * 2,
                    period * 4
                )

    def test_ultimate_oscillator_description(self):
        """Ultimate Oscillator説明文のテスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        description = indicator.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "Ultimate" in description or "アルティメット" in description

    def test_ultimate_oscillator_parameter_validation(self):
        """Ultimate Oscillatorパラメータ検証のテスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        
        # 無効な期間でのテスト
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, 0)
            
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, -1)

    def test_ultimate_oscillator_empty_data(self):
        """空データでのUltimate Oscillatorテスト"""
        if UltimateOscillatorIndicator is None:
            pytest.skip("UltimateOscillatorIndicatorが実装されていません")
            
        indicator = UltimateOscillatorIndicator()
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        with pytest.raises(Exception):
            indicator.calculate(empty_data, 7)


class TestMomentumAdapterUltimateOscillator:
    """MomentumAdapterのUltimate Oscillatorメソッドのテスト"""

    def setup_method(self):
        """テスト初期化"""
        self.test_high = pd.Series(
            np.random.uniform(105, 115, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='high'
        )
        self.test_low = pd.Series(
            np.random.uniform(95, 105, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='low'
        )
        self.test_close = pd.Series(
            np.random.uniform(100, 110, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='close'
        )

    def test_momentum_adapter_ultimate_oscillator_method_exists(self):
        """MomentumAdapter.ultimate_oscillatorメソッドが存在することをテスト"""
        # Red: まだ実装されていないのでAttributeErrorが発生する予定
        assert hasattr(MomentumAdapter, 'ultimate_oscillator'), "MomentumAdapter.ultimate_oscillatorメソッドが実装されていません"

    def test_momentum_adapter_ultimate_oscillator_calculation(self):
        """MomentumAdapter.ultimate_oscillatorの計算テスト"""
        if not hasattr(MomentumAdapter, 'ultimate_oscillator'):
            pytest.skip("MomentumAdapter.ultimate_oscillatorが実装されていません")
            
        period1, period2, period3 = 7, 14, 28
        result = MomentumAdapter.ultimate_oscillator(
            self.test_high, self.test_low, self.test_close, period1, period2, period3
        )
        
        # 結果の検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_close)

    def test_momentum_adapter_ultimate_oscillator_different_periods(self):
        """MomentumAdapter.ultimate_oscillatorの異なる期間でのテスト"""
        if not hasattr(MomentumAdapter, 'ultimate_oscillator'):
            pytest.skip("MomentumAdapter.ultimate_oscillatorが実装されていません")
            
        period_sets = [(7, 14, 28), (10, 20, 40)]
        
        for period1, period2, period3 in period_sets:
            result = MomentumAdapter.ultimate_oscillator(
                self.test_high, self.test_low, self.test_close, period1, period2, period3
            )
            
            assert isinstance(result, pd.Series)

    def test_momentum_adapter_ultimate_oscillator_parameter_validation(self):
        """MomentumAdapter.ultimate_oscillatorのパラメータ検証テスト"""
        if not hasattr(MomentumAdapter, 'ultimate_oscillator'):
            pytest.skip("MomentumAdapter.ultimate_oscillatorが実装されていません")
            
        # 無効なパラメータでのテスト
        with pytest.raises(Exception):
            MomentumAdapter.ultimate_oscillator(
                self.test_high, self.test_low, self.test_close, 0, 14, 28
            )
            
        with pytest.raises(Exception):
            MomentumAdapter.ultimate_oscillator(
                self.test_high, self.test_low, self.test_close, 7, -1, 28
            )


class TestUltimateOscillatorIntegration:
    """Ultimate Oscillatorの統合テスト"""

    def test_ultimate_oscillator_in_momentum_indicators_factory(self):
        """get_momentum_indicator関数でUltimate Oscillatorが取得できることをテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import get_momentum_indicator
            
            # Red: まだUltimate Oscillatorが追加されていないのでValueErrorが発生する予定
            indicator = get_momentum_indicator("ULTOSC")
            assert indicator.indicator_type == "ULTOSC"
            
        except (ImportError, ValueError):
            pytest.fail("Ultimate Oscillatorがget_momentum_indicator関数に追加されていません")

    def test_ultimate_oscillator_in_indicators_info(self):
        """MOMENTUM_INDICATORS_INFOにUltimate Oscillatorが含まれることをテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import MOMENTUM_INDICATORS_INFO
            
            # Red: まだUltimate Oscillatorが追加されていないのでKeyErrorが発生する予定
            assert "ULTOSC" in MOMENTUM_INDICATORS_INFO
            
            ultosc_info = MOMENTUM_INDICATORS_INFO["ULTOSC"]
            assert "periods" in ultosc_info
            assert "description" in ultosc_info
            assert "category" in ultosc_info
            assert ultosc_info["category"] == "momentum"
            
        except (ImportError, KeyError):
            pytest.fail("Ultimate OscillatorがMOMENTUM_INDICATORS_INFOに追加されていません")

    def test_ultimate_oscillator_in_main_indicators_module(self):
        """メインのindicatorsモジュールでUltimate Oscillatorが利用できることをテスト"""
        try:
            from app.core.services.indicators import UltimateOscillatorIndicator, get_indicator_by_type
            
            # UltimateOscillatorIndicatorの直接インポート
            assert UltimateOscillatorIndicator is not None
            
            # ファクトリー関数経由での取得
            indicator = get_indicator_by_type("ULTOSC")
            assert indicator.indicator_type == "ULTOSC"
            
        except (ImportError, ValueError):
            pytest.fail("Ultimate Oscillatorがメインのindicatorsモジュールに統合されていません")


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
