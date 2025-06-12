"""
VWMA (Volume Weighted Moving Average) 指標のテスト

TDD方式でVWMAIndicatorクラスの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# テスト対象のインポート（まだ実装されていないのでImportErrorが発生する予定）
try:
    from app.core.services.indicators.trend_indicators import VWMAIndicator
    from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
except ImportError:
    # まだ実装されていない場合はNoneを設定
    VWMAIndicator = None
    TrendAdapter = None


class TestVWMAIndicator:
    """VWMAIndicatorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テストデータの作成
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # より現実的な価格・出来高データを生成
        base_price = 100
        price_trend = np.linspace(0, 10, 100)
        price_noise = np.random.normal(0, 1, 100)
        prices = base_price + price_trend + price_noise
        
        # 出来高は価格変動と逆相関させる（現実的なパターン）
        volumes = 10000 + np.random.uniform(-2000, 2000, 100)
        
        self.test_data = pd.DataFrame({
            'open': prices + np.random.uniform(-0.5, 0.5, 100),
            'high': prices + np.random.uniform(0.5, 1.5, 100),
            'low': prices + np.random.uniform(-1.5, -0.5, 100),
            'close': prices,
            'volume': volumes
        }, index=self.dates)

    def test_vwma_indicator_import(self):
        """VWMAIndicatorクラスがインポートできることをテスト"""
        # Red: まだ実装されていないのでNoneになっているはず
        assert VWMAIndicator is not None, "VWMAIndicatorクラスが実装されていません"

    def test_vwma_indicator_initialization(self):
        """VWMAIndicatorの初期化テスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        
        # 基本属性の確認
        assert indicator.indicator_type == "VWMA"
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0
        
        # 期待される期間が含まれているか
        expected_periods = [10, 20, 30, 50]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_vwma_calculation_basic(self):
        """VWMA計算の基本テスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        period = 20
        
        # モックを使用してTrendAdapter.vwmaをテスト
        with patch.object(TrendAdapter, 'vwma') as mock_vwma:
            # モックの戻り値を設定
            expected_result = pd.Series(
                np.random.uniform(100, 110, 100), 
                index=self.test_data.index,
                name=f"VWMA_{period}"
            )
            mock_vwma.return_value = expected_result
            
            # VWMA計算を実行
            result = indicator.calculate(self.test_data, period)
            
            # 結果の検証
            assert isinstance(result, pd.Series)
            assert len(result) == len(self.test_data)
            assert result.name == f"VWMA_{period}"
            
            # TrendAdapter.vwmaが正しい引数で呼ばれたか確認
            mock_vwma.assert_called_once_with(
                self.test_data["close"], 
                self.test_data["volume"], 
                period
            )

    def test_vwma_calculation_different_periods(self):
        """異なる期間でのVWMA計算テスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        
        for period in [10, 20, 30]:
            with patch.object(TrendAdapter, 'vwma') as mock_vwma:
                expected_result = pd.Series(
                    np.random.uniform(100, 110, 100), 
                    index=self.test_data.index,
                    name=f"VWMA_{period}"
                )
                mock_vwma.return_value = expected_result
                
                result = indicator.calculate(self.test_data, period)
                
                assert isinstance(result, pd.Series)
                assert result.name == f"VWMA_{period}"
                mock_vwma.assert_called_once_with(
                    self.test_data["close"], 
                    self.test_data["volume"], 
                    period
                )

    def test_vwma_description(self):
        """VWMA説明文のテスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        description = indicator.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "VWMA" in description or "出来高" in description
        assert "移動平均" in description

    def test_vwma_parameter_validation(self):
        """VWMAパラメータ検証のテスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        
        # 無効な期間でのテスト
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, 0)
            
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, -1)

    def test_vwma_missing_volume_data(self):
        """出来高データなしでのVWMAテスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        
        # 出来高データを除いたDataFrame
        data_without_volume = self.test_data.drop(columns=['volume'])
        
        with pytest.raises(Exception):
            indicator.calculate(data_without_volume, 20)

    def test_vwma_empty_data(self):
        """空データでのVWMAテスト"""
        if VWMAIndicator is None:
            pytest.skip("VWMAIndicatorが実装されていません")
            
        indicator = VWMAIndicator()
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        with pytest.raises(Exception):
            indicator.calculate(empty_data, 20)


class TestTrendAdapterVWMA:
    """TrendAdapterのVWMAメソッドのテスト"""

    def setup_method(self):
        """テスト初期化"""
        self.test_close = pd.Series(
            np.random.uniform(100, 110, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='close'
        )
        self.test_volume = pd.Series(
            np.random.uniform(1000, 10000, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='volume'
        )

    def test_trend_adapter_vwma_method_exists(self):
        """TrendAdapter.vwmaメソッドが存在することをテスト"""
        # Red: まだ実装されていないのでAttributeErrorが発生する予定
        assert hasattr(TrendAdapter, 'vwma'), "TrendAdapter.vwmaメソッドが実装されていません"

    def test_trend_adapter_vwma_calculation(self):
        """TrendAdapter.vwmaの計算テスト"""
        if not hasattr(TrendAdapter, 'vwma'):
            pytest.skip("TrendAdapter.vwmaが実装されていません")
            
        period = 20
        result = TrendAdapter.vwma(self.test_close, self.test_volume, period)
        
        # 結果の検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_close)
        assert result.name == f"VWMA_{period}"

    def test_trend_adapter_vwma_different_periods(self):
        """TrendAdapter.vwmaの異なる期間でのテスト"""
        if not hasattr(TrendAdapter, 'vwma'):
            pytest.skip("TrendAdapter.vwmaが実装されていません")
            
        for period in [10, 20, 30]:
            result = TrendAdapter.vwma(self.test_close, self.test_volume, period)
            
            assert isinstance(result, pd.Series)
            assert result.name == f"VWMA_{period}"

    def test_trend_adapter_vwma_parameter_validation(self):
        """TrendAdapter.vwmaのパラメータ検証テスト"""
        if not hasattr(TrendAdapter, 'vwma'):
            pytest.skip("TrendAdapter.vwmaが実装されていません")
            
        # 無効なパラメータでのテスト
        with pytest.raises(Exception):
            TrendAdapter.vwma(self.test_close, self.test_volume, 0)
            
        with pytest.raises(Exception):
            TrendAdapter.vwma(self.test_close, self.test_volume, -1)


class TestVWMAIntegration:
    """VWMAの統合テスト"""

    def test_vwma_in_trend_indicators_factory(self):
        """get_trend_indicator関数でVWMAが取得できることをテスト"""
        try:
            from app.core.services.indicators.trend_indicators import get_trend_indicator
            
            # Red: まだVWMAが追加されていないのでValueErrorが発生する予定
            indicator = get_trend_indicator("VWMA")
            assert indicator.indicator_type == "VWMA"
            
        except (ImportError, ValueError):
            pytest.fail("VWMAがget_trend_indicator関数に追加されていません")

    def test_vwma_in_indicators_info(self):
        """TREND_INDICATORS_INFOにVWMAが含まれることをテスト"""
        try:
            from app.core.services.indicators.trend_indicators import TREND_INDICATORS_INFO
            
            # Red: まだVWMAが追加されていないのでKeyErrorが発生する予定
            assert "VWMA" in TREND_INDICATORS_INFO
            
            vwma_info = TREND_INDICATORS_INFO["VWMA"]
            assert "periods" in vwma_info
            assert "description" in vwma_info
            assert "category" in vwma_info
            assert vwma_info["category"] == "trend"
            
        except (ImportError, KeyError):
            pytest.fail("VWMAがTREND_INDICATORS_INFOに追加されていません")

    def test_vwma_in_main_indicators_module(self):
        """メインのindicatorsモジュールでVWMAが利用できることをテスト"""
        try:
            from app.core.services.indicators import VWMAIndicator, get_indicator_by_type
            
            # VWMAIndicatorの直接インポート
            assert VWMAIndicator is not None
            
            # ファクトリー関数経由での取得
            indicator = get_indicator_by_type("VWMA")
            assert indicator.indicator_type == "VWMA"
            
        except (ImportError, ValueError):
            pytest.fail("VWMAがメインのindicatorsモジュールに統合されていません")


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
