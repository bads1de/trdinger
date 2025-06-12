"""
HMA (Hull Moving Average) 指標のテスト

TDD方式でHMAIndicatorクラスの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# テスト対象のインポート（まだ実装されていないのでImportErrorが発生する予定）
try:
    from app.core.services.indicators.trend_indicators import HMAIndicator
    from app.core.services.indicators.adapters.trend_adapter import TrendAdapter
except ImportError:
    # まだ実装されていない場合はNoneを設定
    HMAIndicator = None
    TrendAdapter = None


class TestHMAIndicator:
    """HMAIndicatorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テストデータの作成
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=self.dates)

    def test_hma_indicator_import(self):
        """HMAIndicatorクラスがインポートできることをテスト"""
        # Red: まだ実装されていないのでNoneになっているはず
        assert HMAIndicator is not None, "HMAIndicatorクラスが実装されていません"

    def test_hma_indicator_initialization(self):
        """HMAIndicatorの初期化テスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        
        # 基本属性の確認
        assert indicator.indicator_type == "HMA"
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0
        
        # 期待される期間が含まれているか
        expected_periods = [9, 14, 21, 30, 50]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_hma_calculation_basic(self):
        """HMA計算の基本テスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        period = 21
        
        # モックを使用してTrendAdapter.hmaをテスト
        with patch.object(TrendAdapter, 'hma') as mock_hma:
            # モックの戻り値を設定
            expected_result = pd.Series(
                np.random.uniform(100, 110, 100), 
                index=self.test_data.index,
                name=f"HMA_{period}"
            )
            mock_hma.return_value = expected_result
            
            # HMA計算を実行
            result = indicator.calculate(self.test_data, period)
            
            # 結果の検証
            assert isinstance(result, pd.Series)
            assert len(result) == len(self.test_data)
            assert result.name == f"HMA_{period}"
            
            # TrendAdapter.hmaが正しい引数で呼ばれたか確認
            mock_hma.assert_called_once_with(self.test_data["close"], period)

    def test_hma_calculation_different_periods(self):
        """異なる期間でのHMA計算テスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        
        for period in [9, 14, 21, 30]:
            with patch.object(TrendAdapter, 'hma') as mock_hma:
                expected_result = pd.Series(
                    np.random.uniform(100, 110, 100), 
                    index=self.test_data.index,
                    name=f"HMA_{period}"
                )
                mock_hma.return_value = expected_result
                
                result = indicator.calculate(self.test_data, period)
                
                assert isinstance(result, pd.Series)
                assert result.name == f"HMA_{period}"
                mock_hma.assert_called_once_with(self.test_data["close"], period)

    def test_hma_description(self):
        """HMA説明文のテスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        description = indicator.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "HMA" in description or "Hull" in description
        assert "移動平均" in description

    def test_hma_parameter_validation(self):
        """HMAパラメータ検証のテスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        
        # 無効な期間でのテスト
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, 0)
            
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, -1)

    def test_hma_empty_data(self):
        """空データでのHMAテスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        with pytest.raises(Exception):
            indicator.calculate(empty_data, 21)

    def test_hma_insufficient_data(self):
        """データ不足でのHMAテスト"""
        if HMAIndicator is None:
            pytest.skip("HMAIndicatorが実装されていません")
            
        indicator = HMAIndicator()
        
        # 期間より少ないデータ
        short_data = self.test_data.head(10)
        period = 21
        
        with pytest.raises(Exception):
            indicator.calculate(short_data, period)


class TestTrendAdapterHMA:
    """TrendAdapterのHMAメソッドのテスト"""

    def setup_method(self):
        """テスト初期化"""
        self.test_series = pd.Series(
            np.random.uniform(100, 110, 100),
            index=pd.date_range('2023-01-01', periods=100, freq='D'),
            name='close'
        )

    def test_trend_adapter_hma_method_exists(self):
        """TrendAdapter.hmaメソッドが存在することをテスト"""
        # Red: まだ実装されていないのでAttributeErrorが発生する予定
        assert hasattr(TrendAdapter, 'hma'), "TrendAdapter.hmaメソッドが実装されていません"

    def test_trend_adapter_hma_calculation(self):
        """TrendAdapter.hmaの計算テスト"""
        if not hasattr(TrendAdapter, 'hma'):
            pytest.skip("TrendAdapter.hmaが実装されていません")
            
        period = 21
        result = TrendAdapter.hma(self.test_series, period)
        
        # 結果の検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.test_series)
        assert result.name == f"HMA_{period}"

    def test_trend_adapter_hma_different_periods(self):
        """TrendAdapter.hmaの異なる期間でのテスト"""
        if not hasattr(TrendAdapter, 'hma'):
            pytest.skip("TrendAdapter.hmaが実装されていません")
            
        for period in [9, 14, 21, 30]:
            result = TrendAdapter.hma(self.test_series, period)
            
            assert isinstance(result, pd.Series)
            assert result.name == f"HMA_{period}"

    def test_trend_adapter_hma_parameter_validation(self):
        """TrendAdapter.hmaのパラメータ検証テスト"""
        if not hasattr(TrendAdapter, 'hma'):
            pytest.skip("TrendAdapter.hmaが実装されていません")
            
        # 無効なパラメータでのテスト
        with pytest.raises(Exception):
            TrendAdapter.hma(self.test_series, 0)
            
        with pytest.raises(Exception):
            TrendAdapter.hma(self.test_series, -1)


class TestHMAIntegration:
    """HMAの統合テスト"""

    def test_hma_in_trend_indicators_factory(self):
        """get_trend_indicator関数でHMAが取得できることをテスト"""
        try:
            from app.core.services.indicators.trend_indicators import get_trend_indicator
            
            # Red: まだHMAが追加されていないのでValueErrorが発生する予定
            indicator = get_trend_indicator("HMA")
            assert indicator.indicator_type == "HMA"
            
        except (ImportError, ValueError):
            pytest.fail("HMAがget_trend_indicator関数に追加されていません")

    def test_hma_in_indicators_info(self):
        """TREND_INDICATORS_INFOにHMAが含まれることをテスト"""
        try:
            from app.core.services.indicators.trend_indicators import TREND_INDICATORS_INFO
            
            # Red: まだHMAが追加されていないのでKeyErrorが発生する予定
            assert "HMA" in TREND_INDICATORS_INFO
            
            hma_info = TREND_INDICATORS_INFO["HMA"]
            assert "periods" in hma_info
            assert "description" in hma_info
            assert "category" in hma_info
            assert hma_info["category"] == "trend"
            
        except (ImportError, KeyError):
            pytest.fail("HMAがTREND_INDICATORS_INFOに追加されていません")

    def test_hma_in_main_indicators_module(self):
        """メインのindicatorsモジュールでHMAが利用できることをテスト"""
        try:
            from app.core.services.indicators import HMAIndicator, get_indicator_by_type
            
            # HMAIndicatorの直接インポート
            assert HMAIndicator is not None
            
            # ファクトリー関数経由での取得
            indicator = get_indicator_by_type("HMA")
            assert indicator.indicator_type == "HMA"
            
        except (ImportError, ValueError):
            pytest.fail("HMAがメインのindicatorsモジュールに統合されていません")


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
