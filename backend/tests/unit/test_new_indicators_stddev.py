"""
STDDEV (Standard Deviation) 指標のテスト

TDDアプローチで実装:
1. テスト作成（失敗）
2. 最小実装（成功）
3. リファクタリング
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def create_test_data(periods=100):
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    np.random.seed(42)

    base_price = 50000
    returns = np.random.normal(0, 0.02, periods)
    close_prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": close_prices * (1 + np.random.normal(0, 0.001, periods)),
            "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, periods),
        },
        index=dates,
    )


class TestSTDDEVIndicator:
    """STDDEV指標のテストクラス"""

    def test_stddev_indicator_import(self):
        """STDDEVIndicatorクラスのインポートテスト"""
        try:
            from app.core.services.indicators.volatility_indicators import STDDEVIndicator
            assert STDDEVIndicator is not None
        except ImportError:
            pytest.fail("STDDEVIndicatorクラスがインポートできません")

    def test_stddev_indicator_initialization(self):
        """STDDEVIndicatorの初期化テスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        
        stddev = STDDEVIndicator()
        assert stddev.indicator_type == "STDDEV"
        assert hasattr(stddev, 'supported_periods')
        assert isinstance(stddev.supported_periods, list)
        assert len(stddev.supported_periods) > 0

    def test_stddev_calculation_basic(self):
        """STDDEVの基本計算テスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        
        stddev = STDDEVIndicator()
        test_data = create_test_data(100)
        
        result = stddev.calculate(test_data, period=20)
        
        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert result.name == "STDDEV_20"
        assert len(result) == len(test_data)
        
        # STDDEVの値域チェック（正の値）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()

    def test_stddev_calculation_different_periods(self):
        """異なる期間でのSTDDEV計算テスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        
        stddev = STDDEVIndicator()
        test_data = create_test_data(100)
        
        for period in [10, 20, 30]:
            result = stddev.calculate(test_data, period=period)
            assert isinstance(result, pd.Series)
            assert result.name == f"STDDEV_{period}"

    def test_stddev_calculation_insufficient_data(self):
        """データ不足時のエラーハンドリングテスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        stddev = STDDEVIndicator()
        test_data = create_test_data(10)  # 少ないデータ
        
        with pytest.raises(TALibCalculationError):
            stddev.calculate(test_data, period=20)  # 期間がデータより長い

    def test_stddev_calculation_invalid_period(self):
        """無効な期間でのエラーハンドリングテスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        stddev = STDDEVIndicator()
        test_data = create_test_data(100)
        
        with pytest.raises(TALibCalculationError):
            stddev.calculate(test_data, period=0)
        
        with pytest.raises(TALibCalculationError):
            stddev.calculate(test_data, period=-1)

    def test_stddev_description(self):
        """STDDEVの説明テスト"""
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        
        stddev = STDDEVIndicator()
        description = stddev.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "標準偏差" in description or "STDDEV" in description

    def test_stddev_adapter_function(self):
        """STDDEVアダプター関数のテスト"""
        try:
            from app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter
            
            test_data = create_test_data(100)
            result = VolatilityAdapter.stddev(test_data["close"], period=20)
            
            assert isinstance(result, pd.Series)
            assert result.name == "STDDEV_20"
            
            # STDDEVの値域チェック（正の値）
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()
                
        except ImportError:
            pytest.fail("VolatilityAdapterのstddevメソッドが実装されていません")

    def test_stddev_factory_function(self):
        """STDDEVファクトリー関数のテスト"""
        try:
            from app.core.services.indicators.volatility_indicators import get_volatility_indicator
            
            stddev = get_volatility_indicator("STDDEV")
            assert stddev is not None
            assert stddev.indicator_type == "STDDEV"
            
        except (ImportError, ValueError):
            pytest.fail("STDDEVがファクトリー関数に登録されていません")

    def test_stddev_info_dictionary(self):
        """STDDEV情報辞書のテスト"""
        try:
            from app.core.services.indicators.volatility_indicators import VOLATILITY_INDICATORS_INFO
            
            assert "STDDEV" in VOLATILITY_INDICATORS_INFO
            stddev_info = VOLATILITY_INDICATORS_INFO["STDDEV"]
            
            assert "periods" in stddev_info
            assert "description" in stddev_info
            assert "category" in stddev_info
            assert stddev_info["category"] == "volatility"
            
        except (ImportError, KeyError):
            pytest.fail("STDDEVが情報辞書に登録されていません")


def test_stddev_integration():
    """STDDEV統合テスト"""
    print("\n🧪 STDDEV (Standard Deviation) 統合テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.volatility_indicators import STDDEVIndicator
        
        # テストデータ作成
        test_data = create_test_data(100)
        print(f"📊 テストデータ作成: {len(test_data)}件")
        
        # STDDEV計算
        stddev = STDDEVIndicator()
        result = stddev.calculate(test_data, period=20)
        
        print(f"✅ STDDEV計算成功")
        print(f"   期間: 20")
        print(f"   データ数: {len(result)}")
        print(f"   有効値数: {len(result.dropna())}")
        print(f"   最後の値: {result.iloc[-1]:.4f}")
        print(f"   値域: {result.dropna().min():.4f} ～ {result.dropna().max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ STDDEV統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 統合テスト実行
    success = test_stddev_integration()
    
    if success:
        print("\n🎉 STDDEV指標のテストが成功しました！")
    else:
        print("\n⚠️ STDDEV指標のテストが失敗しました。実装が必要です。")
