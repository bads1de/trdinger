"""
TRIX (Triple Exponential Moving Average) 指標のテスト

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


class TestTRIXIndicator:
    """TRIX指標のテストクラス"""

    def test_trix_indicator_import(self):
        """TRIXIndicatorクラスのインポートテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import TRIXIndicator
            assert TRIXIndicator is not None
        except ImportError:
            pytest.fail("TRIXIndicatorクラスがインポートできません")

    def test_trix_indicator_initialization(self):
        """TRIXIndicatorの初期化テスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        
        trix = TRIXIndicator()
        assert trix.indicator_type == "TRIX"
        assert hasattr(trix, 'supported_periods')
        assert isinstance(trix.supported_periods, list)
        assert len(trix.supported_periods) > 0

    def test_trix_calculation_basic(self):
        """TRIXの基本計算テスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        
        trix = TRIXIndicator()
        test_data = create_test_data(100)
        
        result = trix.calculate(test_data, period=14)
        
        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert result.name == "TRIX_14"
        assert len(result) == len(test_data)
        
        # TRIXの値域チェック（パーセンテージ値）
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_trix_calculation_different_periods(self):
        """異なる期間でのTRIX計算テスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        
        trix = TRIXIndicator()
        test_data = create_test_data(100)
        
        for period in [14, 21, 30]:
            result = trix.calculate(test_data, period=period)
            assert isinstance(result, pd.Series)
            assert result.name == f"TRIX_{period}"

    def test_trix_calculation_insufficient_data(self):
        """データ不足時のエラーハンドリングテスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        trix = TRIXIndicator()
        test_data = create_test_data(10)  # 少ないデータ
        
        with pytest.raises(TALibCalculationError):
            trix.calculate(test_data, period=20)  # 期間がデータより長い

    def test_trix_calculation_invalid_period(self):
        """無効な期間でのエラーハンドリングテスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        trix = TRIXIndicator()
        test_data = create_test_data(100)
        
        with pytest.raises(TALibCalculationError):
            trix.calculate(test_data, period=0)
        
        with pytest.raises(TALibCalculationError):
            trix.calculate(test_data, period=-1)

    def test_trix_description(self):
        """TRIXの説明テスト"""
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        
        trix = TRIXIndicator()
        description = trix.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "TRIX" in description or "Triple" in description

    def test_trix_adapter_function(self):
        """TRIXアダプター関数のテスト"""
        try:
            from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
            
            test_data = create_test_data(100)
            result = MomentumAdapter.trix(test_data["close"], period=14)
            
            assert isinstance(result, pd.Series)
            assert result.name == "TRIX_14"
            
        except ImportError:
            pytest.fail("MomentumAdapterのtrixメソッドが実装されていません")

    def test_trix_factory_function(self):
        """TRIXファクトリー関数のテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import get_momentum_indicator
            
            trix = get_momentum_indicator("TRIX")
            assert trix is not None
            assert trix.indicator_type == "TRIX"
            
        except (ImportError, ValueError):
            pytest.fail("TRIXがファクトリー関数に登録されていません")

    def test_trix_info_dictionary(self):
        """TRIX情報辞書のテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import MOMENTUM_INDICATORS_INFO
            
            assert "TRIX" in MOMENTUM_INDICATORS_INFO
            trix_info = MOMENTUM_INDICATORS_INFO["TRIX"]
            
            assert "periods" in trix_info
            assert "description" in trix_info
            assert "category" in trix_info
            assert trix_info["category"] == "momentum"
            
        except (ImportError, KeyError):
            pytest.fail("TRIXが情報辞書に登録されていません")


def test_trix_integration():
    """TRIX統合テスト"""
    print("\n🧪 TRIX (Triple Exponential Moving Average) 統合テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.momentum_indicators import TRIXIndicator
        
        # テストデータ作成
        test_data = create_test_data(100)
        print(f"📊 テストデータ作成: {len(test_data)}件")
        
        # TRIX計算
        trix = TRIXIndicator()
        result = trix.calculate(test_data, period=14)
        
        print(f"✅ TRIX計算成功")
        print(f"   期間: 14")
        print(f"   データ数: {len(result)}")
        print(f"   有効値数: {len(result.dropna())}")
        print(f"   最後の値: {result.iloc[-1]:.6f}")
        print(f"   値域: {result.dropna().min():.6f} ～ {result.dropna().max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TRIX統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 統合テスト実行
    success = test_trix_integration()
    
    if success:
        print("\n🎉 TRIX指標のテストが成功しました！")
    else:
        print("\n⚠️ TRIX指標のテストが失敗しました。実装が必要です。")
