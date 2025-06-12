"""
CMO (Chande Momentum Oscillator) 指標のテスト

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


class TestCMOIndicator:
    """CMO指標のテストクラス"""

    def test_cmo_indicator_import(self):
        """CMOIndicatorクラスのインポートテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import CMOIndicator
            assert CMOIndicator is not None
        except ImportError:
            pytest.fail("CMOIndicatorクラスがインポートできません")

    def test_cmo_indicator_initialization(self):
        """CMOIndicatorの初期化テスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        
        cmo = CMOIndicator()
        assert cmo.indicator_type == "CMO"
        assert hasattr(cmo, 'supported_periods')
        assert isinstance(cmo.supported_periods, list)
        assert len(cmo.supported_periods) > 0

    def test_cmo_calculation_basic(self):
        """CMOの基本計算テスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        
        cmo = CMOIndicator()
        test_data = create_test_data(100)
        
        result = cmo.calculate(test_data, period=14)
        
        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert result.name == "CMO_14"
        assert len(result) == len(test_data)
        
        # CMOの値域チェック（-100から100の範囲）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= -100).all()
            assert (valid_values <= 100).all()

    def test_cmo_calculation_different_periods(self):
        """異なる期間でのCMO計算テスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        
        cmo = CMOIndicator()
        test_data = create_test_data(100)
        
        for period in [7, 14, 21, 28]:
            result = cmo.calculate(test_data, period=period)
            assert isinstance(result, pd.Series)
            assert result.name == f"CMO_{period}"

    def test_cmo_calculation_insufficient_data(self):
        """データ不足時のエラーハンドリングテスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        cmo = CMOIndicator()
        test_data = create_test_data(10)  # 少ないデータ
        
        with pytest.raises(TALibCalculationError):
            cmo.calculate(test_data, period=20)  # 期間がデータより長い

    def test_cmo_calculation_invalid_period(self):
        """無効な期間でのエラーハンドリングテスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        from app.core.services.indicators.adapters import TALibCalculationError
        
        cmo = CMOIndicator()
        test_data = create_test_data(100)
        
        with pytest.raises(TALibCalculationError):
            cmo.calculate(test_data, period=0)
        
        with pytest.raises(TALibCalculationError):
            cmo.calculate(test_data, period=-1)

    def test_cmo_description(self):
        """CMOの説明テスト"""
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        
        cmo = CMOIndicator()
        description = cmo.get_description()
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "CMO" in description or "Chande" in description

    def test_cmo_adapter_function(self):
        """CMOアダプター関数のテスト"""
        try:
            from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
            
            test_data = create_test_data(100)
            result = MomentumAdapter.cmo(test_data["close"], period=14)
            
            assert isinstance(result, pd.Series)
            assert result.name == "CMO_14"
            
            # CMOの値域チェック
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (valid_values >= -100).all()
                assert (valid_values <= 100).all()
                
        except ImportError:
            pytest.fail("MomentumAdapterのcmoメソッドが実装されていません")

    def test_cmo_factory_function(self):
        """CMOファクトリー関数のテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import get_momentum_indicator
            
            cmo = get_momentum_indicator("CMO")
            assert cmo is not None
            assert cmo.indicator_type == "CMO"
            
        except (ImportError, ValueError):
            pytest.fail("CMOがファクトリー関数に登録されていません")

    def test_cmo_info_dictionary(self):
        """CMO情報辞書のテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import MOMENTUM_INDICATORS_INFO
            
            assert "CMO" in MOMENTUM_INDICATORS_INFO
            cmo_info = MOMENTUM_INDICATORS_INFO["CMO"]
            
            assert "periods" in cmo_info
            assert "description" in cmo_info
            assert "category" in cmo_info
            assert cmo_info["category"] == "momentum"
            
        except (ImportError, KeyError):
            pytest.fail("CMOが情報辞書に登録されていません")


def test_cmo_integration():
    """CMO統合テスト"""
    print("\n🧪 CMO (Chande Momentum Oscillator) 統合テスト")
    print("=" * 60)
    
    try:
        from app.core.services.indicators.momentum_indicators import CMOIndicator
        
        # テストデータ作成
        test_data = create_test_data(100)
        print(f"📊 テストデータ作成: {len(test_data)}件")
        
        # CMO計算
        cmo = CMOIndicator()
        result = cmo.calculate(test_data, period=14)
        
        print(f"✅ CMO計算成功")
        print(f"   期間: 14")
        print(f"   データ数: {len(result)}")
        print(f"   有効値数: {len(result.dropna())}")
        print(f"   最後の値: {result.iloc[-1]:.4f}")
        print(f"   値域: {result.dropna().min():.4f} ～ {result.dropna().max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CMO統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 統合テスト実行
    success = test_cmo_integration()
    
    if success:
        print("\n🎉 CMO指標のテストが成功しました！")
    else:
        print("\n⚠️ CMO指標のテストが失敗しました。実装が必要です。")
