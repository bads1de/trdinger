"""
MAMA (MESA Adaptive Moving Average) 指標のテスト

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


class TestMAMAIndicator:
    """MAMA指標のテストクラス"""

    def test_mama_indicator_import(self):
        """MAMAIndicatorクラスのインポートテスト"""
        try:
            from app.core.services.indicators.trend_indicators import MAMAIndicator

            assert MAMAIndicator is not None
        except ImportError:
            pytest.fail("MAMAIndicatorクラスがインポートできません")

    def test_mama_indicator_initialization(self):
        """MAMAIndicatorの初期化テスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator

        mama = MAMAIndicator()
        assert mama.indicator_type == "MAMA"
        assert hasattr(mama, "supported_periods")
        assert isinstance(mama.supported_periods, list)
        assert len(mama.supported_periods) > 0

    def test_mama_calculation_basic(self):
        """MAMAの基本計算テスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator

        mama = MAMAIndicator()
        test_data = create_test_data(100)

        result = mama.calculate(test_data, period=30)

        # 結果の基本検証
        assert isinstance(result, dict)
        assert "mama" in result
        assert "fama" in result
        assert isinstance(result["mama"], pd.Series)
        assert isinstance(result["fama"], pd.Series)
        assert len(result["mama"]) == len(test_data)
        assert len(result["fama"]) == len(test_data)

    def test_mama_calculation_different_periods(self):
        """異なる期間でのMAMA計算テスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator

        mama = MAMAIndicator()
        test_data = create_test_data(100)

        for period in [20, 30, 50]:
            result = mama.calculate(test_data, period=period)
            assert isinstance(result, dict)
            assert "mama" in result
            assert "fama" in result

    def test_mama_calculation_insufficient_data(self):
        """データ不足時のエラーハンドリングテスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator
        from app.core.services.indicators.adapters import TALibCalculationError

        mama = MAMAIndicator()
        test_data = create_test_data(10)  # 少ないデータ

        with pytest.raises(TALibCalculationError):
            mama.calculate(test_data, period=50)  # 期間がデータより長い

    def test_mama_calculation_invalid_period(self):
        """無効なパラメータでのエラーハンドリングテスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator
        from app.core.services.indicators.adapters import TALibCalculationError

        mama = MAMAIndicator()
        test_data = create_test_data(100)

        # 無効なfastlimitでテスト
        with pytest.raises(TALibCalculationError):
            mama.calculate(test_data, period=30, fastlimit=-1)

        # 無効なslowlimitでテスト
        with pytest.raises(TALibCalculationError):
            mama.calculate(test_data, period=30, slowlimit=-1)

    def test_mama_description(self):
        """MAMAの説明テスト"""
        from app.core.services.indicators.trend_indicators import MAMAIndicator

        mama = MAMAIndicator()
        description = mama.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "MAMA" in description or "MESA" in description

    def test_mama_adapter_function(self):
        """MAMAアダプター関数のテスト"""
        try:
            from app.core.services.indicators.adapters.trend_adapter import TrendAdapter

            test_data = create_test_data(100)
            result = TrendAdapter.mama(test_data["close"], 0.5, 0.05)

            assert isinstance(result, dict)
            assert "mama" in result
            assert "fama" in result

        except ImportError:
            pytest.fail("TrendAdapterのmamaメソッドが実装されていません")

    def test_mama_factory_function(self):
        """MAMAファクトリー関数のテスト"""
        try:
            from app.core.services.indicators.trend_indicators import (
                get_trend_indicator,
            )

            mama = get_trend_indicator("MAMA")
            assert mama is not None
            assert mama.indicator_type == "MAMA"

        except (ImportError, ValueError):
            pytest.fail("MAMAがファクトリー関数に登録されていません")

    def test_mama_info_dictionary(self):
        """MAMA情報辞書のテスト"""
        try:
            from app.core.services.indicators.trend_indicators import (
                TREND_INDICATORS_INFO,
            )

            assert "MAMA" in TREND_INDICATORS_INFO
            mama_info = TREND_INDICATORS_INFO["MAMA"]

            assert "periods" in mama_info
            assert "description" in mama_info
            assert "category" in mama_info
            assert mama_info["category"] == "trend"

        except (ImportError, KeyError):
            pytest.fail("MAMAが情報辞書に登録されていません")


def test_mama_integration():
    """MAMA統合テスト"""
    print("\n🧪 MAMA (MESA Adaptive Moving Average) 統合テスト")
    print("=" * 60)

    try:
        from app.core.services.indicators.trend_indicators import MAMAIndicator

        # テストデータ作成
        test_data = create_test_data(100)
        print(f"📊 テストデータ作成: {len(test_data)}件")

        # MAMA計算
        mama = MAMAIndicator()
        result = mama.calculate(test_data, period=30)

        print(f"✅ MAMA計算成功")
        print(f"   期間: 30")
        print(f"   データ数: {len(result['mama'])}")
        print(f"   MAMA有効値数: {len(result['mama'].dropna())}")
        print(f"   FAMA有効値数: {len(result['fama'].dropna())}")
        print(f"   MAMA最後の値: {result['mama'].iloc[-1]:.2f}")
        print(f"   FAMA最後の値: {result['fama'].iloc[-1]:.2f}")

        return True

    except Exception as e:
        print(f"❌ MAMA統合テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 統合テスト実行
    success = test_mama_integration()

    if success:
        print("\n🎉 MAMA指標のテストが成功しました！")
    else:
        print("\n⚠️ MAMA指標のテストが失敗しました。実装が必要です。")
