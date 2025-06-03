"""
基本的なアダプター機能のテスト

TA-Libアダプターの基本機能が正常に動作するかを確認します。
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_basic_imports():
    """基本的なインポートテスト"""
    try:
        from app.core.services.indicators.adapters import (
            BaseAdapter,
            TALibCalculationError,
            TrendAdapter,
            MomentumAdapter,
            VolatilityAdapter,
            VolumeAdapter,
        )

        print("✅ 全てのアダプタークラスのインポートに成功")
        return True
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False


def test_basic_sma_calculation():
    """基本的なSMA計算テスト"""
    try:
        from app.core.services.indicators.adapters import TrendAdapter

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        prices = pd.Series(np.random.random(50) * 100 + 50, index=dates, name="close")

        # SMA計算
        result = TrendAdapter.sma(prices, period=20)

        # 基本的な検証
        assert isinstance(result, pd.Series), "結果がpandas.Seriesではありません"
        assert len(result) == len(prices), "結果の長さが元データと異なります"
        assert result.name == "SMA_20", "結果の名前が正しくありません"

        print("✅ SMA計算テストに成功")
        return True
    except Exception as e:
        print(f"❌ SMA計算テストでエラー: {e}")
        return False


def test_basic_rsi_calculation():
    """基本的なRSI計算テスト"""
    try:
        from app.core.services.indicators.adapters import MomentumAdapter

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        prices = pd.Series(np.random.random(50) * 100 + 50, index=dates, name="close")

        # RSI計算
        result = MomentumAdapter.rsi(prices, period=14)

        # 基本的な検証
        assert isinstance(result, pd.Series), "結果がpandas.Seriesではありません"
        assert len(result) == len(prices), "結果の長さが元データと異なります"
        assert result.name == "RSI_14", "結果の名前が正しくありません"

        # RSIの値域チェック
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all(), "RSI値が0未満です"
            assert (valid_values <= 100).all(), "RSI値が100を超えています"

        print("✅ RSI計算テストに成功")
        return True
    except Exception as e:
        print(f"❌ RSI計算テストでエラー: {e}")
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    try:
        from app.core.services.indicators.adapters import (
            TrendAdapter,
            TALibCalculationError,
        )

        # 空のSeriesでエラーが発生することを確認
        empty_series = pd.Series([], dtype=float)

        try:
            TrendAdapter.sma(empty_series, period=20)
            print("❌ 空のSeriesでエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("✅ 空のSeriesで適切にエラーが発生しました")

        # 期間が不正な場合
        valid_series = pd.Series([1, 2, 3, 4, 5])

        try:
            TrendAdapter.sma(valid_series, period=0)
            print("❌ 不正な期間でエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("✅ 不正な期間で適切にエラーが発生しました")

        return True
    except Exception as e:
        print(f"❌ エラーハンドリングテストでエラー: {e}")
        return False


if __name__ == "__main__":
    print("🔍 基本的なアダプター機能テスト開始")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_basic_sma_calculation,
        test_basic_rsi_calculation,
        test_error_handling,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 テスト結果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 全てのテストに成功しました！")
    else:
        print("⚠️ 一部のテストが失敗しました")
