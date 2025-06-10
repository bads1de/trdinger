#!/usr/bin/env python3
"""
統合テストとエラーハンドリングテスト

システム全体の統合テストと、様々なエラーケースのテストを行います。
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_ta_lib_integration():
    """TA-Lib統合テスト"""
    print("🧪 TA-Lib統合テスト")
    print("=" * 50)

    try:
        import talib

        print(f"✅ TA-Lib バージョン: {talib.__version__}")
        print(f"📊 利用可能な関数数: {len(talib.get_functions())}")

        # 基本的なTA-Lib関数のテスト
        test_data = np.random.random(100) * 100 + 50

        # SMAテスト
        sma_result = talib.SMA(test_data, timeperiod=20)
        print(f"✅ TA-Lib SMA計算成功: 最後の値 {sma_result[-1]:.2f}")

        # EMAテスト
        ema_result = talib.EMA(test_data, timeperiod=20)
        print(f"✅ TA-Lib EMA計算成功: 最後の値 {ema_result[-1]:.2f}")

        # RSIテスト
        rsi_result = talib.RSI(test_data, timeperiod=14)
        print(f"✅ TA-Lib RSI計算成功: 最後の値 {rsi_result[-1]:.2f}")

        print("✅ TA-Lib統合テスト成功")
        return True

    except Exception as e:
        print(f"❌ TA-Lib統合テストエラー: {e}")
        traceback.print_exc()
        return False


def test_adapter_integration():
    """アダプター統合テスト"""
    print("\n🧪 アダプター統合テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.adapters import (
            TrendAdapter,
            MomentumAdapter,
            VolatilityAdapter,
            VolumeAdapter,
        )

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        volumes = np.random.randint(1000, 10000, 100)

        close_series = pd.Series(close_prices, index=dates)
        high_series = pd.Series(high_prices, index=dates)
        low_series = pd.Series(low_prices, index=dates)
        volume_series = pd.Series(volumes, index=dates)

        # 各アダプターのテスト
        print("\n1. TrendAdapterテスト")
        sma_result = TrendAdapter.sma(close_series, period=20)
        ema_result = TrendAdapter.ema(close_series, period=20)
        print(f"   ✅ SMA: {sma_result.iloc[-1]:.2f}, EMA: {ema_result.iloc[-1]:.2f}")

        print("\n2. MomentumAdapterテスト")
        rsi_result = MomentumAdapter.rsi(close_series, period=14)
        mom_result = MomentumAdapter.momentum(close_series, period=10)
        print(f"   ✅ RSI: {rsi_result.iloc[-1]:.2f}, MOM: {mom_result.iloc[-1]:.2f}")

        print("\n3. VolatilityAdapterテスト")
        atr_result = VolatilityAdapter.atr(
            high_series, low_series, close_series, period=14
        )
        print(f"   ✅ ATR: {atr_result.iloc[-1]:.2f}")

        print("\n4. VolumeAdapterテスト")
        ad_result = VolumeAdapter.ad(
            high_series, low_series, close_series, volume_series
        )
        print(f"   ✅ A/D Line: {ad_result.iloc[-1]:.2f}")

        print("✅ アダプター統合テスト成功")
        return True

    except Exception as e:
        print(f"❌ アダプター統合テストエラー: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n🧪 エラーハンドリングテスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.adapters import (
            TrendAdapter,
            MomentumAdapter,
            TALibCalculationError,
        )

        # 1. 空のデータテスト
        print("\n1. 空のデータテスト")
        empty_series = pd.Series([], dtype=float)

        try:
            TrendAdapter.sma(empty_series, period=20)
            print("   ❌ 空データでエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("   ✅ 空データで適切にエラー発生")

        # 2. 不正な期間テスト
        print("\n2. 不正な期間テスト")
        valid_series = pd.Series([1, 2, 3, 4, 5])

        try:
            TrendAdapter.sma(valid_series, period=0)
            print("   ❌ 不正期間(0)でエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("   ✅ 不正期間(0)で適切にエラー発生")

        try:
            TrendAdapter.sma(valid_series, period=-1)
            print("   ❌ 不正期間(-1)でエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("   ✅ 不正期間(-1)で適切にエラー発生")

        # 3. データ長不足テスト
        print("\n3. データ長不足テスト")
        short_series = pd.Series([1, 2, 3, 4, 5])

        try:
            TrendAdapter.sma(short_series, period=10)
            print("   ❌ データ長不足でエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("   ✅ データ長不足で適切にエラー発生")

        # 4. NaN値を含むデータテスト
        print("\n4. NaN値を含むデータテスト")
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data_with_nan = pd.Series(range(50), index=dates, dtype=float)
        data_with_nan.iloc[20:25] = np.nan

        # NaN値があってもTA-Libは処理できることを確認
        result = TrendAdapter.sma(data_with_nan, period=10)
        assert isinstance(result, pd.Series)
        print("   ✅ NaN値を含むデータで正常処理")

        # 5. 異なるデータ型テスト
        print("\n5. 異なるデータ型テスト")

        # リストからの変換
        list_data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        result = TrendAdapter.sma(pd.Series(list_data), period=5)
        assert isinstance(result, pd.Series)
        print("   ✅ リストデータで正常処理")

        # numpy配列からの変換
        array_data = np.array(list_data)
        result = TrendAdapter.sma(pd.Series(array_data), period=5)
        assert isinstance(result, pd.Series)
        print("   ✅ numpy配列データで正常処理")

        print("✅ エラーハンドリングテスト成功")
        return True

    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """パフォーマンステスト"""
    print("\n🧪 パフォーマンステスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.adapters import TrendAdapter
        import time

        # 大きなテストデータ作成
        dates = pd.date_range("2020-01-01", periods=10000, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 10000)
        prices = base_price * np.exp(np.cumsum(returns))
        test_data = pd.Series(prices, index=dates, name="close")

        print(f"📊 大規模テストデータ: {len(test_data)}件")

        # TA-Libでの計算時間
        start_time = time.time()
        talib_result = TrendAdapter.sma(test_data, period=20)
        talib_time = time.time() - start_time

        # pandasでの計算時間
        start_time = time.time()
        pandas_result = test_data.rolling(window=20).mean()
        pandas_time = time.time() - start_time

        print(f"⏱️ TA-Lib時間: {talib_time:.6f}秒")
        print(f"⏱️ pandas時間: {pandas_time:.6f}秒")

        if talib_time > 0:
            speed_ratio = pandas_time / talib_time
            print(f"🚀 速度比: {speed_ratio:.2f}倍高速")
        else:
            print("🚀 TA-Lib: 測定不可能なほど高速")

        # 結果の精度比較
        # NaN値を除外して比較
        talib_clean = talib_result.dropna()
        pandas_clean = pandas_result.dropna()

        # インデックスを合わせる
        common_index = talib_clean.index.intersection(pandas_clean.index)

        if len(common_index) > 0:
            diff = (
                (talib_clean.loc[common_index] - pandas_clean.loc[common_index])
                .abs()
                .max()
            )
            print(f"📊 最大差分: {diff:.10f}")

            if diff < 1e-10:
                print("✅ 計算精度: 完全一致")
            elif diff < 1e-6:
                print("✅ 計算精度: 高精度")
            else:
                print("⚠️ 計算精度: 差分あり")

        print("✅ パフォーマンステスト成功")
        return True

    except Exception as e:
        print(f"❌ パフォーマンステストエラー: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔬 統合テストとエラーハンドリングテスト")
    print("=" * 60)

    # テスト実行
    results = {
        "ta_lib_integration": test_ta_lib_integration(),
        "adapter_integration": test_adapter_integration(),
        "error_handling": test_error_handling(),
        "performance": test_performance(),
    }

    # 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name.replace('_', ' ').title()}テスト: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\n📊 総合結果: {passed_tests}/{total_tests} 成功")

    if passed_tests == total_tests:
        print("🎉 全ての統合テストが成功しました！")
        print("システムは正常に動作しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。修正が必要です。")
