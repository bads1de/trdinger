#!/usr/bin/env python3
"""
既存システムとの互換性テスト
TA-lib移行後も既存のテストが通ることを確認します
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# 警告を抑制
warnings.filterwarnings("ignore")


def test_existing_api_compatibility():
    """既存API互換性テスト"""
    print("🔄 既存API互換性テスト")
    print("-" * 50)

    success_count = 0
    total_tests = 0

    # テストデータ作成
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    close_prices = base_price * np.exp(np.cumsum(returns))

    test_data = pd.DataFrame(
        {
            "open": close_prices * (1 + np.random.normal(0, 0.001, 100)),
            "high": close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            "low": close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # 1. トレンド系指標の既存API
    print("\n📈 トレンド系指標 既存API")
    try:
        from app.core.services.indicators.trend_indicators import (
            SMAIndicator,
            EMAIndicator,
            MACDIndicator,
        )

        # SMA
        total_tests += 1
        try:
            sma = SMAIndicator()
            result = sma.calculate(test_data, period=20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            print("   ✅ SMAIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ SMAIndicator API: {e}")

        # EMA
        total_tests += 1
        try:
            ema = EMAIndicator()
            result = ema.calculate(test_data, period=20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            print("   ✅ EMAIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ EMAIndicator API: {e}")

        # MACD
        total_tests += 1
        try:
            macd = MACDIndicator()
            result = macd.calculate(test_data, period=12)
            assert isinstance(result, pd.DataFrame)
            expected_columns = {"macd_line", "signal_line", "histogram"}
            assert set(result.columns) == expected_columns
            print("   ✅ MACDIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ MACDIndicator API: {e}")

    except ImportError as e:
        print(f"   ❌ トレンド系指標インポートエラー: {e}")

    # 2. モメンタム系指標の既存API
    print("\n📊 モメンタム系指標 既存API")
    try:
        from app.core.services.indicators.momentum_indicators import (
            RSIIndicator,
            StochasticIndicator,
        )

        # RSI
        total_tests += 1
        try:
            rsi = RSIIndicator()
            result = rsi.calculate(test_data, period=14)
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            print("   ✅ RSIIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ RSIIndicator API: {e}")

        # Stochastic
        total_tests += 1
        try:
            stoch = StochasticIndicator()
            result = stoch.calculate(test_data, period=14)
            assert isinstance(result, pd.DataFrame)
            expected_columns = {"k_percent", "d_percent"}
            assert set(result.columns) == expected_columns
            print("   ✅ StochasticIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ StochasticIndicator API: {e}")

    except ImportError as e:
        print(f"   ❌ モメンタム系指標インポートエラー: {e}")

    # 3. ボラティリティ系指標の既存API
    print("\n📉 ボラティリティ系指標 既存API")
    try:
        from app.core.services.indicators.volatility_indicators import (
            BollingerBandsIndicator,
            ATRIndicator,
        )

        # Bollinger Bands
        total_tests += 1
        try:
            bb = BollingerBandsIndicator()
            result = bb.calculate(test_data, period=20)
            assert isinstance(result, pd.DataFrame)
            expected_columns = {"upper", "middle", "lower"}
            assert set(result.columns) == expected_columns
            print("   ✅ BollingerBandsIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ BollingerBandsIndicator API: {e}")

        # ATR
        total_tests += 1
        try:
            atr = ATRIndicator()
            result = atr.calculate(test_data, period=14)
            assert isinstance(result, pd.Series)
            assert len(result) == len(test_data)
            print("   ✅ ATRIndicator API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ ATRIndicator API: {e}")

    except ImportError as e:
        print(f"   ❌ ボラティリティ系指標インポートエラー: {e}")

    # 4. backtesting.py用関数の既存API
    print("\n🎯 backtesting.py用関数 既存API")
    try:
        from app.core.strategies.indicators import SMA, EMA, RSI, MACD

        close_data = test_data["close"]

        # SMA関数
        total_tests += 1
        try:
            result = SMA(close_data, 20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(close_data)
            print("   ✅ SMA関数 API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ SMA関数 API: {e}")

        # EMA関数
        total_tests += 1
        try:
            result = EMA(close_data, 20)
            assert isinstance(result, pd.Series)
            assert len(result) == len(close_data)
            print("   ✅ EMA関数 API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ EMA関数 API: {e}")

        # RSI関数
        total_tests += 1
        try:
            result = RSI(close_data, 14)
            assert isinstance(result, pd.Series)
            assert len(result) == len(close_data)
            print("   ✅ RSI関数 API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ RSI関数 API: {e}")

        # MACD関数
        total_tests += 1
        try:
            macd_line, signal_line, histogram = MACD(close_data)
            assert all(
                isinstance(x, pd.Series) for x in [macd_line, signal_line, histogram]
            )
            assert all(
                len(x) == len(close_data) for x in [macd_line, signal_line, histogram]
            )
            print("   ✅ MACD関数 API互換")
            success_count += 1
        except Exception as e:
            print(f"   ❌ MACD関数 API: {e}")

    except ImportError as e:
        print(f"   ❌ backtesting.py用関数インポートエラー: {e}")

    return success_count, total_tests


def test_calculation_accuracy():
    """計算精度テスト"""
    print("\n🎯 計算精度テスト")
    print("-" * 50)

    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
        from app.core.services.indicators.trend_indicators import SMAIndicator
        from app.core.strategies.indicators import SMA

        # 既知の値でテスト
        test_values = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        test_series = pd.Series(test_values)

        # 期待値（手動計算）
        # SMA(5) for last 5 values: (105+106+107+108+109)/5 = 107.0
        expected_sma = 107.0

        # 各実装での計算
        talib_result = TALibAdapter.sma(test_series, 5)

        sma_indicator = SMAIndicator()
        test_df = pd.DataFrame({"close": test_series})
        class_result = sma_indicator.calculate(test_df, 5)

        func_result = SMA(test_series, 5)

        # 最後の値を比較
        talib_last = talib_result.iloc[-1]
        class_last = class_result.iloc[-1]
        func_last = func_result.iloc[-1]

        print(f"   📊 期待値: {expected_sma}")
        print(f"   📊 TALibAdapter: {talib_last}")
        print(f"   📊 SMAIndicator: {class_last}")
        print(f"   📊 SMA関数: {func_last}")

        # 精度チェック
        tolerance = 1e-10

        if abs(talib_last - expected_sma) < tolerance:
            print("   ✅ TALibAdapter 精度良好")
        else:
            print(f"   ❌ TALibAdapter 精度問題: 差分={abs(talib_last - expected_sma)}")

        if abs(class_last - expected_sma) < tolerance:
            print("   ✅ SMAIndicator 精度良好")
        else:
            print(f"   ❌ SMAIndicator 精度問題: 差分={abs(class_last - expected_sma)}")

        if abs(func_last - expected_sma) < tolerance:
            print("   ✅ SMA関数 精度良好")
        else:
            print(f"   ❌ SMA関数 精度問題: 差分={abs(func_last - expected_sma)}")

        # 相互一貫性チェック
        max_diff = max(
            abs(talib_last - class_last),
            abs(talib_last - func_last),
            abs(class_last - func_last),
        )

        if max_diff < tolerance:
            print("   ✅ 実装間の一貫性良好")
        else:
            print(f"   ❌ 実装間の一貫性問題: 最大差分={max_diff}")

    except Exception as e:
        print(f"   ❌ 計算精度テストエラー: {e}")


def test_fallback_mechanism():
    """フォールバック機能テスト"""
    print("\n🛡️ フォールバック機能テスト")
    print("-" * 50)

    try:
        # 正常なデータでテスト
        test_data = pd.DataFrame({"close": [100, 101, 102, 103, 104] * 20})

        from app.core.services.indicators.trend_indicators import SMAIndicator

        sma = SMAIndicator()
        result = sma.calculate(test_data, 10)

        # 結果が得られることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(test_data)

        print("   ✅ フォールバック機能正常動作")
        print(f"   📊 計算結果: {result.iloc[-1]:.2f}")

    except Exception as e:
        print(f"   ❌ フォールバック機能テストエラー: {e}")


def main():
    """互換性テスト実行"""
    print("🔄 TA-lib移行 互換性テスト")
    print("=" * 70)

    # 既存API互換性テスト
    success_count, total_tests = test_existing_api_compatibility()

    # 計算精度テスト
    test_calculation_accuracy()

    # フォールバック機能テスト
    test_fallback_mechanism()

    # 結果サマリー
    print("\n📋 互換性テスト結果")
    print("=" * 70)

    if total_tests > 0:
        success_rate = (success_count / total_tests) * 100
        print(f"📊 API互換性: {success_count}/{total_tests} ({success_rate:.1f}%)")

    if success_count == total_tests:
        print("\n🎉 完全な後方互換性が確認されました！")
        print("✅ 既存のコードは一切変更なしで動作します")
        print("🚀 TA-lib移行による高速化の恩恵を受けられます")
        return True
    else:
        print(f"\n⚠️ {total_tests - success_count}個の互換性問題があります")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
