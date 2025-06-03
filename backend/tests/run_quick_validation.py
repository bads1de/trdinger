#!/usr/bin/env python3
"""
クイック検証スクリプト

テクニカル指標サービスの基本機能を迅速に検証します。
"""

import sys
import os
import traceback
from datetime import datetime

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(backend_dir)

sys.path.insert(0, backend_dir)
sys.path.insert(0, project_dir)


def test_ta_lib_basic():
    """TA-Lib基本動作テスト"""
    print("🧪 TA-Lib基本動作テスト")
    print("-" * 40)

    try:
        import talib
        import numpy as np

        print(f"✅ TA-Lib バージョン: {talib.__version__}")
        print(f"📊 利用可能関数数: {len(talib.get_functions())}")

        # 基本計算テスト
        test_data = np.random.random(100) * 100 + 50

        sma = talib.SMA(test_data, timeperiod=20)
        ema = talib.EMA(test_data, timeperiod=20)
        rsi = talib.RSI(test_data, timeperiod=14)

        print(f"✅ SMA計算: {sma[-1]:.2f}")
        print(f"✅ EMA計算: {ema[-1]:.2f}")
        print(f"✅ RSI計算: {rsi[-1]:.2f}")

        return True
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_adapters_import():
    """アダプタークラスインポートテスト"""
    print("\n🧪 アダプタークラスインポートテスト")
    print("-" * 40)

    try:
        from app.core.services.indicators.adapters import (
            BaseAdapter,
            TALibCalculationError,
            TrendAdapter,
            MomentumAdapter,
            VolatilityAdapter,
            VolumeAdapter,
        )

        print("✅ BaseAdapter インポート成功")
        print("✅ TALibCalculationError インポート成功")
        print("✅ TrendAdapter インポート成功")
        print("✅ MomentumAdapter インポート成功")
        print("✅ VolatilityAdapter インポート成功")
        print("✅ VolumeAdapter インポート成功")

        return True
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        traceback.print_exc()
        return False


def test_trend_adapter():
    """TrendAdapterテスト"""
    print("\n🧪 TrendAdapterテスト")
    print("-" * 40)

    try:
        import pandas as pd
        import numpy as np
        from app.core.services.indicators.adapters import TrendAdapter

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        prices = pd.Series(np.random.random(50) * 100 + 50, index=dates)

        # SMAテスト
        sma_result = TrendAdapter.sma(prices, period=20)
        print(f"✅ SMA計算成功: {sma_result.iloc[-1]:.2f}")
        print(f"   データ長: {len(sma_result)}, 名前: {sma_result.name}")

        # EMAテスト
        ema_result = TrendAdapter.ema(prices, period=20)
        print(f"✅ EMA計算成功: {ema_result.iloc[-1]:.2f}")
        print(f"   データ長: {len(ema_result)}, 名前: {ema_result.name}")

        return True
    except Exception as e:
        print(f"❌ TrendAdapterエラー: {e}")
        traceback.print_exc()
        return False


def test_momentum_adapter():
    """MomentumAdapterテスト"""
    print("\n🧪 MomentumAdapterテスト")
    print("-" * 40)

    try:
        import pandas as pd
        import numpy as np
        from app.core.services.indicators.adapters import MomentumAdapter

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)
        prices = pd.Series(np.random.random(50) * 100 + 50, index=dates)

        # RSIテスト
        rsi_result = MomentumAdapter.rsi(prices, period=14)
        print(f"✅ RSI計算成功: {rsi_result.iloc[-1]:.2f}")
        print(f"   データ長: {len(rsi_result)}, 名前: {rsi_result.name}")

        # RSI範囲チェック
        valid_values = rsi_result.dropna()
        if len(valid_values) > 0:
            min_val = valid_values.min()
            max_val = valid_values.max()
            print(f"   RSI範囲: {min_val:.2f} - {max_val:.2f}")
            if 0 <= min_val and max_val <= 100:
                print("   ✅ RSI範囲正常 (0-100)")
            else:
                print("   ⚠️ RSI範囲異常")

        return True
    except Exception as e:
        print(f"❌ MomentumAdapterエラー: {e}")
        traceback.print_exc()
        return False


def test_indicator_orchestrator():
    """TechnicalIndicatorServiceテスト"""
    print("\n🧪 TechnicalIndicatorServiceテスト")
    print("-" * 40)

    try:
        from app.core.services.indicators.indicator_orchestrator import (
            TechnicalIndicatorService,
        )

        # インスタンス作成
        orchestrator = TechnicalIndicatorService()
        print("✅ TechnicalIndicatorService インスタンス作成成功")

        # サポート指標確認
        supported = orchestrator.get_supported_indicators()
        print(f"✅ サポート指標数: {len(supported)}")

        # 主要指標の確認
        key_indicators = ["SMA", "EMA", "RSI", "MACD"]
        for indicator in key_indicators:
            if indicator in supported:
                periods = supported[indicator].get("periods", [])
                print(f"   {indicator}: {periods}")
            else:
                print(f"   ❌ {indicator}: サポートなし")

        return True
    except Exception as e:
        print(f"❌ TechnicalIndicatorServiceエラー: {e}")
        traceback.print_exc()
        return False


def test_individual_indicators():
    """個別指標クラステスト"""
    print("\n🧪 個別指標クラステスト")
    print("-" * 40)

    try:
        import pandas as pd
        import numpy as np
        from app.core.services.indicators.trend_indicators import (
            SMAIndicator,
            EMAIndicator,
        )
        from app.core.services.indicators.momentum_indicators import RSIIndicator

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 50)
        close_prices = base_price * np.exp(np.cumsum(returns))

        test_data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * 1.01,
                "low": close_prices * 0.99,
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 50),
            },
            index=dates,
        )

        # SMA指標テスト
        sma_indicator = SMAIndicator()
        sma_result = sma_indicator.calculate(test_data, period=20)
        print(f"✅ SMA指標: {sma_result.iloc[-1]:.2f}")

        # EMA指標テスト
        ema_indicator = EMAIndicator()
        ema_result = ema_indicator.calculate(test_data, period=20)
        print(f"✅ EMA指標: {ema_result.iloc[-1]:.2f}")

        # RSI指標テスト
        rsi_indicator = RSIIndicator()
        rsi_result = rsi_indicator.calculate(test_data, period=14)
        print(f"✅ RSI指標: {rsi_result.iloc[-1]:.2f}")

        return True
    except Exception as e:
        print(f"❌ 個別指標エラー: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n🧪 エラーハンドリングテスト")
    print("-" * 40)

    try:
        import pandas as pd
        from app.core.services.indicators.adapters import (
            TrendAdapter,
            TALibCalculationError,
        )

        # 空データテスト
        empty_series = pd.Series([], dtype=float)
        try:
            TrendAdapter.sma(empty_series, period=20)
            print("❌ 空データでエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("✅ 空データで適切にエラー発生")

        # 不正期間テスト
        valid_series = pd.Series([1, 2, 3, 4, 5])
        try:
            TrendAdapter.sma(valid_series, period=0)
            print("❌ 不正期間でエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("✅ 不正期間で適切にエラー発生")

        return True
    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("🔬 テクニカル指標サービス クイック検証")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # テスト実行
    tests = [
        ("TA-Lib基本動作", test_ta_lib_basic),
        ("アダプタークラスインポート", test_adapters_import),
        ("TrendAdapter", test_trend_adapter),
        ("MomentumAdapter", test_momentum_adapter),
        ("TechnicalIndicatorService", test_indicator_orchestrator),
        ("個別指標クラス", test_individual_indicators),
        ("エラーハンドリング", test_error_handling),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}で予期しないエラー: {e}")
            results[test_name] = False

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name}: {status}")

    print(f"\n📊 総合結果: {passed_tests}/{total_tests} 成功")
    print(f"📈 成功率: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 全てのテストが成功しました！")
        print("テクニカル指標サービスは正常に動作しています。")
    else:
        print(f"\n⚠️ {total_tests - passed_tests}個のテストが失敗しました。")
        print("詳細なエラー情報を確認して修正してください。")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
