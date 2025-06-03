#!/usr/bin/env python3
"""
TALibAdapterの基本テスト
"""

import sys
import os
import pandas as pd
import numpy as np

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_talib_adapter():
    """TALibAdapterの基本テスト"""
    print("🧪 TALibAdapter テスト開始")
    print("=" * 50)

    try:
        from app.core.services.indicators.talib_adapter import (
            TALibAdapter,
            TALibCalculationError,
        )

        print("✅ TALibAdapter インポート成功")

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        test_data = pd.Series(prices, index=dates, name="close")
        print(f"📊 テストデータ作成: {len(test_data)}件")

        # SMAテスト
        print("\n1. SMA計算テスト")
        try:
            sma_result = TALibAdapter.sma(test_data, period=20)
            print(f"   ✅ SMA計算成功")
            print(f"   📈 結果の型: {type(sma_result)}")
            print(f"   📊 データ長: {len(sma_result)}")
            print(f"   🏷️ 名前: {sma_result.name}")
            print(f"   📉 最初の有効値: {sma_result.dropna().iloc[0]:.2f}")
            print(f"   📈 最後の値: {sma_result.iloc[-1]:.2f}")

            # 基本的な検証
            assert isinstance(sma_result, pd.Series)
            assert len(sma_result) == len(test_data)
            assert sma_result.index.equals(test_data.index)
            assert sma_result.name == "SMA_20"
            print("   ✅ SMA検証完了")

        except Exception as e:
            print(f"   ❌ SMAテスト失敗: {e}")
            return False

        # EMAテスト
        print("\n2. EMA計算テスト")
        try:
            ema_result = TALibAdapter.ema(test_data, period=20)
            print(f"   ✅ EMA計算成功")
            print(f"   📈 結果の型: {type(ema_result)}")
            print(f"   🏷️ 名前: {ema_result.name}")
            print(f"   📈 最後の値: {ema_result.iloc[-1]:.2f}")

            assert isinstance(ema_result, pd.Series)
            assert ema_result.name == "EMA_20"
            print("   ✅ EMA検証完了")

        except Exception as e:
            print(f"   ❌ EMAテスト失敗: {e}")
            return False

        # RSIテスト
        print("\n3. RSI計算テスト")
        try:
            rsi_result = TALibAdapter.rsi(test_data, period=14)
            print(f"   ✅ RSI計算成功")
            print(f"   📈 結果の型: {type(rsi_result)}")
            print(f"   🏷️ 名前: {rsi_result.name}")
            print(f"   📈 最後の値: {rsi_result.iloc[-1]:.2f}")

            # RSIの範囲チェック
            valid_values = rsi_result.dropna()
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()
            print("   ✅ RSI範囲検証完了 (0-100)")

        except Exception as e:
            print(f"   ❌ RSIテスト失敗: {e}")
            return False

        # MACDテスト
        print("\n4. MACD計算テスト")
        try:
            macd_result = TALibAdapter.macd(test_data, fast=12, slow=26, signal=9)
            print(f"   ✅ MACD計算成功")
            print(f"   📈 結果の型: {type(macd_result)}")
            print(f"   🔑 キー: {list(macd_result.keys())}")

            assert isinstance(macd_result, dict)
            assert "macd_line" in macd_result
            assert "signal_line" in macd_result
            assert "histogram" in macd_result

            for key, series in macd_result.items():
                assert isinstance(series, pd.Series)
                print(f"   📊 {key}: {series.iloc[-1]:.4f}")

            print("   ✅ MACD検証完了")

        except Exception as e:
            print(f"   ❌ MACDテスト失敗: {e}")
            return False

        # エラーハンドリングテスト
        print("\n5. エラーハンドリングテスト")
        try:
            # 空のSeriesでエラーが発生することを確認
            empty_series = pd.Series([], dtype=float)
            try:
                TALibAdapter.sma(empty_series, period=20)
                print("   ❌ 空データでエラーが発生しませんでした")
                return False
            except TALibCalculationError:
                print("   ✅ 空データエラー正常")

            # 期間が不正な場合
            valid_series = pd.Series([1, 2, 3, 4, 5])
            try:
                TALibAdapter.sma(valid_series, period=0)
                print("   ❌ 不正期間でエラーが発生しませんでした")
                return False
            except TALibCalculationError:
                print("   ✅ 不正期間エラー正常")

            print("   ✅ エラーハンドリング検証完了")

        except Exception as e:
            print(f"   ❌ エラーハンドリングテスト失敗: {e}")
            return False

        print("\n🎉 全てのテストが成功しました！")
        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n🚀 パフォーマンス比較テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.talib_adapter import TALibAdapter
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
        talib_result = TALibAdapter.sma(test_data, period=20)
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
        diff = (talib_result - pandas_result).abs().max()
        print(f"📊 最大差分: {diff:.10f}")

        if diff < 1e-10:
            print("✅ 計算精度: 完全一致")
        elif diff < 1e-6:
            print("✅ 計算精度: 高精度")
        else:
            print("⚠️ 計算精度: 差分あり")

        return True

    except Exception as e:
        print(f"❌ パフォーマンステスト失敗: {e}")
        return False


if __name__ == "__main__":
    print("🔬 TALibAdapter 包括的テスト")
    print("=" * 60)

    # 基本テスト
    basic_success = test_talib_adapter()

    # パフォーマンステスト
    if basic_success:
        perf_success = test_performance_comparison()
    else:
        perf_success = False

    # 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    print(f"基本機能テスト: {'✅ 成功' if basic_success else '❌ 失敗'}")
    print(f"パフォーマンステスト: {'✅ 成功' if perf_success else '❌ 失敗'}")

    if basic_success and perf_success:
        print("\n🎉 TALibAdapter は正常に動作しています！")
        print("次のステップ: 既存指標クラスの更新")
    else:
        print("\n⚠️ 問題が発見されました。修正が必要です。")
