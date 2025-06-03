#!/usr/bin/env python3
"""
テクニカル指標サービスの包括的テスト

現在の実装に合わせて、全てのアダプタークラスと指標クラスをテストします。
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_adapters_import():
    """アダプタークラスのインポートテスト"""
    print("🧪 アダプタークラス インポートテスト")
    print("=" * 50)

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
        return True, {
            "BaseAdapter": BaseAdapter,
            "TALibCalculationError": TALibCalculationError,
            "TrendAdapter": TrendAdapter,
            "MomentumAdapter": MomentumAdapter,
            "VolatilityAdapter": VolatilityAdapter,
            "VolumeAdapter": VolumeAdapter,
        }
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        traceback.print_exc()
        return False, {}


def test_trend_adapter(adapters):
    """TrendAdapterのテスト"""
    print("\n🧪 TrendAdapter テスト")
    print("=" * 50)

    try:
        TrendAdapter = adapters["TrendAdapter"]
        TALibCalculationError = adapters["TALibCalculationError"]

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
        sma_result = TrendAdapter.sma(test_data, period=20)
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(test_data)
        assert sma_result.name == "SMA_20"
        print(f"   ✅ SMA計算成功: 最後の値 {sma_result.iloc[-1]:.2f}")

        # EMAテスト
        print("\n2. EMA計算テスト")
        ema_result = TrendAdapter.ema(test_data, period=20)
        assert isinstance(ema_result, pd.Series)
        assert ema_result.name == "EMA_20"
        print(f"   ✅ EMA計算成功: 最後の値 {ema_result.iloc[-1]:.2f}")

        # TEMAテスト
        print("\n3. TEMA計算テスト")
        tema_result = TrendAdapter.tema(test_data, period=30)
        assert isinstance(tema_result, pd.Series)
        assert tema_result.name == "TEMA_30"
        print(f"   ✅ TEMA計算成功: 最後の値 {tema_result.iloc[-1]:.2f}")

        # エラーハンドリングテスト
        print("\n4. エラーハンドリングテスト")
        empty_series = pd.Series([], dtype=float)
        try:
            TrendAdapter.sma(empty_series, period=20)
            print("   ❌ 空データでエラーが発生しませんでした")
            return False
        except TALibCalculationError:
            print("   ✅ 空データエラー正常")

        print("✅ TrendAdapter 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ TrendAdapterテスト失敗: {e}")
        traceback.print_exc()
        return False


def test_momentum_adapter(adapters):
    """MomentumAdapterのテスト"""
    print("\n🧪 MomentumAdapter テスト")
    print("=" * 50)

    try:
        MomentumAdapter = adapters["MomentumAdapter"]
        TALibCalculationError = adapters["TALibCalculationError"]

        # テストデータ作成
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        test_data = pd.Series(prices, index=dates, name="close")

        # RSIテスト
        print("\n1. RSI計算テスト")
        rsi_result = MomentumAdapter.rsi(test_data, period=14)
        assert isinstance(rsi_result, pd.Series)
        assert rsi_result.name == "RSI_14"

        # RSIの範囲チェック
        valid_values = rsi_result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()
            print(f"   ✅ RSI計算成功: 最後の値 {rsi_result.iloc[-1]:.2f}")

        # モメンタムテスト
        print("\n2. モメンタム計算テスト")
        mom_result = MomentumAdapter.momentum(test_data, period=10)
        assert isinstance(mom_result, pd.Series)
        assert mom_result.name == "MOM_10"
        print(f"   ✅ モメンタム計算成功: 最後の値 {mom_result.iloc[-1]:.2f}")

        print("✅ MomentumAdapter 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ MomentumAdapterテスト失敗: {e}")
        traceback.print_exc()
        return False


def test_volatility_adapter(adapters):
    """VolatilityAdapterのテスト"""
    print("\n🧪 VolatilityAdapter テスト")
    print("=" * 50)

    try:
        VolatilityAdapter = adapters["VolatilityAdapter"]

        # OHLCVテストデータ作成
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))

        # ATRテスト
        print("\n1. ATR計算テスト")
        atr_result = VolatilityAdapter.atr(
            pd.Series(high_prices, index=dates),
            pd.Series(low_prices, index=dates),
            pd.Series(close_prices, index=dates),
            period=14,
        )
        assert isinstance(atr_result, pd.Series)
        assert atr_result.name == "ATR_14"
        print(f"   ✅ ATR計算成功: 最後の値 {atr_result.iloc[-1]:.2f}")

        print("✅ VolatilityAdapter 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ VolatilityAdapterテスト失敗: {e}")
        traceback.print_exc()
        return False


def test_volume_adapter(adapters):
    """VolumeAdapterのテスト"""
    print("\n🧪 VolumeAdapter テスト")
    print("=" * 50)

    try:
        VolumeAdapter = adapters["VolumeAdapter"]

        # OHLCVテストデータ作成
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        volumes = np.random.randint(1000, 10000, 100)

        # A/D Lineテスト
        print("\n1. A/D Line計算テスト")
        ad_result = VolumeAdapter.ad(
            pd.Series(high_prices, index=dates),
            pd.Series(low_prices, index=dates),
            pd.Series(close_prices, index=dates),
            pd.Series(volumes, index=dates),
        )
        assert isinstance(ad_result, pd.Series)
        print(f"   ✅ A/D Line計算成功: 最後の値 {ad_result.iloc[-1]:.2f}")

        print("✅ VolumeAdapter 全テスト成功")
        return True

    except Exception as e:
        print(f"❌ VolumeAdapterテスト失敗: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔬 テクニカル指標サービス 包括的テスト")
    print("=" * 60)

    # テスト実行
    import_success, adapters = test_adapters_import()

    results = {
        "import": import_success,
        "trend": False,
        "momentum": False,
        "volatility": False,
        "volume": False,
    }

    if import_success:
        results["trend"] = test_trend_adapter(adapters)
        results["momentum"] = test_momentum_adapter(adapters)
        results["volatility"] = test_volatility_adapter(adapters)
        results["volume"] = test_volume_adapter(adapters)

    # 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name.capitalize()}テスト: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\n📊 総合結果: {passed_tests}/{total_tests} 成功")

    if passed_tests == total_tests:
        print("🎉 全てのテストが成功しました！")
    else:
        print("⚠️ 一部のテストが失敗しました。修正が必要です。")
