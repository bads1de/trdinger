#!/usr/bin/env python3
"""
IndicatorOrchestratorの包括的テスト

メインの統合サービスであるIndicatorOrchestratorクラスをテストします。
"""

import sys
import os
import pandas as pd
import numpy as np
import asyncio
import traceback

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


async def test_indicator_orchestrator_import():
    """IndicatorOrchestratorのインポートテスト"""
    print("🧪 IndicatorOrchestrator インポートテスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.indicator_orchestrator import (
            IndicatorOrchestrator,
        )

        print("✅ IndicatorOrchestrator インポート成功")

        # インスタンス作成
        orchestrator = IndicatorOrchestrator()
        print("✅ IndicatorOrchestrator インスタンス作成成功")

        return True, orchestrator
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        traceback.print_exc()
        return False, None


async def test_supported_indicators(orchestrator):
    """サポートされている指標の確認テスト"""
    print("\n🧪 サポート指標確認テスト")
    print("=" * 50)

    try:
        supported = orchestrator.get_supported_indicators()
        print(f"📊 サポートされている指標数: {len(supported)}")

        for indicator_type, info in supported.items():
            periods = info.get("periods", [])
            description = info.get("description", "N/A")
            print(f"   {indicator_type}: {periods} - {description}")

        # 基本的な指標が含まれていることを確認
        expected_indicators = ["SMA", "EMA", "RSI"]
        for indicator in expected_indicators:
            if indicator in supported:
                print(f"   ✅ {indicator} サポート確認")
            else:
                print(f"   ❌ {indicator} サポートなし")
                return False

        print("✅ サポート指標確認完了")
        return True
    except Exception as e:
        print(f"❌ サポート指標確認エラー: {e}")
        traceback.print_exc()
        return False


async def test_parameter_validation(orchestrator):
    """パラメータ検証テスト"""
    print("\n🧪 パラメータ検証テスト")
    print("=" * 50)

    try:
        # 無効な指標タイプ
        try:
            orchestrator._validate_parameters("BTCUSDT", "1d", "INVALID_INDICATOR", 20)
            print("   ❌ 無効な指標タイプでエラーが発生しませんでした")
            return False
        except ValueError:
            print("   ✅ 無効な指標タイプで適切にエラー発生")

        # 無効な期間
        try:
            orchestrator._validate_parameters("BTCUSDT", "1d", "SMA", 999)
            print("   ❌ 無効な期間でエラーが発生しませんでした")
            return False
        except ValueError:
            print("   ✅ 無効な期間で適切にエラー発生")

        # 有効なパラメータ
        try:
            orchestrator._validate_parameters("BTCUSDT", "1d", "SMA", 20)
            print("   ✅ 有効なパラメータで正常処理")
        except Exception as e:
            print(f"   ❌ 有効なパラメータでエラー: {e}")
            return False

        print("✅ パラメータ検証テスト完了")
        return True
    except Exception as e:
        print(f"❌ パラメータ検証テストエラー: {e}")
        traceback.print_exc()
        return False


async def test_indicator_instance_creation(orchestrator):
    """指標インスタンス作成テスト"""
    print("\n🧪 指標インスタンス作成テスト")
    print("=" * 50)

    try:
        # SMA指標インスタンス取得
        sma_indicator = orchestrator._get_indicator_instance("SMA")
        print(f"   ✅ SMA指標インスタンス取得: {type(sma_indicator)}")

        # EMA指標インスタンス取得
        ema_indicator = orchestrator._get_indicator_instance("EMA")
        print(f"   ✅ EMA指標インスタンス取得: {type(ema_indicator)}")

        # RSI指標インスタンス取得
        rsi_indicator = orchestrator._get_indicator_instance("RSI")
        print(f"   ✅ RSI指標インスタンス取得: {type(rsi_indicator)}")

        # 無効な指標タイプ
        try:
            invalid_indicator = orchestrator._get_indicator_instance("INVALID")
            print("   ❌ 無効な指標タイプでエラーが発生しませんでした")
            return False
        except ValueError:
            print("   ✅ 無効な指標タイプで適切にエラー発生")

        print("✅ 指標インスタンス作成テスト完了")
        return True
    except Exception as e:
        print(f"❌ 指標インスタンス作成テストエラー: {e}")
        traceback.print_exc()
        return False


async def test_mock_calculation():
    """モック計算テスト（実際のデータベースを使わない）"""
    print("\n🧪 モック計算テスト")
    print("=" * 50)

    try:
        from app.core.services.indicators.trend_indicators import SMAIndicator

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

        print(f"📊 テストデータ作成: {len(test_data)}件")

        # SMA指標で直接計算
        sma_indicator = SMAIndicator()
        sma_result = sma_indicator.calculate(test_data, period=20)

        print(f"✅ SMA計算成功")
        print(f"   📈 結果の型: {type(sma_result)}")
        print(f"   📊 データ長: {len(sma_result)}")
        print(f"   🏷️ 名前: {sma_result.name}")
        print(f"   📈 最後の値: {sma_result.iloc[-1]:.2f}")

        # 基本的な検証
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(test_data)
        assert sma_result.name == "SMA_20"

        print("✅ モック計算テスト完了")
        return True
    except Exception as e:
        print(f"❌ モック計算テストエラー: {e}")
        traceback.print_exc()
        return False


async def main():
    """メインテスト関数"""
    print("🔬 IndicatorOrchestrator 包括的テスト")
    print("=" * 60)

    # テスト実行
    import_success, orchestrator = await test_indicator_orchestrator_import()

    results = {
        "import": import_success,
        "supported_indicators": False,
        "parameter_validation": False,
        "instance_creation": False,
        "mock_calculation": False,
    }

    if import_success and orchestrator:
        results["supported_indicators"] = await test_supported_indicators(orchestrator)
        results["parameter_validation"] = await test_parameter_validation(orchestrator)
        results["instance_creation"] = await test_indicator_instance_creation(
            orchestrator
        )

    # モック計算テスト（オーケストレーターに依存しない）
    results["mock_calculation"] = await test_mock_calculation()

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
        print("🎉 全てのテストが成功しました！")
        print("次のステップ: 実際のデータベース統合テスト")
    else:
        print("⚠️ 一部のテストが失敗しました。修正が必要です。")


if __name__ == "__main__":
    asyncio.run(main())
