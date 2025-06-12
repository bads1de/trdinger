#!/usr/bin/env python3
"""
Ultimate Oscillator実装のテストスクリプト

新しく実装したUltimateOscillatorIndicatorクラスの動作確認を行います。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))


def test_ultimate_oscillator_indicator():
    """UltimateOscillatorIndicatorクラスのテスト"""
    try:
        from app.core.services.indicators import UltimateOscillatorIndicator

        print("✅ UltimateOscillatorIndicatorのインポート成功")

        # テストデータの作成（Ultimate Oscillatorは高値・安値・終値データが必要）
        dates = pd.date_range("2023-01-01", periods=150, freq="D")

        # より現実的な価格データを生成
        base_price = 100
        price_trend = np.linspace(0, 20, 150)  # 上昇トレンド
        price_noise = np.random.normal(0, 2, 150)  # ノイズ
        close_prices = base_price + price_trend + price_noise

        # 高値・安値を終値から生成
        high_prices = close_prices + np.random.uniform(0.5, 1.5, 150)
        low_prices = close_prices - np.random.uniform(0.5, 1.5, 150)

        test_data = pd.DataFrame(
            {
                "open": close_prices + np.random.uniform(-1, 1, 150),
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 150),
            },
            index=dates,
        )

        # UltimateOscillatorIndicatorのインスタンス化
        ultosc_indicator = UltimateOscillatorIndicator()
        print("✅ UltimateOscillatorIndicatorのインスタンス化成功")
        print(f"   サポート期間: {ultosc_indicator.supported_periods}")

        # 異なる期間でのUltimate Oscillator計算テスト
        for period in [7, 14, 28]:
            try:
                result = ultosc_indicator.calculate(test_data, period)

                print(f"✅ Ultimate Oscillator計算成功 (期間: {period})")
                print(f"   結果の型: {type(result)}")
                print(f"   結果の長さ: {len(result)}")
                print(f"   非NaN値の数: {result.notna().sum()}")
                print(f"   値の範囲: {result.min():.2f} - {result.max():.2f}")
                print(f"   最後の5つの値:")
                print(f"   {result.tail().round(2)}")
                print()

            except Exception as e:
                print(f"❌ Ultimate Oscillator計算失敗 (期間: {period}): {e}")
                return False

        # 説明の取得テスト
        description = ultosc_indicator.get_description()
        print(f"✅ 説明取得成功: {description}")

        return True

    except Exception as e:
        print(f"❌ UltimateOscillatorIndicatorテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ultimate_oscillator_vs_rsi():
    """Ultimate OscillatorとRSIの比較テスト"""
    try:
        from app.core.services.indicators import (
            UltimateOscillatorIndicator,
            RSIIndicator,
        )

        print("\n📊 Ultimate OscillatorとRSIの比較テスト:")

        # テストデータの作成
        dates = pd.date_range("2023-01-01", periods=80, freq="D")

        # 複雑な価格パターン: 急上昇 → 調整 → 再上昇
        price_pattern = np.concatenate(
            [
                np.linspace(100, 130, 25),  # 急上昇
                np.linspace(130, 115, 15),  # 調整
                np.linspace(115, 140, 40),  # 再上昇
            ]
        )

        # ノイズを追加
        close_prices = price_pattern + np.random.normal(0, 1, 80)
        high_prices = close_prices + np.random.uniform(0.5, 2, 80)
        low_prices = close_prices - np.random.uniform(0.5, 2, 80)

        test_data = pd.DataFrame(
            {
                "open": close_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 80),
            },
            index=dates,
        )

        period = 14

        # 各指標を計算
        ultosc_indicator = UltimateOscillatorIndicator()
        rsi_indicator = RSIIndicator()

        ultosc_result = ultosc_indicator.calculate(test_data, period)
        rsi_result = rsi_indicator.calculate(test_data, period)

        # 結果の比較（最後の10個の値）
        print(f"   期間: {period}")
        print(f"   価格パターン: 急上昇 → 調整 → 再上昇")
        print(f"   最後の10個の値の比較:")

        comparison_df = pd.DataFrame(
            {
                "Close": test_data["close"].tail(10).round(2),
                "RSI": rsi_result.tail(10).round(2),
                "UltOsc": ultosc_result.tail(10).round(2),
            }
        )

        print(comparison_df)

        # 感度の比較
        rsi_volatility = rsi_result.std()
        ultosc_volatility = ultosc_result.std()

        print(f"\n   ボラティリティ比較:")
        print(f"   RSI標準偏差: {rsi_volatility:.2f}")
        print(f"   Ultimate Oscillator標準偏差: {ultosc_volatility:.2f}")

        # Ultimate Oscillatorは複数期間を使用するため、一般的にRSIより安定
        if ultosc_volatility < rsi_volatility:
            print(
                f"   ✅ Ultimate OscillatorがRSIより安定（{ultosc_volatility:.2f} < {rsi_volatility:.2f}）"
            )
        else:
            print(f"   ⚠️  安定性の違いが期待通りでない可能性")

        return True

    except Exception as e:
        print(f"❌ 比較テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ultimate_oscillator_parameters():
    """Ultimate Oscillatorのパラメータテスト"""
    try:
        from app.core.services.indicators import UltimateOscillatorIndicator

        print("\n🔢 Ultimate Oscillatorのパラメータテスト:")

        # テストデータの作成
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # トレンド変化のある価格データ
        base_price = 100
        trend_changes = np.concatenate(
            [
                np.linspace(100, 120, 35),  # 上昇
                np.linspace(120, 110, 30),  # 下降
                np.linspace(110, 130, 35),  # 再上昇
            ]
        )

        close_prices = trend_changes + np.random.normal(0, 1, 100)
        high_prices = close_prices + np.random.uniform(0.5, 2, 100)
        low_prices = close_prices - np.random.uniform(0.5, 2, 100)

        test_data = pd.DataFrame(
            {
                "open": close_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        ultosc_indicator = UltimateOscillatorIndicator()

        # 異なるパラメータセットでの計算
        parameter_sets = [
            (7, 14, 28),  # デフォルト
            (5, 10, 20),  # 短期重視
            (10, 20, 40),  # 長期重視
            (7, 21, 42),  # カスタム
        ]

        results = {}

        for period1, period2, period3 in parameter_sets:
            result = ultosc_indicator.calculate(
                test_data, period1, period2=period2, period3=period3
            )
            results[(period1, period2, period3)] = result

            # 最終値と標準偏差の表示
            final_value = result.iloc[-1]
            volatility = result.std()
            print(
                f"   期間({period1}, {period2}, {period3}): 最終値={final_value:.2f}, 標準偏差={volatility:.2f}"
            )

        # パラメータの影響確認
        print(f"\n   パラメータ効果の確認:")

        # デフォルトとの比較
        default_result = results[(7, 14, 28)]
        default_volatility = default_result.std()

        for params, result in results.items():
            if params == (7, 14, 28):
                continue

            period1, period2, period3 = params
            current_volatility = result.std()
            volatility_ratio = current_volatility / default_volatility

            print(
                f"   期間({period1}, {period2}, {period3}): ボラティリティ比率={volatility_ratio:.2f}"
            )

            if period1 < 7:  # 短期重視
                print(f"     → 短期重視: より敏感な反応")
            elif period1 > 7:  # 長期重視
                print(f"     → 長期重視: より安定した反応")

        return True

    except Exception as e:
        print(f"❌ パラメータテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ultimate_oscillator_integration():
    """Ultimate Oscillatorの統合テスト"""
    try:
        from app.core.services.indicators import get_indicator_by_type

        print("\n🔗 Ultimate Oscillator統合テスト:")

        # ファクトリー関数経由での取得
        ultosc_indicator = get_indicator_by_type("ULTOSC")
        print("✅ ファクトリー関数からのUltimate Oscillator取得成功")
        print(f"   指標タイプ: {ultosc_indicator.indicator_type}")
        print(f"   サポート期間: {ultosc_indicator.supported_periods}")

        return True

    except Exception as e:
        print(f"❌ 統合テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("🧪 Ultimate Oscillator実装テスト開始\n")

    tests = [
        ("UltimateOscillatorIndicatorクラス", test_ultimate_oscillator_indicator),
        ("Ultimate OscillatorとRSIの比較", test_ultimate_oscillator_vs_rsi),
        ("Ultimate Oscillatorのパラメータ", test_ultimate_oscillator_parameters),
        ("Ultimate Oscillator統合", test_ultimate_oscillator_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}のテスト:")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー:")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 全てのテストが成功しました！")
        print("Ultimate Oscillator の実装が完了しています。")
        print("Ultimate Oscillatorは複数期間のTrue Rangeベースのモメンタム指標です。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
