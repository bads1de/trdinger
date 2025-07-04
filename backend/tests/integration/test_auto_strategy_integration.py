#!/usr/bin/env python3
"""
オートストラテジー統合テスト
新規実装された指標がオートストラテジー生成で使用可能かテスト
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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


def test_random_gene_generator():
    """RandomGeneGeneratorのテスト"""
    print("\nRandomGeneGenerator 新規指標統合テスト")
    print("=" * 60)

    try:
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # GAConfigのインスタンス化
        ga_config = GAConfig()

        # ジェネレーターのインスタンス化
        generator = RandomGeneGenerator(config=ga_config)
        print("RandomGeneGeneratorのインスタンス化成功")

        # 新規指標が含まれているか確認
        new_indicators = [
            "VWMA",
            "VWAP",
            "KELTNER",
            "STOCHRSI",
            "ULTOSC",
            "CMO",
            "TRIX",
            "MAMA",
            "STDDEV",
        ]

        print(f"\n利用可能な指標数: {len(generator.available_indicators)}")
        print("新規指標の統合状況:")

        all_included = True
        for indicator in new_indicators:
            included = indicator in generator.available_indicators
            status = "  "  # 絵文字を削除
            print(f"   {status} {indicator}: {'統合済み' if included else '未統合'}")
            if not included:
                all_included = False

        if all_included:
            print("\n全ての新規指標が統合されています！")
        else:
            print("\n一部の新規指標が未統合です")
            return False

        # ランダム遺伝子の生成テスト
        print("\n📋 ランダム遺伝子生成テスト:")
        macd_found = False
        for i in range(10):  # 試行回数を増やす
            gene = generator.generate_random_gene()
            print(f"   遺伝子 {i+1}: {len(gene.indicators)}個の指標")
            for indicator in gene.indicators:
                print(f"     - {indicator.type}: {indicator.parameters}")
                if indicator.type == "MACD":
                    macd_found = True
                    print(f"       ✨ MACD指標が生成されました: {indicator.parameters}")
                    # ここでMACDを含む遺伝子をさらに詳細に検証することも可能
                    # 例: MACDのfast_period < slow_period が満たされているか
                    if indicator.parameters.get(
                        "fast_period", 0
                    ) >= indicator.parameters.get("slow_period", 0):
                        print(
                            "       ⚠️  MACD: fast_period >= slow_period の問題が検出されました！"
                        )
                        return False  # テスト失敗
            if macd_found:
                break  # MACDが見つかったらループを抜ける

        if not macd_found:
            print("⚠️  MACD指標が生成されませんでした。テストを続行します。")
            # MACDが生成されない場合でもテストを失敗させないが、警告を出す
            # これは、ランダム生成のため必ずしもMACDが生成されるとは限らないため
            # より厳密なテストには、生成されるまでループするか、モックを使用する

        print("ランダム遺伝子生成成功")
        return True

    except Exception as e:
        print(f"RandomGeneGeneratorテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ga_config():
    """GAConfigのテスト"""
    print("\nGAConfig 新規指標統合テスト")
    print("=" * 60)

    try:
        from app.core.services.auto_strategy.models.ga_config import GAConfig

        # GAConfigのインスタンス化
        config = GAConfig()
        print("GAConfigのインスタンス化成功")

        # 新規指標が含まれているか確認
        new_indicators = [
            "VWMA",
            "VWAP",
            "KELTNER",
            "STOCHRSI",
            "ULTOSC",
            "CMO",
            "TRIX",
            "MAMA",
            "STDDEV",
        ]

        print(f"\n許可された指標数: {len(config.allowed_indicators)}")
        print("新規指標の統合状況:")

        all_included = True
        for indicator in new_indicators:
            included = indicator in config.allowed_indicators
            status = "  "  # 絵文字を削除
            print(f"   {status} {indicator}: {'統合済み' if included else '未統合'}")
            if not included:
                all_included = False

        if all_included:
            print("\n全ての新規指標が統合されています！")
        else:
            print("\n一部の新規指標が未統合です")
            return False

        print("GAConfig統合テスト成功")
        return True

    except Exception as e:
        print(f"GAConfigテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_indicator_calculation():
    """新規指標の計算テスト"""
    print("\n新規指標計算テスト")
    print("=" * 60)

    test_data = create_test_data(100)
    print(f"テストデータ作成: {len(test_data)}件")

    # 新規指標のテスト
    new_indicators_tests = [
        ("VWMA", "app.core.services.indicators.trend", "VWMAIndicator"),
        ("VWAP", "app.core.services.indicators.volume", "VWAPIndicator"),
        (
            "KELTNER",
            "app.core.services.indicators.volatility",
            "KeltnerChannelsIndicator",
        ),
        (
            "STOCHRSI",
            "app.core.services.indicators.momentum",
            "StochasticRSIIndicator",
        ),
        (
            "ULTOSC",
            "app.core.services.indicators.momentum",
            "UltimateOscillatorIndicator",
        ),
        ("CMO", "app.core.services.indicators.momentum", "CMOIndicator"),
        ("TRIX", "app.core.services.indicators.momentum", "TRIXIndicator"),
        ("MAMA", "app.core.services.indicators.trend", "MAMAIndicator"),
        (
            "STDDEV",
            "app.core.services.indicators.volatility",
            "STDDEVIndicator",
        ),
    ]

    success_count = 0
    for indicator_type, module_name, class_name in new_indicators_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            indicator = indicator_class()

            # 適切な期間で計算テスト
            period = (
                indicator.supported_periods[0] if indicator.supported_periods else 14
            )
            result = indicator.calculate(test_data, period)

            print(f"{indicator_type}: 計算成功 (期間: {period})")
            success_count += 1

        except Exception as e:
            print(f"{indicator_type}: 計算失敗 - {e}")

    print(f"\n計算テスト結果: {success_count}/{len(new_indicators_tests)} 成功")
    return success_count == len(new_indicators_tests)


def main():
    """メインテスト実行"""
    print("オートストラテジー新規指標統合テスト開始")
    print("=" * 80)

    tests = [
        ("RandomGeneGenerator統合", test_random_gene_generator),
        ("GAConfig統合", test_ga_config),
        ("新規指標計算", test_indicator_calculation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}のテスト:")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 80)
    print("テスト結果サマリー:")
    print("=" * 80)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("全てのテストが成功しました！")
        print(
            "新規実装された10個の指標がオートストラテジー生成で使用可能になりました。"
        )
        print("戦略の多様性と精度の向上が期待されます。")
    else:
        print("一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    print("=" * 80)


if __name__ == "__main__":
    main()
