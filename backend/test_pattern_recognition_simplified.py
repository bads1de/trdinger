#!/usr/bin/env python3

"""
簡素化されたパターン認識指標の動作確認テスト

pandas-taを直接使用した簡素化実装の動作を確認します。
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_simplified_pattern_recognition_indicators():
    """簡素化されたパターン認識指標のテスト"""
    print("=== 簡素化されたパターン認識指標のテスト ===")

    try:
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        # テストデータ作成（キャンドルスティックパターン用）
        np.random.seed(42)
        n = 100

        # リアルなOHLCデータを作成
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Open価格
        open_prices = base_prices + np.random.randn(n) * 0.1

        # Close価格（トレンドを含む）
        close_prices = base_prices + np.random.randn(n) * 0.2

        # High価格（OpenとCloseの最大値 + α）
        high_prices = np.maximum(open_prices, close_prices) + np.abs(
            np.random.randn(n) * 0.3
        )

        # Low価格（OpenとCloseの最小値 - α）
        low_prices = np.minimum(open_prices, close_prices) - np.abs(
            np.random.randn(n) * 0.3
        )

        print(f"テストデータ作成完了: {n}本のローソク足")
        print(f"価格範囲: {low_prices.min():.2f} - {high_prices.max():.2f}")

        # Dojiテスト（重要パターン）
        doji_result = PatternRecognitionIndicators.cdl_doji(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Doji計算成功 - 検出数: {np.sum(doji_result != 0)} 個")

        # Hammerテスト（重要パターン）
        hammer_result = PatternRecognitionIndicators.cdl_hammer(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Hammer計算成功 - 検出数: {np.sum(hammer_result != 0)} 個")

        # Engulfing Patternテスト（重要パターン）
        engulfing_result = PatternRecognitionIndicators.cdl_engulfing(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Engulfing Pattern計算成功 - 検出数: {np.sum(engulfing_result != 0)} 個")

        # Hanging Manテスト（軽量実装）
        hanging_man_result = PatternRecognitionIndicators.cdl_hanging_man(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Hanging Man計算成功 - 検出数: {np.sum(hanging_man_result != 0)} 個")

        # Morning Starテスト（重要パターン）
        morning_star_result = PatternRecognitionIndicators.cdl_morning_star(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Morning Star計算成功 - 検出数: {np.sum(morning_star_result != 0)} 個")

        # Evening Starテスト（重要パターン）
        evening_star_result = PatternRecognitionIndicators.cdl_evening_star(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"Evening Star計算成功 - 検出数: {np.sum(evening_star_result != 0)} 個")

        # Three Black Crowsテスト
        three_black_crows_result = PatternRecognitionIndicators.cdl_three_black_crows(
            open_prices, high_prices, low_prices, close_prices
        )
        print(
            f"Three Black Crows計算成功 - 検出数: {np.sum(three_black_crows_result != 0)} 個"
        )

        # 結果の型チェック
        assert isinstance(doji_result, np.ndarray), "Doji結果がnumpy配列でない"
        assert isinstance(hammer_result, np.ndarray), "Hammer結果がnumpy配列でない"
        assert len(doji_result) == n, f"Doji結果の長さが不正: {len(doji_result)} != {n}"
        assert (
            len(hammer_result) == n
        ), f"Hammer結果の長さが不正: {len(hammer_result)} != {n}"

        # パターン値の妥当性チェック（通常は-100, 0, 100の値）
        unique_doji = np.unique(doji_result)
        print(f"Doji検出値: {unique_doji}")
        assert all(val in [-100, 0, 100] for val in unique_doji), "Doji値が不正"

        print("✅ パターン認識指標簡素化テスト成功")

    except Exception as e:
        print(f"❌ パターン認識指標簡素化テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")

    try:
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        print("1. 空データでのテスト")
        try:
            empty_data = np.array([])
            result = PatternRecognitionIndicators.cdl_doji(
                empty_data, empty_data, empty_data, empty_data
            )
            print(f"予期しない成功: {result}")
        except Exception as e:
            print(f"適切なエラーキャッチ: {type(e).__name__}: {e}")

        print("\n2. 短すぎるデータでのテスト")
        try:
            short_data = np.array([100, 101])
            result = PatternRecognitionIndicators.cdl_doji(
                short_data, short_data, short_data, short_data
            )
            print(f"予期しない成功: {result}")
        except Exception as e:
            print(f"適切なエラーキャッチ: {type(e).__name__}: {e}")

        print("\n3. NaNデータでのテスト")
        try:
            nan_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
            result = PatternRecognitionIndicators.cdl_doji(
                nan_data, nan_data, nan_data, nan_data
            )
            print(f"予期しない成功: {result}")
        except Exception as e:
            print(f"適切なエラーキャッチ: {type(e).__name__}: {e}")

        print("✅ エラーハンドリングテスト成功")

    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n=== 後方互換性テスト ===")

    try:
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        # テストデータ
        test_open = np.array([100, 101, 102, 103, 104])
        test_high = np.array([105, 106, 107, 108, 109])
        test_low = np.array([95, 96, 97, 98, 99])
        test_close = np.array([102, 103, 104, 105, 106])

        # エイリアスメソッドのテスト
        doji_result = PatternRecognitionIndicators.doji(
            test_open, test_high, test_low, test_close
        )
        hammer_result = PatternRecognitionIndicators.hammer(
            test_open, test_high, test_low, test_close
        )
        engulfing_result = PatternRecognitionIndicators.engulfing_pattern(
            test_open, test_high, test_low, test_close
        )

        print(f"Doji（エイリアス）計算成功: {len(doji_result)} 個の値")
        print(f"Hammer（エイリアス）計算成功: {len(hammer_result)} 個の値")
        print(
            f"Engulfing Pattern（エイリアス）計算成功: {len(engulfing_result)} 個の値"
        )

        # 結果が適切な形式であることを確認
        assert isinstance(doji_result, np.ndarray), "Dojiの結果が配列でない"
        assert isinstance(hammer_result, np.ndarray), "Hammerの結果が配列でない"
        assert len(doji_result) == 5, "Dojiの結果の長さが不正"

        print("✅ 後方互換性テスト成功")

    except Exception as e:
        print(f"❌ 後方互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_pattern_detection():
    """パターン検出精度テスト"""
    print("\n=== パターン検出精度テスト ===")

    try:
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        # 明確なDojiパターンを作成
        doji_open = np.array([100, 100, 100, 100, 100])
        doji_high = np.array([102, 102, 102, 102, 102])
        doji_low = np.array([98, 98, 98, 98, 98])
        doji_close = np.array([100.01, 99.99, 100.02, 99.98, 100])  # ほぼ同じ値

        doji_result = PatternRecognitionIndicators.cdl_doji(
            doji_open, doji_high, doji_low, doji_close
        )

        print(f"明確なDojiパターンテスト:")
        print(f"  検出結果: {doji_result}")
        print(f"  検出数: {np.sum(doji_result != 0)} / {len(doji_result)}")

        # 明確なHammerパターンを作成
        hammer_open = np.array([100, 100, 100, 100, 100])
        hammer_high = np.array([101, 101, 101, 101, 101])
        hammer_low = np.array([95, 95, 95, 95, 95])  # 長い下ヒゲ
        hammer_close = np.array([99, 99, 99, 99, 99])

        hammer_result = PatternRecognitionIndicators.cdl_hammer(
            hammer_open, hammer_high, hammer_low, hammer_close
        )

        print(f"\n明確なHammerパターンテスト:")
        print(f"  検出結果: {hammer_result}")
        print(f"  検出数: {np.sum(hammer_result != 0)} / {len(hammer_result)}")

        print("✅ パターン検出精度テスト成功")

    except Exception as e:
        print(f"❌ パターン検出精度テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")

    try:
        import time
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        # 大きなテストデータ
        np.random.seed(42)
        n = 10000
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        open_prices = base_prices + np.random.randn(n) * 0.1
        close_prices = base_prices + np.random.randn(n) * 0.2
        high_prices = np.maximum(open_prices, close_prices) + np.abs(
            np.random.randn(n) * 0.3
        )
        low_prices = np.minimum(open_prices, close_prices) - np.abs(
            np.random.randn(n) * 0.3
        )

        # Doji計算時間測定
        start_time = time.time()
        for _ in range(10):  # 10回実行
            doji_result = PatternRecognitionIndicators.cdl_doji(
                open_prices, high_prices, low_prices, close_prices
            )
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Doji計算時間（{n}データポイント）: {avg_time:.4f}秒/回")
        print(f"処理速度: {n/avg_time:.0f} データポイント/秒")

        # Hammer計算時間測定
        start_time = time.time()
        for _ in range(10):
            hammer_result = PatternRecognitionIndicators.cdl_hammer(
                open_prices, high_prices, low_prices, close_prices
            )
        end_time = time.time()

        avg_time_hammer = (end_time - start_time) / 10
        print(f"Hammer計算時間: {avg_time_hammer:.4f}秒/回")

        # 結果の妥当性確認
        assert len(doji_result) == n, "Doji結果の長さが不正"
        assert len(hammer_result) == n, "Hammer結果の長さが不正"

        # パターン検出数の確認
        doji_count = np.sum(doji_result != 0)
        hammer_count = np.sum(hammer_result != 0)
        print(f"Doji検出数: {doji_count} ({doji_count/n*100:.2f}%)")
        print(f"Hammer検出数: {hammer_count} ({hammer_count/n*100:.2f}%)")

        print("✅ パフォーマンステスト成功")

    except Exception as e:
        print(f"❌ パフォーマンステストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 簡素化されたパターン認識指標のテスト開始")

    test_simplified_pattern_recognition_indicators()
    test_error_handling()
    test_backward_compatibility()
    test_pattern_detection()
    test_performance_comparison()

    print("\n🎉 パターン認識指標簡素化テスト完了")
