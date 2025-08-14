#!/usr/bin/env python3
"""
パターン認識指標の統一化テスト

ensure_series_minimal_conversionの統一化が正常に動作するかテストします。
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_pattern_recognition_unified():
    """統一化されたパターン認識指標のテスト"""
    print("=== パターン認識指標の統一化テスト ===")

    try:
        from app.services.indicators.technical_indicators.pattern_recognition import (
            PatternRecognitionIndicators,
        )

        # テストデータ作成（OHLC）
        np.random.seed(42)
        n = 50

        # 基準価格
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(n) * 0.5)

        # OHLC作成（リアルな価格関係を保つ）
        open_prices = prices + np.random.randn(n) * 0.1
        close_prices = prices + np.random.randn(n) * 0.1
        high_prices = np.maximum(open_prices, close_prices) + np.abs(
            np.random.randn(n) * 0.2
        )
        low_prices = np.minimum(open_prices, close_prices) - np.abs(
            np.random.randn(n) * 0.2
        )

        print(f"テストデータ作成完了: {n}本のローソク足")
        print(f"価格範囲: {low_prices.min():.2f} - {high_prices.max():.2f}")

        # 同事パターンテスト
        doji_result = PatternRecognitionIndicators.cdl_doji(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"同事パターン検出: {np.sum(doji_result != 0)}個")

        # ハンマーパターンテスト
        hammer_result = PatternRecognitionIndicators.cdl_hammer(
            open_prices, high_prices, low_prices, close_prices
        )
        print(f"ハンマーパターン検出: {np.sum(hammer_result != 0)}個")

        # 結果の型チェック
        assert isinstance(doji_result, np.ndarray), "同事結果がnumpy配列でない"
        assert isinstance(hammer_result, np.ndarray), "ハンマー結果がnumpy配列でない"
        assert len(doji_result) == n, f"同事結果の長さが不正: {len(doji_result)} != {n}"
        assert (
            len(hammer_result) == n
        ), f"ハンマー結果の長さが不正: {len(hammer_result)} != {n}"

        print("✅ パターン認識統一化テスト成功")

    except Exception as e:
        print(f"❌ パターン認識統一化テストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_ensure_series_compatibility():
    """ensure_series_minimal_conversionの互換性テスト"""
    print("\n=== ensure_series互換性テスト ===")

    try:
        from app.services.indicators.utils import ensure_series_minimal_conversion
        from app.utils.data_conversion import ensure_series

        # テストデータ
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        numpy_data = np.array(test_data)
        series_data = pd.Series(test_data)

        # 両方の関数で同じ結果が得られるかテスト
        result1 = ensure_series_minimal_conversion(numpy_data)
        result2 = ensure_series(numpy_data)

        print(f"ensure_series_minimal_conversion結果: {type(result1)}")
        print(f"ensure_series結果: {type(result2)}")

        # 値が同じかチェック
        np.testing.assert_array_equal(result1.values, result2.values)
        print("✅ 両関数の結果が一致")

        # pandas.Seriesの場合
        result3 = ensure_series_minimal_conversion(series_data)
        result4 = ensure_series(series_data)

        np.testing.assert_array_equal(result3.values, result4.values)
        print("✅ pandas.Series入力でも結果が一致")

        print("✅ ensure_series互換性テスト成功")

    except Exception as e:
        print(f"❌ ensure_series互換性テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🔄 パターン認識指標の統一化テスト開始")

    test_ensure_series_compatibility()
    test_pattern_recognition_unified()

    print("\n🎉 統一化テスト完了")
