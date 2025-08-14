#!/usr/bin/env python3
"""
パターン認識指標の統一化テスト
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


if __name__ == "__main__":
    print("🔄 パターン認識指標の統一化テスト開始")
    test_pattern_recognition_unified()

    print("\n🎉 統一化テスト完了")
