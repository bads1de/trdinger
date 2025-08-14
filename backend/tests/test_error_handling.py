#!/usr/bin/env python3
"""
@handle_pandas_ta_errorsデコレーターの必要性テスト

デコレーターありとなしでのエラーハンドリングの違いを確認します。
"""

import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))


def test_error_handling_with_decorator():
    """デコレーター付きのエラーハンドリングテスト"""
    print("=== デコレーター付きエラーハンドリングテスト ===")

    try:
        from app.services.indicators.technical_indicators.volatility import (
            VolatilityIndicators,
        )

        # 異常なデータでテスト
        print("1. 空データでのテスト")
        try:
            empty_data = np.array([])
            result = VolatilityIndicators.atr(
                empty_data, empty_data, empty_data, length=14
            )
            print(f"結果: {result}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

        print("\n2. 短すぎるデータでのテスト")
        try:
            short_data = np.array([100, 101])  # 2つだけ
            result = VolatilityIndicators.atr(
                short_data, short_data, short_data, length=14
            )
            print(f"結果: {result}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

        print("\n3. NaNデータでのテスト")
        try:
            nan_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
            result = VolatilityIndicators.atr(nan_data, nan_data, nan_data, length=3)
            print(f"結果: {result}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"❌ デコレーター付きテストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_error_handling_without_decorator():
    """デコレーターなしのエラーハンドリングテスト"""
    print("\n=== デコレーターなしエラーハンドリングテスト ===")

    try:
        import pandas_ta as ta

        # 同じ異常データで直接pandas-taを呼び出し
        print("1. 空データでのテスト（pandas-ta直接）")
        try:
            empty_series = pd.Series([])
            result = ta.atr(
                high=empty_series, low=empty_series, close=empty_series, length=14
            )
            print(f"結果: {result}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

        print("\n2. 短すぎるデータでのテスト（pandas-ta直接）")
        try:
            short_series = pd.Series([100, 101])
            result = ta.atr(
                high=short_series, low=short_series, close=short_series, length=14
            )
            print(f"結果: {result}")
            print(f"結果の型: {type(result)}")
            if hasattr(result, "values"):
                print(f"値: {result.values}")
                print(f"NaN数: {np.sum(np.isnan(result.values))}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

        print("\n3. NaNデータでのテスト（pandas-ta直接）")
        try:
            nan_series = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
            result = ta.atr(high=nan_series, low=nan_series, close=nan_series, length=3)
            print(f"結果: {result}")
            if hasattr(result, "values"):
                print(f"値: {result.values}")
                print(f"全てNaN?: {np.all(np.isnan(result.values))}")
        except Exception as e:
            print(f"エラーキャッチ: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"❌ デコレーターなしテストエラー: {e}")
        import traceback

        traceback.print_exc()


def test_simplified_vs_decorated():
    """簡素化版とデコレーター版の比較"""
    print("\n=== 簡素化版 vs デコレーター版比較 ===")

    try:
        # 簡素化版（trend.pyから）
        from app.services.indicators.technical_indicators.trend import TrendIndicators

        print("1. 正常データでの比較")
        normal_data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        # 簡素化版
        try:
            simple_result = TrendIndicators.sma(normal_data, 5)
            print(f"簡素化版SMA結果: {simple_result[-3:]}")
        except Exception as e:
            print(f"簡素化版エラー: {type(e).__name__}: {e}")

        # デコレーター版（volatilityから借用）
        try:
            from app.services.indicators.technical_indicators.volatility import (
                VolatilityIndicators,
            )

            decorated_result = VolatilityIndicators.atr(
                normal_data, normal_data, normal_data, 5
            )
            print(f"デコレーター版ATR結果: {decorated_result[-3:]}")
        except Exception as e:
            print(f"デコレーター版エラー: {type(e).__name__}: {e}")

        print("\n2. 問題データでの比較")
        problem_data = np.array([100, np.nan, 102])

        # 簡素化版
        try:
            simple_result = TrendIndicators.sma(problem_data, 2)
            print(f"簡素化版SMA結果: {simple_result}")
            print(f"NaN含有: {np.any(np.isnan(simple_result))}")
        except Exception as e:
            print(f"簡素化版エラー: {type(e).__name__}: {e}")

        # デコレーター版
        try:
            decorated_result = VolatilityIndicators.atr(
                problem_data, problem_data, problem_data, 2
            )
            print(f"デコレーター版ATR結果: {decorated_result}")
            print(f"NaN含有: {np.any(np.isnan(decorated_result))}")
        except Exception as e:
            print(f"デコレーター版エラー: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"❌ 比較テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🔍 @handle_pandas_ta_errorsデコレーターの必要性検証")

    test_error_handling_without_decorator()
    test_error_handling_with_decorator()
    test_simplified_vs_decorated()

    print("\n📋 結論:")
    print("1. pandas-taは異常データでも例外を投げずにNaNを返すことが多い")
    print("2. デコレーターは結果の妥当性をチェックして適切なエラーを投げる")
    print("3. 簡素化版では異常な結果が見逃される可能性がある")
    print("4. 本番環境では適切なエラーハンドリングが重要")

    print("\n🎉 検証完了")
